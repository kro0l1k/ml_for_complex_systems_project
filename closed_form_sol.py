# todo: implementation of the closed form solution for one dimensional jump-diffusion.
import time
import numpy as np
import torch
import math
from scipy.stats import multivariate_normal as normal
from collections import namedtuple
import matplotlib.pyplot as plt
from pres_config_file import Config

LAMBDA = 0.05

# Set default tensor type to float
torch.set_default_dtype(torch.float32)

# Check for available devices 
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class ClosedFormSolver(object):
    def __init__(self, x_0_value=1.0):
        self.config = Config()
        self.x_0_value = x_0_value
          
        self.valid_size = self.config.valid_size
        self.batch_size = self.config.batch_size
        self.num_iterations = self.config.num_iterations
        self.logging_frequency = self.config.logging_frequency
        self.lr_values = self.config.lr_values
        self.lr_boundaries = self.config.lr_boundaries
        self.x_0_value = x_0_value

    def get_solution(self):
        """
        should implement the cloed formula from 4.1
        """
        start_time = time.time()
        training_history = []
        validation_data = self.config.sample(self.batch_size)
        
        # compute Lambda(t) = sigma(t)**2 + mean_jump ** 2 + std_jump ** 2
        def Lambda(t):
            return torch.tensor(self.config.sigma(t)**2 + self.config.jump_intensity * self.config.jump_size_mean[0] ** 2 + self.config.jump_intensity * self.config.jump_size_std[0] ** 2, 
                                dtype=torch.float32, device=device)
        
        terminal_time_tensor = torch.tensor(self.config.terminal_time, dtype=torch.float32, device=device)
        print("Lambda(0): ", Lambda(0).item())
        print("terminal_time_tensor: ", terminal_time_tensor.item())
        
        def phi(t): 
            I_phi = (self.config.mu(t) - self.config.rho(t)) **2 / Lambda(t) - 2 * self.config.rho(t) 
            I_phi = torch.tensor(I_phi, dtype=torch.float32, device=device)
            return -1.0 * torch.exp(I_phi * (terminal_time_tensor - t))
        
        def psi(t):
            I_psi = (self.config.mu(t) - self.config.rho(t)) **2 / Lambda(t) - self.config.rho(t) 
            I_psi = torch.tensor(I_psi, dtype=torch.float32, device=device)
            return self.config.a_in_cost() * torch.exp(I_psi * (self.config.terminal_time - t))
        
        # NOTE: constants to compute the feedback law
        # print(" ratio of the greeks at time ", 0.2 , "  phi(t) / psi(t), :", phi(0.2) / psi(0.2))
        # print(" ratio of the greeks at time ", 0.4 , "  phi(t) / psi(t), :", phi(0.4) / psi(0.4))
        # print(" ratio of the greeks at time ", 0.7 , "  phi(t) / psi(t), :", phi(0.7) / psi(0.7))
        # print(" ratio of the greeks at time ", 0.9 , "  phi(t) / psi(t), :", phi(0.9) / psi(0.9))
        
        def u_star(t_1, X_BX):
            # t is a scalar, 
            # X is a tensor of shape (batch_size, dim_X)
            # output shape: (batch_size, dim_X)
            if not isinstance(t_1, torch.Tensor):
                t_1 = torch.tensor(t_1, dtype=torch.float32, device=device)
            # the mistake was here!
            ustar = (self.config.rho(t_1) - self.config.mu(t_1)) * (phi(t_1) * X_BX + psi(t_1)) / (phi(t_1) * Lambda(t_1)) 
            return ustar
        
        def inspect_the_feedback_law():
            # Inspect the feedback law u_star(t, X_BX)
            # Define time steps and x values to inspect
            time_steps = [0, 0.25, 0.5, 0.75, 1]
            x_values = [-0.5, 0, 0.25, 0.5, 1.0, 1.25, 1.4,  1.5, 1.6, 1.75, 2.0, 2.5,5,10]

            # Prepare data for plotting
            u_star_values = []
            for t in time_steps:
                u_star_at_t = []
                for x in x_values:
                    X_BX = torch.full((1, self.config.dim_X), x, dtype=torch.float32, device=device)
                    
                    # print(" ustar shape: ", u_star(t, X_BX).shape)
                    u_star_at_t.append(u_star(t, X_BX).detach().cpu().numpy()[0,0])
                    u_star_values.append(u_star_at_t)

            # Plot the feedback law
            fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
            for i, ax in enumerate(axes):
                ax.plot(x_values, u_star_values[i])
                ax.set_title(f"u_star at t = {time_steps[i]}")
                ax.set_xlabel("x values")
                ax.set_ylabel("u_star values")
                ax.grid()

            plt.tight_layout()
            plt.show()
            
        # inspect_the_feedback_law() # NOTE: uncomment to see V(t,x) and u_star(t,x)
        
        
        # get the tensor X_0 and do the forward pass with feeback control u_star
        X_BX = torch.ones(self.batch_size, self.config.dim_X).to(device) * self.x_0_value # NOTE: now with the added x_0_value
        t = torch.linspace(0, self.config.terminal_time, self.config.time_step_count + 1, dtype=torch.float32, device=device)
        S = np.zeros((self.config.time_step_count + 1, self.batch_size, self.config.dim_X), dtype=float)
        S[0,:,:] = X_BX.detach().cpu().numpy()  # Initial stock price
        
        # Take only the first batch_size samples from validation_data
        delta_W = validation_data.delta_W_TBW[:, :self.batch_size, :]
        jump_mask = validation_data.jump_mask_TBLC[:, :self.batch_size, :, :]
        jump_sizes = validation_data.jump_sizes_BLCX[:self.batch_size, :, :, :]
        
        for i in range(self.config.time_step_count):
                X_BX = X_BX + self.config.drift(t[i], X_BX, u_star(t[i], X_BX)) * self.config.delta_t + \
                torch.einsum('bxw,bw->bx', self.config.diffusion(t[i], X_BX, u_star(t[i], X_BX)), delta_W[i, :, :]) + \
                u_star(t[i], X_BX) * torch.einsum('blc,blcx->bx', jump_mask[i, :, :, :], jump_sizes) + \
                - self.config.jump_intensity[0] * self.config.jump_size_mean[0] * u_star(t[i], X_BX) * self.config.delta_t # NOTE: why with a minus sign? # REPLY: because this compensation term is intended to cancel out the jumps "on average" (over time and probability space)
                
                S[i+1,:,:] = X_BX.detach().cpu().numpy()
        
        # Plot the wealth plot
        t = t.detach().cpu().numpy()
        for b in range(self.batch_size):
            plt.plot(t, S[:, b, 0])
        plt.title('Wealth Simulation')
        plt.xlabel('t')
        plt.ylabel('$S_1(t)$')
        plt.grid()
        plt.show()
        
        # get the final value of the portfolio
        S_T = S[-1, :, :]
        # compute the cost functional (S_T - TARGET_MEAN_A)**2 for each of the samples
        cost_functional = (S_T - self.config.TARGET_MEAN_A)**2
        # print("Cost functional: ", cost_functional)
        mean_const_functional = np.mean(cost_functional)
        std_const_functional = np.std(cost_functional)
        print("Time taken: ", time.time() - start_time)
        return S, mean_const_functional, std_const_functional


def main():
    
    x_0_values = np.array([70, 80, 90, 100, 110, 120])
    V_for_different_x0 = []
    std_for_different_x0 = []

    for x_0 in x_0_values:
        print('solving the closed form solution for x_0 = ', x_0)

        solver = ClosedFormSolver(x_0_value=x_0)
        S, mean_const_functional, std_const_functional = solver.get_solution()
        V_for_different_x0.append(mean_const_functional)
        std_for_different_x0.append(std_const_functional)

    print("x_0 values: \n", x_0_values)
    print("Mean Cost Functional for different x_0 values: \n", V_for_different_x0)
    ### plot the cost functional for different x_0 values. add a transparent area for +-1 std
    plt.figure()
    plt.plot(x_0_values, V_for_different_x0, label='Mean Cost Functional')
    plt.fill_between(x_0_values, 
                     np.array(V_for_different_x0) - np.array(std_for_different_x0), 
                     np.array(V_for_different_x0) + np.array(std_for_different_x0), 
                     alpha=0.2, label='1 Std Dev')
    plt.title('Cost Functional for Different Initial Values of X0')
    plt.xlabel('Initial Value of X0')
    plt.ylabel('Cost Functional')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()