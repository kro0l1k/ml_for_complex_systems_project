# todo: implementation of the closed form solution for one dimensional jump-diffusion.
import time
import numpy as np
import torch
import math
from scipy.stats import multivariate_normal as normal
from collections import namedtuple
import matplotlib.pyplot as plt

LAMBDA = 0.05

# Set default tensor type to float
torch.set_default_dtype(torch.float32)

# Check for available devices 
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class ClosedFormSolver(object):
    def __init__(self,):
        self.batch_size = 100
        self.valid_size = 1000
        self.config = Config()

        
    def get_solution(self):
        """
        should implement the cloed formula from 4.1    
        """
        start_time = time.time()
        training_history = []
        validation_data = self.config.sample(self.valid_size)
        
        # compute Lambda(t) = sigma(t)**2 + mean_jump ** 2 + std_jump ** 2
        def Lambda(t):
            return torch.tensor(self.config.sigma(t)**2 + self.config.jump_intensity * self.config.jump_size_std[0] ** 2, 
                                dtype=torch.float32, device=device)
        
        terminal_time_tensor = torch.tensor(self.config.terminal_time, dtype=torch.float32, device=device)

        def phi(t): 
            # Previous: self.config.mu(t) - self.config.rho(t) **2
            # Correction: (self.config.mu(t) - self.config.rho(t)) **2
            I_phi = (self.config.mu(t) - self.config.rho(t)) **2 / Lambda(t) - 2 * self.config.rho(t) 
            I_phi = torch.tensor(I_phi, dtype=torch.float32, device=device)
            return -1.0 * torch.exp(I_phi * (terminal_time_tensor - t))
        
        def psi(t):
            # Previous: self.config.mu(t) - self.config.rho(t) **2
            # Correction: (self.config.mu(t) - self.config.rho(t)) **2
            I_psi = (self.config.mu(t) - self.config.rho(t)) **2 / Lambda(t) -  self.config.rho(t) 
            I_psi = torch.tensor(I_psi, dtype=torch.float32, device=device)
            return self.config.a_in_cost() * torch.exp(I_psi * (self.config.terminal_time - t))
        
        print(" ratio of the greeks at time ", 0.2 , "  phi(t) / psi(t), :", phi(0.2) / psi(0.2))
        print(" ratio of the greeks at time ", 0.4 , "  phi(t) / psi(t), :", phi(0.4) / psi(0.4))
        print(" ratio of the greeks at time ", 0.7 , "  phi(t) / psi(t), :", phi(0.7) / psi(0.7))
        print(" ratio of the greeks at time ", 0.9 , "  phi(t) / psi(t), :", phi(0.9) / psi(0.9))
        
        def u_star(t_1, X_BX):
            # t is a scalar, 
            # X is a tensor of shape (batch_size, dim_X)
            # output shape: (batch_size, dim_X)
            if not isinstance(t_1, torch.Tensor):
                t_1 = torch.tensor(t_1, dtype=torch.float32, device=device)
            
            # Make sure X_BX is on the same device
            X_BX = X_BX.to(device)
            
            ustar = (self.config.rho(0) - self.config.mu(0)) * (phi(t_1) * X_BX + psi(t_1)) / (phi(t_1) * Lambda(0))
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
            
        inspect_the_feedback_law()
        
        
        # get the tensor X_0 and do the forward pass with feeback control u_star
        X_BX = torch.ones(self.batch_size, self.config.dim_X).to(device)
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
                X_BX * torch.einsum('blc,blcx->bx', jump_mask[i, :, :, :], jump_sizes) + \
                - self.config.jump_intensity[0] * self.config.jump_size_mean[0] * X_BX * self.config.delta_t # NOTE: why with a minus sign? # REPLY: because this compensation term is intended to cancel out the jumps "on average" (over time and probability space)
                
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
        
        # compute the mean and the variance of the portfolio
        mean_portfolio = np.mean(S_T, axis=0)
        var_portfolio = np.var(S_T, axis=0)
        
        # compute E[ (X_T - a)^2 ]
        # E[ (X_T - a)^2 ] = E[ X_T^2 ] - 2 * a * E[ X_T ] + a^2
        # E[ X_T^2 ] = E[ S_T^2 ]
        # E[ X_T ] = E[ S_T ]
        a = self.config.a_in_cost()
        E_X_T_square = np.mean(S_T ** 2, axis=0)
        E_X_T = np.mean(S_T, axis=0)
        cost_function = E_X_T_square - 2 * a * E_X_T + a ** 2
        print("Cost function: ", cost_function)
        print("Mean of the portfolio: ", mean_portfolio)
        print("Variance of the portfolio: ", var_portfolio)
        print("Time taken: ", time.time() - start_time)
        return S, mean_portfolio, var_portfolio
        
        
        


# A simple object to contain the sample data
SampleData = namedtuple('SampleData', [
    'delta_W_TBW',    # torch.Tensor, shape (time_step_count, batch_size, dim_W)
    'jump_times_BLC', # torch.Tensor, (batch_size, dim_L, max_counts)
    'jump_mask_TBLC',  # torch.BoolTensor, (time_step_count, batch_size, dim_L, max_counts)
    'jump_sizes_BLCX'  # torch.Tensor, (batch_size, dim_L, max_counts, n)
])

class Config(object):
    """Define the configs in the systems"""
    def __init__(self):
        super(Config, self).__init__()
        self.dim_X = 1              # The integer n
        self.dim_u = 1              # The dimension of U
        self.dim_W = 1              # The integer m
        self.dim_L = 1              # The integer l 
        
        # make sure L is one, if not throw a warning
        if self.dim_L != 1:
            print('Warning: The dimension of L is not 1, which may cause problems in the code.')

        self.X_init = torch.ones(self.dim_X, dtype=torch.float32, device=device) # The initial value of X at time 0
        
        self.jump_intensity = np.array([1,  # Jumps per year in average, 
                                        ], dtype=float) # The l-dimensional vector lambda
        self.log_normal_mu = np.array([-0.05,
                                        ], dtype=float)
        self.log_normal_sigma = np.array([0.1,
                                       ], dtype=float)
        self.jump_size_mean = np.exp(self.log_normal_mu + 0.5 * self.log_normal_sigma ** 2) - 1
        self.jump_size_std = np.sqrt((np.exp(self.log_normal_sigma ** 2) - 1) * np.exp(2 * self.log_normal_mu + self.log_normal_sigma ** 2))

        # The terminal time in years
        self.terminal_time = 1.0
        # Roughly the number of trading days
        self.time_step_count = math.floor(self.terminal_time * 200)  # 20 trading days in a month. keep it small for testing.
        self.delta_t = float(self.terminal_time) / self.time_step_count

        self.MC_sample_size = 10  # The integer M
        # Generate sample points for integration with respect to nu 
        MC_sample_points_LMX = np.random.lognormal(mean=self.log_normal_mu[0], sigma=self.log_normal_sigma[0], size=(self.dim_L, self.MC_sample_size, self.dim_X)) - 1
        self.MC_sample_points_LMX = torch.tensor(MC_sample_points_LMX, dtype=torch.float32).to(device)

    def sample(self, sample_size : int):
        delta_W_TBW = np.random.normal(size=(self.time_step_count, sample_size, self.dim_W)) * np.sqrt(self.delta_t)

        # jump_counts is of shape (sample_size, dim_L)
        # Each entry in jump_counts[:,k-1] is sampled from a Poisson distribution with intensity self.jump_intensity[k-1]
        jump_counts = np.random.poisson(self.jump_intensity * self.terminal_time, size=(sample_size, self.dim_L))
        # jump_counts = np.zeros_like(jump_counts, dtype=int) 
        # print('jump_counts: ', jump_counts.shape)
        jump_times_BLC = np.random.uniform(0, self.terminal_time, size=(sample_size, self.dim_L, np.max(jump_counts)))

        # Set the excess samples of jump times to dummy value
        for sample in range(sample_size):
            for k in range(self.dim_L):
                jump_times_BLC[sample, k, jump_counts[sample, k]:] = self.terminal_time + 1.0 # Dummy value
                # jump_times_BLC[sample, k, :jump_counts[sample, k]] = np.sort(jump_times_BLC[sample, k, :jump_counts[sample, k]])

        # Create a mask which indicates whether each jump time is in the interval [i * delta_t, (i + 1) * delta_t)
        jump_mask_TBLC = np.zeros((self.time_step_count, sample_size, self.dim_L, np.max(jump_counts)), dtype=int)
        for i in range(self.time_step_count):
            jump_mask_TBLC[i, :, :, :] = (jump_times_BLC[:, :, :] >= i * self.delta_t) & (jump_times_BLC[:, :, :] < (i + 1) * self.delta_t)

        # Sample the jump sizes as normal distributions with predefined means and stds
        jump_sizes_BLCX = np.zeros((sample_size, self.dim_L, np.max(jump_counts), self.dim_X), dtype=float)
        for l in range(self.dim_L):
           jump_sizes_BLCX[:, l, :, :] = np.random.lognormal(mean=self.log_normal_mu[l], sigma=self.log_normal_sigma[l], size=(sample_size, np.max(jump_counts), self.dim_X)) - 1
            

        return SampleData(delta_W_TBW   = torch.tensor(delta_W_TBW,    dtype=torch.float32).to(device),
                          jump_times_BLC = torch.tensor(jump_times_BLC, dtype=torch.float32).to(device),
                          jump_mask_TBLC  = torch.tensor(jump_mask_TBLC,  dtype=torch.float32).to(device),
                          jump_sizes_BLCX = torch.tensor(jump_sizes_BLCX, dtype=torch.float32).to(device),)
        
    def a_in_cost(self):
        # the constant a in the equivalent problem formulation: sup E [x(T - a)**2]
        # output shape: scalar
        return 1.1

    def f(self, t, x, u):
        # Output shape: (batch_size, 1)
        return torch.zeros(x.shape[0], 1, dtype=torch.float32).to(x.device)
    
    def f_x(self, t, x, u):
        # Output shape: (batch_size, dim_X)
        return torch.zeros(x.shape[0], self.dim_X, dtype=torch.float32).to(x.device)
    
    def f_u(self, t, x, u):
        # Output shape: (batch_size, dim_u)
        return torch.zeros(x.shape[0], self.dim_u, dtype=torch.float32).to(x.device)

    def g(self, x): # - 0.5 * (x - 1.1)^2
        # Output shape: (batch_size, 1)
        return - 0.5 * torch.sum((x - self.a_in_cost()) ** 2, dim=1, keepdim=True)

    def g_x(self, x):
        # Output shape: (batch_size, dim_X)
        return - x + self.a_in_cost()
    
    def rho(self, t): # Risk-free interest rate
        # Output shape: scalar
        return 0.05
    
    def mu(self, t): # Expected return of the stock
        # Output shape: scalar
        return 0.1
    
    def sigma(self, t): # Volatility of the stock
        # Output shape: scalar
        return 0.2

    def drift(self, t, x, u):
        # Output shape: (batch_size, dim_X)
        # b(t, x, u) = rho(t) * x + (mu(t) - rho(t)) * u
        return self.rho(t) * x + (self.mu(t) - self.rho(t)) * u
    
    def drift_x(self, t, x, u): 
        # Drift coefficient in front of dt
        # Partial derivatives of each component of drift coefficient with respect to x
        # Output shape: (batch_size, dim_X, dim_X)
        # drift_x(t, x, u)[b, j, :] = \partial drift^j(t, x, u) / \partial x
        return self.rho(t) * torch.ones(x.shape[0], self.dim_X, self.dim_X, dtype=torch.float32).to(x.device)
    
    def drift_u(self, t, x, u):
        # Partial derivatives of each component of drift coefficient with respect to u
        # Output shape: (batch_size, dim_X, dim_u)
        # drift_u(t, x, u)[b, j, :] = \partial drift^j(t, x, u) / \partial u
        return (self.mu(t) - self.rho(t)) * torch.ones(x.shape[0], self.dim_X, self.dim_u, dtype=torch.float32).to(x.device)

    def diffusion(self, t, x, u):
        # Diffusion coefficient in front of dW_t
        # Input shape of u: (batch_size, dim_X)
        # Output shape: (batch_size, dim_X, dim_W)
        # diffusion(t) = 0.2 * u
        return self.sigma(t) * u.unsqueeze(2).repeat(1, 1, self.dim_W)
    
    def diffusion_x(self, t, x, u): 
        # Partial derivatives of each component of diffusion coefficient with respect to x
        # Input shape of u: (batch_size, dim_X)
        # Output shape: (batch_size, dim_X, dim_W, dim_X)
        # diffusion_x(t, x, u)[b, j, k, :] = \partial diffusion^{j,k}(t, x, u) / \partial x
        return torch.zeros(x.shape[0], self.dim_X, self.dim_W, self.dim_X, dtype=torch.float32).to(x.device)
    
    def diffusion_u(self, t, x, u):
        # Partial derivatives of each component of diffusion coefficient with respect to u
        # Input shape of u: (batch_size, dim_X)
        # Output shape: (batch_size, dim_X, dim_W, dim_u)
        # diffusion_u(t, x, u)[b, j, k, :] = \partial diffusion^{j,k}(t, x, u) / \partial u
        return self.sigma(t) * torch.ones(x.shape[0], self.dim_X, self.dim_W, self.dim_u, dtype=torch.float32).to(x.device)
    
    def eta(self, t, x, u, z):
        # Shape of u: (batch_size, dim_X)
        # Shape of z: (batch_size, dim_L)
        # eta(u, z) = u * transpose(z)
        # Output shape: (batch_size, dim_X, dim_L)
        return u.unsqueeze(2) * z.unsqueeze(1)
    
    def eta_x(self, t, x, u, z):
        # Partial derivatives of each component of eta with respect to x
        # Output shape: (batch_size, dim_X, dim_L, dim_X)
        # eta_x(t, x, u, z)[b, j, k, :] = \partial eta^{j,k}(t, x, u, z) / \partial x
        return torch.zeros(x.shape[0], self.dim_X, self.dim_L, self.dim_X, dtype=torch.float32).to(x.device)
    
    def eta_u(self, t, x, u, z):
        # Partial derivatives of each component of eta with respect to u
        # Output shape: (batch_size, dim_X, dim_L, dim_u)
        # eta_u(t, x, u, z)[b, j, k, :] = \partial eta^{j,k}(t, x, u, z) / \partial u
        return z.unsqueeze(1).repeat(1, self.dim_X, 1, 1)

    def sample_stock_price(self, sample_size):
        # Generate dS_t = S_t * (mu(t) * dt + sigma(t) * dW_t + \int_{\R} eta(t,z) * N(dt, dz))
        sample_data = self.sample(sample_size)
        t = torch.linspace(0, self.terminal_time, self.time_step_count + 1, dtype=torch.float32, device=device)

        # Initialize S_t, which will be used to record the stock price at each time step
        S = np.zeros((self.time_step_count + 1, sample_size, self.dim_X), dtype=float)
        
        # Create a copy of X_init
        X = self.X_init.repeat(sample_size, 1)  # Shape: (sample_size, dim_X)
        S[0,:,:] = X.detach().cpu().numpy()  # Initial stock price

        for i in range(self.time_step_count):
                X = X + self.drift(t[i], X, X) * self.delta_t + \
                torch.einsum('bxw,bw->bx', self.diffusion(t[i], X, X), sample_data.delta_W_TBW[i, :, :]) + \
                X * torch.einsum('blc,blcx->bx', sample_data.jump_mask_TBLC[i, :, :, :], sample_data.jump_sizes_BLCX) + \
                - self.jump_intensity[0] * self.jump_size_mean[0] * X * self.delta_t
                # S_t[i, :] * sample_data.jump_mask_TBLC[i, 0, :, :] * sample_data.jump_sizes_BLCX[0, :, :, :] * sample_data.jump_mask_TBLC[i, 0, :, :].unsqueeze(2)
                
                S[i+1,:,:] = X.detach().cpu().numpy()
            
        # Plot the stock price
        t = t.detach().cpu().numpy()
        for b in range(sample_size):
            plt.plot(t, S[:, b, 0])
        plt.title('Stock Price Simulation')
        plt.xlabel('t')
        plt.ylabel('$S_1(t)$')
        plt.grid()
        plt.show()
        return S

def main():
    solver = ClosedFormSolver()
    
    print('solving the closed form solution')
    solver.get_solution()

if __name__ == '__main__':
    main()