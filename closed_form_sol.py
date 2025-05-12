# todo: impleement closed form solution for the meaan variance portfolio optimization problem


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
        self.valid_size = 200
        self.batch_size = 100
        self.config = Config()

        
    def get_solution(self):
        """
        should implement the cloed formula from 4.1
        
        
        """
        start_time = time.time()
        training_history = []
        validation_data = self.config.sample(self.valid_size)
        
        # compute Lambda(t) = sigma(t)**2 + mean_jump ** 2 + std_jump ** 2
        Lambda = self.config.sigma_stock(0)**2 + self.config.jump_size_mean[0] ** 2 + self.config.jump_size_std[0] ** 2 
        print(" to compute the term Lambda(t) we are using: ", Lambda, self.config.sigma_stock(0)**2 , 
              self.config.jump_size_mean[0] ** 2 ,  self.config.jump_size_std[0] ** 2)
        
        # compute the integrands: 
        I_phi = self.config.mu(0) - self.config.rho(0) **2 / Lambda - 2 * self.config.rho(0) 
        I_psi = self.config.mu(0) - self.config.rho(0) **2 / Lambda -  self.config.rho(0) 
        
        # Convert to PyTorch tensors
        I_phi_tensor = torch.tensor(I_phi, dtype=torch.float32, device=device)
        I_psi_tensor = torch.tensor(I_psi, dtype=torch.float32, device=device)
        terminal_time_tensor = torch.tensor(self.config.terminal_time, dtype=torch.float32, device=device)
        
        print(" self.config.rho(0) ", self.config.rho(0) )
        print("I_phi: ", I_phi)
        print("I_psi: ", I_psi)

        
        def phi(t): 
            # Ensure t is a tensor
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t, dtype=torch.float32, device=device)
            return -1.0 * torch.exp(I_phi_tensor * (terminal_time_tensor - t))
        
        def psi(t):
            # Ensure t is a tensor
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t, dtype=torch.float32, device=device)
            a = self.config.a_in_const_functional()
            return a * torch.exp(I_psi_tensor * (terminal_time_tensor - t))
        
        print("phi(0): ", phi(0))
        print("psi(0): ", psi(0))
        print("phi(1): ", phi(1))
        print("psi(1): ", psi(1))
        
        def u_star(t_1, X_BX):
            # t is a scalar, 
            # X is a tensor of shape (batch_size, dim_X)
            # output shape: (batch_size, dim_X)
            if not isinstance(t_1, torch.Tensor):
                t_1 = torch.tensor(t_1, dtype=torch.float32, device=device)
            
            # Make sure X_BX is on the same device
            X_BX = X_BX.to(device)
            
            ustar = (self.config.rho(0) - self.config.mu(0)) * (phi(t_1) * X_BX + psi(t_1)) / (phi(t_1) * Lambda)
            return ustar
        
        print("u_star(0): ", u_star(0, torch.ones(self.batch_size, self.config.dim_X).to(device))[:5] )
        print("u_star(1): ", u_star(1, torch.ones(self.batch_size, self.config.dim_X).to(device))[:5] )
        print("u_star(0.5): ", u_star(0.5, torch.ones(self.batch_size, self.config.dim_X).to(device))[:5] )
        print("u_star(0): ", u_star(0, torch.ones(self.batch_size, self.config.dim_X).to(device))[:5])
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
                X_BX = X_BX + self.config.b(t[i], X_BX, u_star(t[i], X_BX)) * self.config.delta_t + \
                torch.einsum('bxw,bw->bx', self.config.sigma(t[i], X_BX, u_star(t[i], X_BX)), delta_W[i, :, :]) + \
                X_BX * torch.einsum('blc,blcx->bx', jump_mask[i, :, :, :], jump_sizes) + \
                - self.config.jump_intensity[0] * self.config.jump_size_mean[0] * X_BX * self.config.delta_t
                
                S[i+1,:,:] = X_BX.detach().cpu().numpy()
        
        # Plot the stock price
        t = t.detach().cpu().numpy()
        for b in range(self.batch_size):
            plt.plot(t, S[:, b, 0])
        plt.title('Stock Price Simulation')
        plt.xlabel('t')
        plt.ylabel('$S_1(t)$')
        plt.grid()
        plt.show()
        
        
        


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
        self.dim_X = 2              # The integer n
        self.dim_u = 2              # The dimension of U
        self.dim_W = 2              # The integer m
        self.dim_L = 1              # The integer l 
        
        if self.dim_L > 1:
            print('Warning: dim_L > 1, which is not supported yet. Please set dim_L = 1.')

        self.X_init = torch.ones(self.dim_X, dtype=torch.float32, device=device) # The initial value of X at time 0
        
        self.jump_intensity = np.array([5,  # In average, 5 jumps per year
                                        ], dtype=float) # The l-dimensional vector lambda
        self.jump_size_mean = np.array([-0.02,
                                        ], dtype=float)
        self.jump_size_std = np.array([0.05,
                                       ], dtype=float)

        # The terminal time in years
        self.terminal_time = 1
        # Roughly the number of trading days
        self.time_step_count = math.floor(self.terminal_time * 200)  # 20 trading days in a month. keep it small for testing.
        self.delta_t = float(self.terminal_time) / self.time_step_count

        self.MC_sample_size = 1000  # The integer M
        # Generate sample points for integration with respect to nu 
        MC_sample_points_LMX = np.random.normal(loc=-0.02, scale=0.05, size=(self.dim_L, self.MC_sample_size, self.dim_X))
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
           jump_sizes_BLCX[:, l, :, :] = np.random.normal(loc=self.jump_size_mean[l], scale=self.jump_size_std[l], size=(sample_size, np.max(jump_counts), self.dim_X))
            

        return SampleData(delta_W_TBW   = torch.tensor(delta_W_TBW,    dtype=torch.float32).to(device),
                          jump_times_BLC = torch.tensor(jump_times_BLC, dtype=torch.float32).to(device),
                          jump_mask_TBLC  = torch.tensor(jump_mask_TBLC,  dtype=torch.float32).to(device),
                          jump_sizes_BLCX = torch.tensor(jump_sizes_BLCX, dtype=torch.float32).to(device),)
        
    def a_in_const_functional(self):
        # the constant a in the equivalent problem formulation: sup E [x(T - a)**2]
        # output shape: scalar
        return 1. #TODO: check if changing this changes anything.

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
        return - 0.5 * torch.sum((x - 1.1) ** 2, dim=1, keepdim=True)

    def g_x(self, x):
        # Output shape: (batch_size, dim_X)
        return - x + 1.1
    
    def rho(self, t): # Risk-free interest rate
        # Output shape: scalar
        return 0.05
    
    def mu(self, t): # Expected return of the stock
        # Output shape: scalar
        return 0.1
    
    def sigma_stock(self, t): # Volatility of the stock
        # Output shape: scalar
        return 0.2

    def b(self, t, x, u):
        # Output shape: (batch_size, dim_X)
        # b(t, x, u) = rho(t) * x + (mu(t) - rho(t)) * u
        return self.rho(t) * x + (self.mu(t) - self.rho(t)) * u
    
    def b_x(self, t, x, u): 
        # Partial derivatives of each component of b with respect to x
        # Output shape: (batch_size, dim_X, dim_X)
        # b_x(t, x, u)[b, j, :] = \partial b^j(t, x, u) / \partial x
        return self.rho(t) * torch.ones(x.shape[0], self.dim_X, self.dim_X, dtype=torch.float32).to(x.device)
    
    def b_u(self, t, x, u):
        # Partial derivatives of each component of b with respect to u
        # Output shape: (batch_size, dim_X, dim_u)
        # b_u(t, x, u)[b, j, :] = \partial b^j(t, x, u) / \partial u
        return (self.mu(t) - self.rho(t)) * torch.ones(x.shape[0], self.dim_X, self.dim_u, dtype=torch.float32).to(x.device)
    

    def sigma(self, t, x, u): # NOTE: should this not be named "drift term or sth like that? "
        # Input shape of u: (batch_size, dim_X)
        # Output shape: (batch_size, dim_X, dim_W)
        # sigma(t) = 0.2 * u
        return 0.2 * u.unsqueeze(2).repeat(1, 1, self.dim_W)
    
    def sigma_x(self, t, x, u): 
        # Partial derivatives of each component of sigma with respect to x
        # Input shape of u: (batch_size, dim_X)
        # Output shape: (batch_size, dim_X, dim_W, dim_X)
        # sigma_x(t, x, u)[b, j, k, :] = \partial sigma^{j,k}(t, x, u) / \partial x
        return torch.zeros(x.shape[0], self.dim_X, self.dim_W, self.dim_X, dtype=torch.float32).to(x.device)
    
    def sigma_u(self, t, x, u):
        # Partial derivatives of each component of sigma with respect to u
        # Input shape of u: (batch_size, dim_X)
        # Output shape: (batch_size, dim_X, dim_W, dim_u)
        # sigma_u(t, x, u)[b, j, k, :] = \partial sigma^{j,k}(t, x, u) / \partial u
        return 0.2 * torch.ones(x.shape[0], self.dim_X, self.dim_W, self.dim_u, dtype=torch.float32).to(x.device)
    
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
                X = X + self.b(t[i], X, X) * self.delta_t + \
                torch.einsum('bxw,bw->bx', self.sigma(t[i], X, X), sample_data.delta_W_TBW[i, :, :]) + \
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
    config = Config()
    config.sample_stock_price(sample_size=100)
    print('sample data generated!')
    
    solver = ClosedFormSolver()
    
    print('solving the closed form solution')
    solver.get_solution()

if __name__ == '__main__':
    main()