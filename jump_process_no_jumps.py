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

class Solver(object):
    def __init__(self,):
        self.valid_size = 200
        self.batch_size = 100
        self.num_iterations = 5000
        self.logging_frequency = 200
        self.lr_values = [5e-3, 5e-3, 5e-3]
        
        self.lr_boundaries = [2000, 4000]
        self.config = Config()

        self.model = WholeNet().to(device)  # Move model to the selected device
        print("y_intial: ", self.model.p_init.detach().cpu().numpy())
        
        # PyTorch doesn't have PiecewiseConstantDecay directly, so we'll use a custom scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_values[0], eps=1e-8)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, 
            milestones=self.lr_boundaries, 
            gamma=1.0  # No decrease since we manually set the learning rates
        )
        
        # Custom adjustment for learning rates at boundaries
        self.original_lr = self.lr_values[0]

    def train(self):
        """Training the model"""
        start_time = time.time()
        training_history = []
        validation_data = self.config.sample(self.valid_size)
        
        for step in range(self.num_iterations+1):
            # Custom learning rate adjustment
            if step in self.lr_boundaries:
                idx = self.lr_boundaries.index(step) + 1
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr_values[idx]
            
            # Calculate validation loss and log it
            if step % self.logging_frequency == 0:
                with torch.no_grad():
                    loss, performance, ratio = self.model(validation_data)
                    p_init_euclidean = np.linalg.norm(self.model.p_init.detach().cpu().numpy())
                    elapsed_time = time.time() - start_time
                    training_history.append([step, performance.item(), p_init_euclidean, loss.item(), ratio.item()])
                    print(f"step: {step:5d}, loss: {loss.item():.4e}, ||Y0||: {p_init_euclidean:.4e}, performance: {performance.item():.4e}, "
                          f"elapsed time: {int(elapsed_time):3d}, ratio: {ratio.item():.4e}")
            
            # Gradient descent
            self.optimizer.zero_grad()
            loss, _, _ = self.model(self.config.sample(self.batch_size))
            loss.backward()
            self.optimizer.step()

            self.scheduler.step()
        
        self.training_history = training_history

class WholeNet(torch.nn.Module):
    """Building the neural network architecture"""
    def __init__(self):
        super(WholeNet, self).__init__()
        self.config = Config()
        # Initialize p_init as a parameter
        self.p_init = torch.nn.Parameter(
            torch.randn(1, self.config.dim_X, dtype=torch.float32).to(device),  # Ensure float32 and correct device
            requires_grad=True
        )
        self.q_net = FNNetQ()
        self.u_net = FNNetU()
        self.r_nets = torch.nn.ModuleList()
        for _ in range(self.config.dim_L):
            self.r_nets.append(FNNetR())

    def forward(self, sample, training=True):
        delta_W_TBW, jump_times_BLC, jump_mask_TBLC, jump_sizes_BLCX = sample
        sample_size = delta_W_TBW.shape[1]
        jump_counts = jump_mask_TBLC.shape[3]
        t = np.arange(0, self.config.time_step_count) * self.config.delta_t 
        X_BX = self.config.X_init.repeat(sample_size, 1) # Shape: (sample_size, dim_X)
        p_BX = self.p_init.repeat(sample_size, 1) # Shape: (sample_size, dim_X)
        r_jump_BLCX = torch.zeros(sample_size, self.config.dim_L, jump_counts, self.config.dim_X, dtype=torch.float32).to(device) # Shape: (sample_size, dim_L, jump_counts, dim_X)
        r_monte_BLMX = torch.zeros(sample_size, self.config.dim_L, self.config.MC_sample_size, self.config.dim_X, dtype=torch.float32).to(device) # Shape: (sample_size, dim_L, MC_sample_size, dim_X)
        int_Hu = 0.0  # The constraint term
        MC_sample_points_BLMX = self.config.MC_sample_points_LMX.repeat(sample_size, 1, 1, 1) # Shape: (sample_size, dim_L, MC_sample_size, dim_X)
        
        for i in range(0, self.config.time_step_count):
            u_BU = self.u_net((t[i], X_BX))   # Shape: (sample_size, dim_u)
            q_BXW = self.q_net((t[i], X_BX))   # Shape: (sample_size, dim_X, dim_W)
            for l in range(self.config.dim_L):
                # Calculate the r_nets for each jump
                r_jump_BLCX[:, l, :, :] = self.r_nets[l]((t[i], X_BX, jump_sizes_BLCX[:, l, :, :]))
                r_monte_BLMX[:, l, :, :] = self.r_nets[l]((t[i], X_BX, MC_sample_points_BLMX[:, l, :, :]))
            
            
            Hu = torch.einsum('bjn,bj->bn', self.config.b_u(t[i], X_BX, u_BU), p_BX) + \
            torch.einsum('bjkn,bjk->bn', self.config.sigma_u(t[i], X_BX, u_BU), q_BXW)
            for l in range(self.config.dim_L):
                Hu = Hu + self.config.jump_intensity[l] * torch.einsum('bc,bcx->bx', jump_mask_TBLC[i, :, l, :], jump_sizes_BLCX[:, l, :, :] * r_jump_BLCX[:, l, :, :])

            int_Hu = int_Hu + Hu**2 # NOTE: this is where the square was missing.

            # Update p
            Hx = torch.einsum('bjn,bj->bn', self.config.b_x(t[i], X_BX, u_BU), p_BX)
            p_BX = p_BX - Hx + torch.einsum('bxw,bw->bx', q_BXW, delta_W_TBW[i, :, :])
            for l in range(self.config.dim_L):
                p_BX = p_BX + torch.einsum('bc,bcx->bx', jump_mask_TBLC[i, :, l, :], r_jump_BLCX[:, l, :, :])
                p_BX = p_BX - self.config.jump_intensity[l] * torch.mean(r_monte_BLMX[:, l, :, :], dim=1) * self.config.delta_t

            # Update X
            X_BX = X_BX + self.config.b(t[i], X_BX, u_BU) * self.config.delta_t + \
                X_BX * torch.einsum('bxw,bw->bx', self.config.sigma(t[i], X_BX, u_BU), delta_W_TBW[i, :, :])
            for l in range(self.config.dim_L):
                X_BX = X_BX + u_BU * torch.einsum('bc,bcx->bx', jump_mask_TBLC[i, :, l, :], jump_sizes_BLCX[:, l, :, :])
                X_BX = X_BX - self.config.jump_intensity[l] * self.config.jump_size_mean[l] * u_BU * self.config.delta_t

        terminal_value_loss = p_BX - self.config.g_x(X_BX)
        loss = torch.mean(torch.sum(terminal_value_loss**2, 1, keepdim=True) + LAMBDA * int_Hu)
        ratio = torch.mean(torch.sum(terminal_value_loss**2, 1, keepdim=True)) / torch.mean(int_Hu)

        performance = torch.mean(self.config.g(X_BX))

        return loss, performance, ratio

class FNNetQ(torch.nn.Module):
    """ Define the feedforward neural network """
    def __init__(self):
        super(FNNetQ, self).__init__()
        self.config = Config()
        num_hiddens = [self.config.dim_X + 10, 
                       self.config.dim_X * 2 + 10, 
                       self.config.dim_X * self.config.dim_W + 10]
        
        # Create layer lists 
        self.bn_layers = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(
                1 + self.config.dim_X,  # Shape of input (t, X)
                momentum=0.99,
                eps=1e-6
            )
        ])
        
        self.dense_layers = torch.nn.ModuleList()
        
        # Hidden layers
        for i in range(len(num_hiddens)):
            input_size = 1 + self.config.dim_X if i == 0 else num_hiddens[i-1]
            self.dense_layers.append(
                torch.nn.Linear(input_size, num_hiddens[i], bias=False)
            )
            self.bn_layers.append(
                torch.nn.BatchNorm1d(
                    num_hiddens[i],
                    momentum=0.99,
                    eps=1e-6
                )
            )
        
        # Output layer: output dim_X * dim_W values per sample
        self.dense_layers.append(
            torch.nn.Linear(num_hiddens[-1], self.config.dim_X * self.config.dim_W)
        )
        self.bn_layers.append(
            torch.nn.BatchNorm1d(
                self.config.dim_X * self.config.dim_W,
                momentum=0.99,
                eps=1e-6
            )
        )
        
        # Initialize weights similar to TF
        for i, module in enumerate(self.modules()):
            if isinstance(module, torch.nn.BatchNorm1d):
                torch.nn.init.normal_(module.weight, 0.3, 0.2)
                torch.nn.init.normal_(module.bias, 0.0, 0.1)

    def forward(self, inputs):
        """structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense -> bn"""
        t, x = inputs
        ts = torch.ones(x.shape[0], 1, dtype=torch.float32).to(x.device) * t  # Ensure correct shape and device
        x = torch.cat([ts, x], dim=1)  # Concatenate along the feature dimension
        x = self.bn_layers[0](x)
        
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = self.bn_layers[i+1](x)
            x = torch.nn.functional.relu(x)
        
        x = self.dense_layers[-1](x)
        x = self.bn_layers[-1](x)
        # Reshape to (batch_size, dim_X, dim_W)
        x = x.view(x.shape[0], self.config.dim_X, self.config.dim_W)
        return x

class FNNetU(torch.nn.Module):
    """ Define the feedforward neural network """
    def __init__(self):
        super(FNNetU, self).__init__()
        self.config = Config()
        num_hiddens = [self.config.dim_X+10, self.config.dim_X+10, self.config.dim_X+10]
        
        # Create layer lists 
        self.bn_layers = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(
                1 + self.config.dim_X,  # Shape of input (t, X)
                momentum=0.99,
                eps=1e-6
            )
        ])
        
        self.dense_layers = torch.nn.ModuleList()
        
        # Hidden layers
        for i in range(len(num_hiddens)):
            # Input size for the first layer
            input_size = 1 + self.config.dim_X if i == 0 else num_hiddens[i-1]
            
            self.dense_layers.append(
                torch.nn.Linear(input_size, num_hiddens[i], bias=False)
            )
            
            self.bn_layers.append(
                torch.nn.BatchNorm1d(
                    num_hiddens[i],
                    momentum=0.99,
                    eps=1e-6
                )
            )
        
        # Output layer
        self.dense_layers.append(
            torch.nn.Linear(num_hiddens[-1], self.config.dim_u)
        )
        
        self.bn_layers.append(
            torch.nn.BatchNorm1d(
                self.config.dim_u,
                momentum=0.99,
                eps=1e-6
            )
        )
        
        # Initialize weights similar to TF
        for i, module in enumerate(self.modules()):
            if isinstance(module, torch.nn.BatchNorm1d):
                torch.nn.init.normal_(module.weight, 0.3, 0.2)
                torch.nn.init.normal_(module.bias, 0.0, 0.1)

    def forward(self, inputs):
        """structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense -> bn"""
        t, x = inputs
        ts = torch.ones(x.shape[0], 1, dtype=torch.float32).to(x.device) * t  # Ensure correct shape and device
        x = torch.cat([ts, x], dim=1)  # Concatenate along the feature dimension
        x = self.bn_layers[0](x)
        
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = self.bn_layers[i+1](x)
            x = torch.nn.functional.relu(x)
        
        x = self.dense_layers[-1](x)
        x = self.bn_layers[-1](x)
        return x
    
class FNNetR(torch.nn.Module):
    """ Define the feedforward neural network """
    def __init__(self):
        super(FNNetR, self).__init__()
        self.config = Config()
        num_hiddens = [self.config.dim_X * 2 + 10, 
                       self.config.dim_X * 4 + 10, 
                       self.config.dim_X * 2 + 10]
        
        # Create layer lists 
        self.bn_layers = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(
                1 + self.config.dim_X * 2,  # Shape of input (t, X, z)
                momentum=0.99,
                eps=1e-6
            )
        ])
        
        self.dense_layers = torch.nn.ModuleList()
        
        # Hidden layers
        for i in range(len(num_hiddens)):
            input_size = 1 + self.config.dim_X * 2 if i == 0 else num_hiddens[i-1]
            self.dense_layers.append(
                torch.nn.Linear(input_size, num_hiddens[i], bias=False)
            )
            self.bn_layers.append(
                torch.nn.BatchNorm1d(
                    num_hiddens[i],
                    momentum=0.99,
                    eps=1e-6
                )
            )
        
        # Output layer: output dim_X values per sample
        self.dense_layers.append(
            torch.nn.Linear(num_hiddens[-1], self.config.dim_X)
        )
        self.bn_layers.append(
            torch.nn.BatchNorm1d(
                self.config.dim_X,
                momentum=0.99,
                eps=1e-6
            )
        )
        
        # Initialize weights similar to TF
        for i, module in enumerate(self.modules()):
            if isinstance(module, torch.nn.BatchNorm1d):
                torch.nn.init.normal_(module.weight, 0.3, 0.2)
                torch.nn.init.normal_(module.bias, 0.0, 0.1)

    def forward(self, inputs):
        """structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense -> bn"""
        t, x, z = inputs
        # Shape of z: (batch_size, jump_counts, dim_X)

        # Reshape t: (1) -> (batch_size, jump_counts, 1)
        t_ = torch.full((z.shape[0], z.shape[1], 1), t, dtype=torch.float32, device=device)  

        # Reshape x: (batch_size, dim_X) -> (batch_size, jump_counts, dim_X)
        x_ = x.unsqueeze(1).expand(-1, z.shape[1], -1)  

        # Shape of inputs: (batch_size, jump_counts, dim_X * 2 + 1)
        inputs = torch.cat([t_, x_, z], dim=2)

        # Reshape to (batch_size * np.max(jump_counts), dim_X * 2 + 1)
        inputs = inputs.view(-1, inputs.shape[2])

        inputs = self.bn_layers[0](inputs)
        
        for i in range(len(self.dense_layers) - 1):
            inputs = self.dense_layers[i](inputs)
            inputs = self.bn_layers[i+1](inputs)
            inputs = torch.nn.functional.relu(inputs)
        
        inputs = self.dense_layers[-1](inputs)
        inputs = self.bn_layers[-1](inputs)
        # Reshape to (batch_size, jump_counts, dim_X)
        return inputs.view(z.shape[0], z.shape[1], x.shape[1])

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
        self.dim_X = 20              # The integer n
        self.dim_u = 20              # The dimension of U
        self.dim_W = 20              # The integer m
        self.dim_L = 1             # The integer l # NOTE: the code only works when this is 1. What does it mean when L > 1? 

        if self.dim_L != 1:
            print('Warning: The dimension of L is not 1, which may cause problems in the code.')

        
        
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

        self.MC_sample_size = 10  # The integer M
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
                jump_times_BLC[sample, k, :jump_counts[sample, k]] = np.sort(jump_times_BLC[sample, k, :jump_counts[sample, k]]) # we need to sort the jump times # NOTE: or do we? 

        # Create a mask which indicates whether each jump time is in the interval [i * delta_t, (i + 1) * delta_t)
        jump_mask_TBLC = np.zeros((self.time_step_count, sample_size, self.dim_L, np.max(jump_counts)), dtype=int)
        for i in range(self.time_step_count):
            jump_mask_TBLC[i, :, :, :] = (jump_times_BLC[:, :, :] >= i * self.delta_t) & (jump_times_BLC[:, :, :] < (i + 1) * self.delta_t)

        # Sample the jump sizes as normal distributions with predefined means and stds
        jump_sizes_BLCX = np.zeros((sample_size, self.dim_L, np.max(jump_counts), self.dim_X), dtype=float)
        
        # NOTE: commenting this out means there will be no jumps
        # for l in range(self.dim_L):
        #    jump_sizes_BLCX[:, l, :, :] = np.random.normal(loc=self.jump_size_mean[l], scale=self.jump_size_std[l], size=(sample_size, np.max(jump_counts), self.dim_X))
            
        

        return SampleData(delta_W_TBW   = torch.tensor(delta_W_TBW,    dtype=torch.float32).to(device),
                          jump_times_BLC = torch.tensor(jump_times_BLC, dtype=torch.float32).to(device),
                          jump_mask_TBLC  = torch.tensor(jump_mask_TBLC,  dtype=torch.float32).to(device),
                          jump_sizes_BLCX = torch.tensor(jump_sizes_BLCX, dtype=torch.float32).to(device),)

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

    def sigma(self, t, x, u):
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
        
        print('Sample data shapes:')
        print('delta_W_TBW: ', sample_data.delta_W_TBW.shape)
        print('jump_times_BLC: ', sample_data.jump_times_BLC.shape)
        print('jump_mask_TBLC: ', sample_data.jump_mask_TBLC.shape)
        print('jump_sizes_BLCX: ', sample_data.jump_sizes_BLCX.shape)
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
    config.sample_stock_price(sample_size=10)
    print('Training time 1:')
    solver = Solver()
    solver.train()
    k = 10
    ratio = '001'
    data = np.array(solver.training_history)
    output = np.zeros((len(data[:, 0]), 4 + k))
    output[:, 0] = data[:, 0]  # step
    output[:, 1] = data[:, 2]  # p_init
    output[:, 2] = data[:, 3]  # loss
    output[:, 3] = data[:, 4]  # ratio
    output[:, 4] = data[:, 1]  # cost

    for i in range(k - 1):
        print('Training time %3u:' % (i + 2))
        solver = Solver()
        solver.train()
        data = np.array(solver.training_history)
        output[:, 5 + i] = data[:, 1]

    a = ['%d', '%.5e', '%.5e', '%.5e']
    for i in range(k):
        a.append('%.5e')
    np.savetxt('./LQ_data_'+ratio+'_d100.csv', output, fmt=a, delimiter=',')

    print('Solving is done!')

if __name__ == '__main__':
    main()