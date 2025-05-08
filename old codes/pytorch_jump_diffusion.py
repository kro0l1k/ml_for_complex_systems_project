import time
import numpy as np
import torch
import math
from scipy.stats import multivariate_normal as normal
import matplotlib.pyplot as plt

LAMBDA = 0.05

# Set default tensor type to float
torch.set_default_dtype(torch.float32)

# Check for available devices 
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class Solver(object):
    def __init__(self,):
        self.valid_size = 512
        self.batch_size = 256
        self.num_iterations = 5000
        self.logging_frequency = 200
        self.lr_values = [5e-3, 5e-3, 5e-3]
        
        self.lr_boundaries = [2000, 4000]
        self.config = Config()

        self.model = WholeNet().to(device)  # Move model to the selected device
        self.p_init = self.model.p_init
        print("y_intial: ", self.p_init.detach().cpu().numpy())  # Ensure tensor is moved to CPU for printing
        
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
        dW = self.config.sample(self.valid_size)
        print("dW shape: ", dW.shape) # (512, 100, 25) (M = valid size, d,T)
        valid_data = dW.to(device)  # Ensure data is on the correct device
        
        for step in range(self.num_iterations+1):
            # Custom learning rate adjustment
            if step in self.lr_boundaries:
                idx = self.lr_boundaries.index(step) + 1
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr_values[idx]
            
            if step % self.logging_frequency == 0:
                with torch.no_grad():
                    loss, cost, ratio = self.model(valid_data)
                    p_init = self.p_init.detach().cpu().numpy()[0][0]
                    elapsed_time = time.time() - start_time
                    training_history.append([step, cost.item(), p_init, loss.item(), ratio.item()])
                    print(f"step: {step:5d}, loss: {loss.item():.4e}, Y0: {p_init:.4e}, cost: {cost.item():.4e}, "
                          f"elapsed time: {int(elapsed_time):3d}, ratio: {ratio.item():.4e}")
            
            self.train_step(self.config.sample(self.batch_size).to(device))  # Ensure data is on the correct device
            
            self.scheduler.step()
        
        self.training_history = training_history

    def train_step(self, train_data):
        """Updating the gradients"""
        self.optimizer.zero_grad()
        loss, _, _ = self.model(train_data)
        loss.backward()
        self.optimizer.step()

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
        self.r_net = FNNetR()

    def forward(self, dw, training=True):
        # Ensure input tensor is on the correct device
        dw = dw.to(device) # Shape: (sample_size, dim_X, time_step_count)
        x_init = torch.ones(1, self.config.dim_X, dtype=torch.float32).to(device) * 0.0 
        time_stamp = np.arange(0, self.config.time_step_count) * self.config.delta_t 
        all_one_vec = torch.ones(dw.shape[0], 1, dtype=torch.float32).to(device) # Shape: (sample_size, 1)
        x = torch.matmul(all_one_vec, x_init) # Shape: (sample_size, dim_X)
        p = torch.matmul(all_one_vec, self.p_init)
        l = 0.0  # The cost functional
        H = 0.0  # The constraint term
        
        for t in range(0, self.config.time_step_count):
            data = (time_stamp[t], x)
            q = self.q_net(data)
            u = self.u_net(data)
            l = l + self.config.f_fn(time_stamp[t], x, u) * self.config.delta_t
            H = H + self.config.Hu_fn(time_stamp[t], x, p, q, u)
            b_ = self.config.b_fn(time_stamp[t], x, u)
            sigma_ = self.config.sigma_fn(time_stamp[t], x, u)
            f_ = self.config.Hx_fn(time_stamp[t], x, u, p, q)

            x = x + b_ * self.config.delta_t + sigma_ * dw[:, :, t]
            p = p - f_ * self.config.delta_t + q * dw[:, :, t]

        delta = p + self.config.g_x_fn(x)
        loss = torch.mean(torch.sum(delta**2, 1, keepdim=True) + LAMBDA * H)
        ratio = torch.mean(torch.sum(delta**2, 1, keepdim=True)) / torch.mean(H)

        l = l + self.config.g_fn(x)
        cost = torch.mean(l)

        return loss, cost, ratio

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
                       self.config.dim_X * self.config.dim_L + 10]
        
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
        
        # Output layer: output dim_X * dim_L values per sample
        self.dense_layers.append(
            torch.nn.Linear(num_hiddens[-1], self.config.dim_X * self.config.dim_L)
        )
        self.bn_layers.append(
            torch.nn.BatchNorm1d(
                self.config.dim_X * self.config.dim_L,
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
        # Reshape to (batch_size, dim_X, dim_L)
        x = x.view(x.shape[0], self.config.dim_X, self.config.dim_L)
        return x

class Config(object):
    """Define the configs in the systems"""
    def __init__(self):
        super(Config, self).__init__()
        self.dim_X = 10             # The integer n
        self.dim_u = 1              # The dimension of U
        self.dim_W = 10             # The integer m
        self.dim_L = 1              # The integer l

        self.X_init = np.ones(self.dim_X, dtype=float) # The initial value of X at time 0
        
        self.jump_intensity = np.array([1, 
                                        ], dtype=float) # The l-dimensional vector lambda
        self.jump_size_distribution = [normal(mean=np.zeros(self.dim_X), cov = np.eye(self.dim_X)),
                                       ] # The l-tuple of jump size distributions nu
        
        assert len(self.jump_intensity) == self.dim_L, "jump_intensity should be of length dim_L"
        assert len(self.jump_size_distribution) == self.dim_L, "jump_size_distribution should be of length dim_L"

        self.MC_sample_size = 1000  # The integer M
        self.terminal_time = 10     # The terminal time T
        self.time_step_count = 100
        self.delta_t = (self.terminal_time + 0.0) / self.time_step_count

    def sample(self, sample_size : int):
        
        delta_W = normal.rvs(size=[sample_size, self.dim_W, self.time_step_count]) * np.sqrt(self.delta_t)

        # jump_counts is of shape (sample_size, dim_L)
        # Each entry in jump_counts[:,k-1] is sampled from a Poisson distribution with intensity self.jump_intensity[k-1]
        jump_counts = np.random.poisson(self.jump_intensity * self.terminal_time, size=(sample_size, self.dim_L))
        jump_times = np.random.uniform(0, self.terminal_time, size=(sample_size, self.dim_L, np.max(jump_counts)))

        # Set the excess samples of jump times to dummy value
        for sample in range(sample_size):
            for k in range(self.dim_L):
                jump_times[sample, k, jump_counts[sample, k]:] = self.terminal_time + 1.0 # Dummy value
                # jump_times[sample, k, :jump_counts[sample, k]] = np.sort(jump_times[sample, k, :jump_counts[sample, k]])

        # Create a mask which indicates whehter each jump time is in the interval [i * delta_t, (i + 1) * delta_t)
        jump_mask = np.zeros((self.time_step_count, sample_size, self.dim_L, np.max(jump_counts)), dtype=int)
        for i in range(self.time_step_count):
            jump_mask[i, :, :, :] = (jump_times[:, :, :] >= i * self.delta_t) & (jump_times[:, :, :] < (i + 1) * self.delta_t)

        # Sample the jump sizes from the corresponding distributions
        jump_sizes = np.zeros((sample_size, self.dim_L, np.max(jump_counts), self.dim_X))
        for sample in range(sample_size):
            for k in range(self.dim_L):
                jump_sizes[sample, k, :jump_counts[sample, k], :] = self.jump_size_distribution[k].rvs(size=jump_counts[sample, k])

        print(f"Jump counts: {jump_counts}")
        print(f"One sample of jump times: {jump_times[0, :, :]}")
        for i in range(self.time_step_count):
            print(f"In interval [{i * self.delta_t:.1f}, {(i+1) * self.delta_t:.1f}) are {jump_times[0, :, :] * jump_mask[i, 0, :, :]}")
        print(f"One sample of jump sizes: {jump_sizes[0, :, :, :]}")

        # Generate sample points for integration with respect to nu
        MC_sample_points = np.zeros((self.dim_L, self.MC_sample_size, self.dim_X))
        for k in range(self.dim_L):
            MC_sample_points[k, :, :] = self.jump_size_distribution[k].rvs(size=self.MC_sample_size)

        self.delta_W = torch.tensor(delta_W, dtype=torch.float32).to(device)
        self.jump_times = torch.tensor(jump_times, dtype=torch.float32).to(device)
        self.jump_mask = torch.tensor(jump_mask, dtype=torch.bool).to(device)
        self.jump_sizes = torch.tensor(jump_sizes, dtype=torch.float32).to(device)
        self.MC_sample_points = torch.tensor(MC_sample_points, dtype=torch.float32).to(device)

    def f_fn(self, t, x, u):
        return 0

    def g_fn(self, x):
        return - 0.5 * torch.sum((x - 10) ** 2, dim=1, keepdim=True)

    def g_x_fn(self, x):
        return - (x - 10)
    
    def b_fn(self, t, x, u):
        return 2 * u

    def sigma_fn(self, t, x, u):
        return np.sqrt(2)
    
    def eta_fn(self, t, x, u, z):
        return u * z

    def Hx_fn(self, t, x, u, p, q, r):
        return 0

    def Hu_fn(self, t, x, u, p, q, r):
        return 0

def main():
    config = Config()
    config.sample(10)  # Test the sample function
    return

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