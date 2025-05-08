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
        self.y_init = self.model.y_init
        print("y_intial: ", self.y_init.detach().cpu().numpy())  # Ensure tensor is moved to CPU for printing
        
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
                    y_init = self.y_init.detach().cpu().numpy()[0][0]
                    elapsed_time = time.time() - start_time
                    training_history.append([step, cost.item(), y_init, loss.item(), ratio.item()])
                    print(f"step: {step:5d}, loss: {loss.item():.4e}, Y0: {y_init:.4e}, cost: {cost.item():.4e}, "
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
        # Initialize y_init as a parameter
        self.y_init = torch.nn.Parameter(
            torch.randn(1, self.config.dim_y, dtype=torch.float32).to(device),  # Ensure float32 and correct device
            requires_grad=True
        )
        self.z_net = FNNetZ()
        self.u_net = FNNetU()

    def forward(self, dw, training=True):
        # Ensure input tensor is on the correct device
        dw = dw.to(device)
        x_init = torch.ones(1, self.config.dim_x, dtype=torch.float32).to(device) * 0.0
        time_stamp = np.arange(0, self.config.num_time_interval) * self.config.delta_t
        all_one_vec = torch.ones(dw.shape[0], 1, dtype=torch.float32).to(device)
        x = torch.matmul(all_one_vec, x_init)
        y = torch.matmul(all_one_vec, self.y_init)
        l = 0.0  # The cost functional
        H = 0.0  # The constraint term
        
        for t in range(0, self.config.num_time_interval):
            data = (time_stamp[t], x)
            z = self.z_net(data)
            u = self.u_net(data)
            l = l + self.config.f_fn(time_stamp[t], x, u) * self.config.delta_t
            H = H + self.config.Hu_fn(time_stamp[t], x, y, z, u)
            b_ = self.config.b_fn(time_stamp[t], x, u)
            sigma_ = self.config.sigma_fn(time_stamp[t], x, u)
            f_ = self.config.Hx_fn(time_stamp[t], x, u, y, z)

            x = x + b_ * self.config.delta_t + sigma_ * dw[:, :, t]
            y = y - f_ * self.config.delta_t + z * dw[:, :, t]

        delta = y + self.config.hx_tf(self.config.total_T, x)
        loss = torch.mean(torch.sum(delta**2, 1, keepdim=True) + LAMBDA * H)
        ratio = torch.mean(torch.sum(delta**2, 1, keepdim=True)) / torch.mean(H)

        l = l + self.config.h_fn(self.config.total_T, x)
        cost = torch.mean(l)

        return loss, cost, ratio

class FNNetZ(torch.nn.Module):
    """ Define the feedforward neural network """
    def __init__(self):
        super(FNNetZ, self).__init__()
        self.config = Config()
        num_hiddens = [self.config.dim_x+10, self.config.dim_x+10, self.config.dim_x+10]
        
        # Create layer lists 
        self.bn_layers = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(
                self.config.dim_x + 1,  # First layer input size
                momentum=0.99,
                eps=1e-6
            )
        ])
        
        self.dense_layers = torch.nn.ModuleList()
        
        # Hidden layers
        for i in range(len(num_hiddens)):
            # Input size for the first layer
            input_size = self.config.dim_x + 1 if i == 0 else num_hiddens[i-1]
            
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
            torch.nn.Linear(num_hiddens[-1], self.config.dim_z)
        )
        
        self.bn_layers.append(
            torch.nn.BatchNorm1d(
                self.config.dim_z,
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

class FNNetU(torch.nn.Module):
    """ Define the feedforward neural network """
    def __init__(self):
        super(FNNetU, self).__init__()
        self.config = Config()
        num_hiddens = [self.config.dim_x+10, self.config.dim_x+10, self.config.dim_x+10]
        
        # Create layer lists 
        self.bn_layers = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(
                self.config.dim_x + 1,  # First layer input size
                momentum=0.99,
                eps=1e-6
            )
        ])
        
        self.dense_layers = torch.nn.ModuleList()
        
        # Hidden layers
        for i in range(len(num_hiddens)):
            # Input size for the first layer
            input_size = self.config.dim_x + 1 if i == 0 else num_hiddens[i-1]
            
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

class Config(object):
    """Define the configs in the systems"""
    def __init__(self):
        super(Config, self).__init__()
        self.dim_x = 100
        self.dim_y = 100
        self.dim_z = 100
        self.dim_u = 100
        self.num_time_interval = 25
        self.total_T = 1.0
        self.delta_t = (self.total_T + 0.0) / self.num_time_interval
        self.sqrth = np.sqrt(self.delta_t)
        self.t_stamp = np.arange(0, self.num_time_interval) * self.delta_t

    def sample(self, num_sample):
        dw_sample = normal.rvs(size=[num_sample, self.dim_x, self.num_time_interval]) * self.sqrth
        dw_sample = dw_sample.astype(np.float32)  # Ensure float32
        dw_sample_tensor = torch.tensor(dw_sample, dtype=torch.float32).to(device)  # Ensure tensor is on the correct device
        return dw_sample_tensor

    def f_fn(self, t, x, u):
        return torch.sum(u ** 2, dim=1, keepdim=True)

    def h_fn(self, t, x):
        return torch.log(0.5*(1 + torch.sum(x**2, dim=1, keepdim=True)))

    def b_fn(self, t, x, u):
        return 2 * u

    def sigma_fn(self, t, x, u):
        return np.sqrt(2)

    def Hx_fn(self, t, x, u, y, z):
        return 0

    def hx_tf(self, t, x):
        a = 1 + torch.sum(x ** 2, dim=1, keepdim=True)
        return 2 * x / a

    def Hu_fn(self, t, x, y, z, u):
        a = 2 * y - 2 * u
        return torch.sum(a**2, dim=1, keepdim=True)

def main():
    print('Training time 1:')
    solver = Solver()
    solver.train()
    k = 10
    ratio = '001'
    data = np.array(solver.training_history)
    output = np.zeros((len(data[:, 0]), 4 + k))
    output[:, 0] = data[:, 0]  # step
    output[:, 1] = data[:, 2]  # y_init
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