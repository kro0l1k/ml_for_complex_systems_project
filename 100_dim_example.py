import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
from scipy.stats import multivariate_normal as normal
import matplotlib.pyplot as plt

LAMBDA = 0.05

# Set device
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

# Set default tensor type to float32 (MPS doesn't support float64)
torch.set_default_dtype(torch.float32)

class Config:
    """Define the configs in the systems"""
    def __init__(self):
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
        return torch.tensor(dw_sample, dtype=torch.float32, device=device)

    def f_fn(self, t, x, u):
        return torch.sum(u ** 2, dim=1, keepdim=True)

    def h_fn(self, t, x):
        return torch.log(0.5 * (1 + torch.sum(x**2, dim=1, keepdim=True)))

    def b_fn(self, t, x, u):
        return 2 * u

    def sigma_fn(self, t, x, u):
        return float(np.sqrt(2))

    def Hx_fn(self, t, x, u, y, z):
        return 0

    def hx_tf(self, t, x):
        a = 1 + torch.sum(x ** 2, dim=1, keepdim=True)
        return 2 * x / a

    def Hu_fn(self, t, x, y, z, u):
        a = 2 * y - 2 * u
        return torch.sum(a**2, dim=1, keepdim=True)


class FNNetZ(nn.Module):
    """ Define the feedforward neural network for Z"""
    def __init__(self, config):
        super(FNNetZ, self).__init__()
        self.config = config

        hidden_dim = self.config.dim_x + 10
        self.bn_input = nn.BatchNorm1d(self.config.dim_x + 1,
                                     momentum=0.99,
                                     eps=1e-6)

        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        # Create hidden layers
        for i in range(3):  # 3 hidden layers as in original
            self.layers.append(nn.Linear(self.config.dim_x + 1 if i == 0 else hidden_dim,
                                        hidden_dim,
                                        bias=False))
            self.bn_layers.append(nn.BatchNorm1d(hidden_dim,
                                              momentum=0.99,
                                              eps=1e-6))

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, self.config.dim_z)
        self.bn_output = nn.BatchNorm1d(self.config.dim_z,
                                      momentum=0.99,
                                      eps=1e-6)

        # Initialize parameters with the similar distribution to TF
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)

        for bn in self.bn_layers + [self.bn_input, self.bn_output]:
            if bn.weight is not None:
                nn.init.uniform_(bn.weight, 0.1, 0.5)
            if bn.bias is not None:
                nn.init.normal_(bn.bias, 0.0, 0.1)

    def forward(self, inputs, training=True):
        t, x = inputs
        batch_size = x.shape[0]

        # Create time tensor and concatenate
        ts = torch.ones((batch_size, 1), dtype=torch.float32, device=device) * t
        x = torch.cat([ts, x], dim=1)

        # Apply input batch norm
        x = self.bn_input(x)

        # Apply hidden layers
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.bn_layers[i](x)
            x = torch.relu(x)

        # Apply output layer
        x = self.output_layer(x)
        x = self.bn_output(x)

        return x


class FNNetU(nn.Module):
    """ Define the feedforward neural network for U"""
    def __init__(self, config):
        super(FNNetU, self).__init__()
        self.config = config

        hidden_dim = self.config.dim_x + 10
        self.bn_input = nn.BatchNorm1d(self.config.dim_x + 1,
                                     momentum=0.99,
                                     eps=1e-6)

        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        # Create hidden layers
        for i in range(3):  # 3 hidden layers as in original
            self.layers.append(nn.Linear(self.config.dim_x + 1 if i == 0 else hidden_dim,
                                        hidden_dim,
                                        bias=False))
            self.bn_layers.append(nn.BatchNorm1d(hidden_dim,
                                              momentum=0.99,
                                              eps=1e-6))

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, self.config.dim_u)
        self.bn_output = nn.BatchNorm1d(self.config.dim_u,
                                      momentum=0.99,
                                      eps=1e-6)

        # Initialize parameters with the similar distribution to TF
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)

        for bn in self.bn_layers + [self.bn_input, self.bn_output]:
            if bn.weight is not None:
                nn.init.uniform_(bn.weight, 0.1, 0.5)
            if bn.bias is not None:
                nn.init.normal_(bn.bias, 0.0, 0.1)

    def forward(self, inputs, training=True):
        t, x = inputs
        batch_size = x.shape[0]

        # Create time tensor and concatenate
        ts = torch.ones((batch_size, 1), dtype=torch.float32, device=device) * t
        x = torch.cat([ts, x], dim=1)

        # Apply input batch norm
        x = self.bn_input(x)

        # Apply hidden layers
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.bn_layers[i](x)
            x = torch.relu(x)

        # Apply output layer
        x = self.output_layer(x)
        x = self.bn_output(x)

        return x


class WholeNet(nn.Module):
    """Building the neural network architecture"""
    def __init__(self):
        super(WholeNet, self).__init__()
        self.config = Config()
        self.y_init = nn.Parameter(torch.randn(1, self.config.dim_y, dtype=torch.float32, device=device))
        self.z_net = FNNetZ(self.config)
        self.u_net = FNNetU(self.config)

    def forward(self, dw, training=True):
        batch_size = dw.shape[0]

        # Initial values
        x_init = torch.zeros(1, self.config.dim_x, dtype=torch.float32, device=device)
        time_stamp = np.arange(0, self.config.num_time_interval) * self.config.delta_t

        # Expand for batch
        all_one_vec = torch.ones(batch_size, 1, dtype=torch.float32, device=device)
        x = torch.matmul(all_one_vec, x_init)
        y = torch.matmul(all_one_vec, self.y_init)

        # Initialize cost and constraint
        l = 0.0  # The cost functional
        H = 0.0  # The constraint term

        # Loop through time steps
        for t in range(0, self.config.num_time_interval):
            # Forward networks
            data = (time_stamp[t], x)
            z = self.z_net(data, training=training)
            u = self.u_net(data, training=training)

            # Update cost and constraint
            l = l + self.config.f_fn(time_stamp[t], x, u) * self.config.delta_t
            H = H + self.config.Hu_fn(time_stamp[t], x, y, z, u)

            # Update state variables
            b_ = self.config.b_fn(time_stamp[t], x, u)
            sigma_ = self.config.sigma_fn(time_stamp[t], x, u)
            f_ = self.config.Hx_fn(time_stamp[t], x, u, y, z)

            x = x + b_ * self.config.delta_t + sigma_ * dw[:, :, t]
            y = y - f_ * self.config.delta_t + z * dw[:, :, t]

        # Compute loss and ratio
        delta = y + self.config.hx_tf(self.config.total_T, x)
        loss = torch.mean(torch.sum(delta**2, dim=1, keepdim=True) + LAMBDA * H)
        ratio = torch.mean(torch.sum(delta**2, dim=1, keepdim=True)) / torch.mean(H)

        # Update cost
        l = l + self.config.h_fn(self.config.total_T, x)
        cost = torch.mean(l)

        return loss, cost, ratio


class Solver:
    def __init__(self):
        self.valid_size = 512
        self.batch_size = 256
        self.num_iterations = 500
        self.logging_frequency = 50
        self.lr_values = [5e-3, 5e-3, 5e-3]
        self.lr_boundaries = [2000, 4000]

        self.config = Config()
        self.model = WholeNet().to(device)
        self.y_init = self.model.y_init

        # Set up learning rate schedule
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr_values[0], eps=1e-8)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=self.lr_boundaries,
            gamma=1.0  # Since we're manually setting LR values
        )

        # Handle manual LR changes at boundaries
        self.lr_idx = 0

    def train(self):
        """Training the model"""
        start_time = time.time()
        training_history = []

        # Generate validation data
        dW = self.config.sample(self.valid_size)
        valid_data = dW

        # Training loop
        for step in range(self.num_iterations + 1):
            # Log progress
            if step % self.logging_frequency == 0:
                self.model.eval()
                with torch.no_grad():
                    loss, cost, ratio = self.model(valid_data)

                y_init = self.y_init.detach().cpu().numpy()[0][0]
                elapsed_time = time.time() - start_time
                training_history.append([step, cost.item(), y_init, loss.item(), ratio.item()])

                print(f"step: {step:5d}, loss: {loss.item():.4e}, Y0: {y_init:.4e}, cost: {cost.item():.4e}, "
                      f"elapsed time: {int(elapsed_time):3d}, ratio: {ratio.item():.4e}")

            # Check learning rate boundaries
            if step in self.lr_boundaries:
                self.lr_idx += 1
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr_values[self.lr_idx]

            # Train step
            self.train_step(self.config.sample(self.batch_size))

        y_init = self.y_init.detach().cpu().numpy()[0][0]
        print(f'Y0_true: {y_init:.4e}')
        self.training_history = training_history

    def train_step(self, train_data):
        """Update model parameters"""
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        loss, _, _ = self.model(train_data)

        # Backward pass and optimize
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()


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
        print(f'Training time {i + 2:3d}:')
        solver = Solver()
        solver.train()
        data = np.array(solver.training_history)
        output[:, 5 + i] = data[:, 1]

    a = ['%d', '%.5e', '%.5e', '%.5e']
    for i in range(k):
        a.append('%.5e')
    np.savetxt(f'./LQ_data_{ratio}_d100.csv', output, fmt=a, delimiter=',')

    print('Solving is done!')


if __name__ == '__main__':
    main()