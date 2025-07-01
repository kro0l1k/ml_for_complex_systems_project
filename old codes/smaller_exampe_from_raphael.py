import time
import numpy as np
import torch
import math
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

eta_1 = 0.01 # or 0.05
eta_2 = 0.5 # or 0
MC_SAM_SIZE = 50
BATCH_SIZE = 512
NUM_ITERATIONS = 100
LOGGING_FREQUENCY = 10
TERMINAL_TIME = 1.0
TICKS_PER_SECOND = 200

# Set default tensor type to float
torch.set_default_dtype(torch.float32)

# Check for available devices
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(" --- config params -----")
print(f"Using device: {device}, LQR example")
print("training using eta_1: ", eta_1)
print("using MC_SAMPLE_SIZE = ", MC_SAM_SIZE)
print("using NUM_ITERATIONS  = ", NUM_ITERATIONS )
print("using TERMINA TIME = ", TERMINAL_TIME)
print("using ticks_per_second = ", TICKS_PER_SECOND)

class Solver(object):
    def __init__(self, config):
        self.valid_size = BATCH_SIZE
        self.batch_size = BATCH_SIZE # NOTE: how big should the batch size be?
        self.num_iterations = NUM_ITERATIONS # NOTE : SMALLER FOR TESTING. CHANGE BACK TO 5000
        self.logging_frequency = LOGGING_FREQUENCY # NOTE: SMALLER FOR TESTING. CHANGE BACK TO 2000
        self.lr_values = [5e-3, 1e-3, 5e-4] # prev all 5e-3

        self.lr_boundaries = [int(0.2 * NUM_ITERATIONS), int(0.6 * NUM_ITERATIONS)] # NOTE: does this make sense?
        self.config = config

        self.model = WholeNet(config).to(device)  # Move model to the selected device
        print("y_initial: ", self.model.p_init.detach().cpu().numpy())

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
        training_history = np.zeros((self.num_iterations+1, 3))  # Store step, loss, performance
        validation_data = self.config.sample(self.valid_size)
        validation_history = np.zeros((self.num_iterations+1, 3))  # Store step, loss, performance

        for step in range(self.num_iterations+1):
            # Custom learning rate adjustment
            if step in self.lr_boundaries:
                idx = self.lr_boundaries.index(step) + 1
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr_values[idx]

            # Calculate validation loss and log it
            if step % self.logging_frequency == 0:
                with torch.no_grad():
                    loss_, cost_, ratio_ = self.model(validation_data)
                    p_init_euclidean = np.linalg.norm(self.model.p_init.detach().cpu().numpy())
                    elapsed_time = time.time() - start_time
                    print(f"step: {step:5d}, loss: {loss_.item():.4e}, ||Y0||: {p_init_euclidean:.4e}, cost functional: {cost_.item():.4e}, "
                          f"elapsed time: {int(elapsed_time):3d}, ratio: {ratio_.item():.4e}")
                    validation_history[step, 0] = step
                    validation_history[step, 1] = loss_.item()
                    validation_history[step, 2] = - cost_.item()
            else:
                validation_history[step, 1] = -1

            # Gradient descent
            self.optimizer.zero_grad()
            loss, cost, ratio = self.model(self.config.sample(self.batch_size), training=True)
            training_history[step, 0] = step
            training_history[step, 1] = loss.item()
            training_history[step, 2] = -cost.item()
            loss.backward()
            self.optimizer.step()

            self.scheduler.step()

        self.training_history = training_history
        validation_history = validation_history[validation_history[:, 1] > 0]
        # Plot the graph of loss
        training_history = np.array(training_history)
        plt.plot(training_history[:, 0], training_history[:, 1], label='Training')
        plt.plot(validation_history[:, 0], validation_history[:, 1], label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()
        plt.grid()
        plt.show()

        # Plot the graph of cost functional
        plt.plot(training_history[:, 0], -training_history[:, 2], label='Training')
        plt.plot(validation_history[:, 0], -validation_history[:, 2], label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Cost')
        plt.legend()
        plt.grid()
        plt.show()

    def generate_trajectoy(self, sample_size):
        """Generate the trajectory of the model. start from the initial value of X0 and apply the control generated by the FFNetU"""

        # Generate sample data for simulation
        delta_W_TBW = self.config.sample(sample_size)
        t = np.arange(0, self.config.time_step_count) * self.config.delta_t

        X_BX = self.config.X_init.repeat(sample_size, 1)  # Shape: (sample_size, dim_X)
        int_f = torch.zeros(sample_size, 1, dtype=torch.float32).to(device)  # The integral term for f
        trajectory = []
        trajectory.append(X_BX.detach().cpu().numpy())  # Store initial state

        for i in range(0, self.config.time_step_count):
            u_BU = self.model.u_net((t[i], X_BX))  # Access u_net through self.model

            int_f = int_f + self.config.f(t[i], X_BX, u_BU) * self.config.delta_t

            jump_sizes_BL1X = torch.zeros(sample_size, self.config.dim_L, 1, self.config.dim_X, dtype=torch.float32).to(device)  # Shape: (sample_size, dim_L, dim_X)
            for l in range(self.config.dim_L):
                jump_counts_B1 = torch.poisson(self.config.jump_intensity(l, t[i], X_BX, u_BU).cpu() * self.config.delta_t).to(device)
                # for jump_count in range(torch.max(jump_counts_B)):
                jump_size = self.config.jump_size_distribution.sample((sample_size,)).to(device)
                jump_sizes_BL1X[:, l, 0, :] += (jump_counts_B1 > 0) * jump_size

            X_BX = X_BX + self.config.drift(t[i], X_BX, u_BU) * self.config.delta_t + \
                torch.einsum('bxw,bw->bx', self.config.diffusion(t[i], X_BX, u_BU), delta_W_TBW[i, :, :])
            for l in range(self.config.dim_L):
                X_BX += jump_sizes_BL1X[:, l, 0, :]

            trajectory.append(X_BX.detach().cpu().numpy())

        costs = int_f + self.config.g(X_BX)
        return np.array(trajectory), costs

    def plot_trajectory(self, trajectory):
        """Plot the trajectory of the model"""
        # Use trajectory.shape[0] to ensure t has the right number of points
        t = np.linspace(0, self.config.terminal_time, trajectory.shape[0])

        for i in range(self.config.dim_X):
            # Plot each sample separately
            for sample_idx in range(trajectory.shape[1]):
                plt.plot(t, trajectory[:, sample_idx, i])

        plt.title('Trajectory of the model')
        plt.xlabel('t')
        plt.ylabel('X(t)')
        plt.grid()
        # plt.show() # to make it run

class WholeNet(torch.nn.Module):
    """Building the neural network architecture"""
    def __init__(self, config):
        super(WholeNet, self).__init__()
        self.config = config
        # Initialize p_init as a parameter
        self.p_init = torch.nn.Parameter(
            torch.randn(1, self.config.dim_X, dtype=torch.float32).to(device),  # Ensure float32 and correct device
            requires_grad=True
        )
        self.q_net = FNNetQ(config)
        self.u_net = FNNetU(config)
        self.r_nets = torch.nn.ModuleList()
        for _ in range(self.config.dim_L):
            self.r_nets.append(FNNetR(config))

    def forward(self, delta_W_TBW, training=True):
        sample_size = delta_W_TBW.shape[1]
        t = np.arange(0, self.config.time_step_count) * self.config.delta_t
        X_BX = self.config.X_init.repeat(sample_size, 1)  # Shape: (sample_size, dim_X)
        p_BX = self.p_init.repeat(sample_size, 1)  # Shape: (sample_size, dim_X)
        int_f = torch.zeros(sample_size, 1, dtype=torch.float32).to(device)  # The integral term for f
        int_Hu = 0.0  # The constraint term
        MC_sample_points_BLMX = self.config.MC_sample_points_LMX.repeat(sample_size, 1, 1, 1)  # Shape: (sample_size, dim_L, MC_sample_size, dim_X)

        for i in range(0, self.config.time_step_count):
            u_BU = self.u_net((t[i], X_BX))   # Shape: (sample_size, dim_u)
            q_BXW = self.q_net((t[i], X_BX))   # Shape: (sample_size, dim_X, dim_W)
            jump_counts_B = torch.poisson(self.config.jump_intensity(0, t[i], X_BX, u_BU).cpu() * self.config.delta_t).to(device)
            jump_sizes_BX = (jump_counts_B > 0) * self.config.jump_size_distribution.sample((sample_size,)).to(device)
            r_monte_BMX = self.r_nets[0]((t[i], X_BX, MC_sample_points_BLMX[:, 0, :, :]))

            Hu = -self.config.f_u(t[i], X_BX, u_BU) + \
                torch.einsum('bjn,bj->bn', self.config.drift_u(t[i], X_BX, u_BU), p_BX) + \
                torch.einsum('bjkn,bjk->bn', self.config.diffusion_u(t[i], X_BX, u_BU), q_BXW) + \
                self.config.jump_intensity_u(0, t[i], X_BX, u_BU) * torch.mean(MC_sample_points_BLMX[:, 0, :, :] * (r_monte_BMX + p_BX.unsqueeze(1)) + X_BX.unsqueeze(1) * r_monte_BMX, dim=1)

            int_Hu = int_Hu + torch.mean(torch.sum(Hu**2, dim=1, keepdim=True) * self.config.delta_t)

            int_f = int_f + self.config.f(t[i], X_BX, u_BU) * self.config.delta_t

            # Update p
            Hx = - self.config.f_x(t[i], X_BX, u_BU) + \
                torch.einsum('bjn,bj->bn', self.config.drift_x(t[i], X_BX, u_BU), p_BX) + \
                torch.einsum('bjkn,bjk->bn', self.config.diffusion_x(t[i], X_BX, u_BU), q_BXW) + \
                self.config.jump_intensity(0, t[i], X_BX, u_BU) * torch.mean(r_monte_BMX, dim=1)

            p_BX = p_BX - Hx * self.config.delta_t + torch.einsum('bxw,bw->bx', q_BXW, delta_W_TBW[i, :, :])
            for l in range(self.config.dim_L):
                r_jump_B1X = self.r_nets[l]((t[i], X_BX, jump_sizes_BX.unsqueeze(1)))
                p_BX = p_BX + r_jump_B1X[:, 0, :]

            # Update X
            X_BX = X_BX + self.config.drift(t[i], X_BX, u_BU) * self.config.delta_t + \
                torch.einsum('bxw,bw->bx', self.config.diffusion(t[i], X_BX, u_BU), delta_W_TBW[i, :, :])
            for l in range(self.config.dim_L):
                X_BX += jump_sizes_BX

        cost = torch.mean(int_f + self.config.g(X_BX))

        terminal_value_loss = p_BX + self.config.g_x(X_BX)
        loss = torch.mean(torch.sum(terminal_value_loss**2, 1, keepdim=True)) + eta_1 * int_Hu + eta_2 * cost
        ratio = torch.mean(torch.sum(terminal_value_loss**2, 1, keepdim=True)) / int_Hu

        return loss, cost, ratio

class FNNetQ(torch.nn.Module):
    """ Define the feedforward neural network """
    def __init__(self, config):
        super(FNNetQ, self).__init__()
        self.config = config
        num_hiddens = [20, 20, 20, 20]

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
    def __init__(self, config):
        super(FNNetU, self).__init__()
        self.config = config
        num_hiddens = [20, 20, 20, 20]

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
    def __init__(self, config):
        super(FNNetR, self).__init__()
        self.config = config
        num_hiddens = [20, 20, 20, 20]

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

class Config(object):
    """Define the configs in the systems"""
    def __init__(self, dim, sigma_XW, jump_Covariance_XX):
        super(Config, self).__init__()
        self.dim_X = dim       # The integer n
        self.dim_u = dim       # The dimension of U
        self.dim_W = dim       # The integer m
        self.dim_L = 1         # The integer l

        self.X_init = torch.zeros(self.dim_X, dtype=torch.float32, device=device) # The initial value of X at time 0

        self.c_1 = 1.00  # Running cost coefficient
        self.c_2 = 0.25  # Terminal cost coefficient

        self.Sigma_XW = sigma_XW
        self.jump_Covariance_XX = jump_Covariance_XX
        self.jump_Mean_X = torch.zeros(self.dim_X, dtype=torch.float32, device=device)  # The mean of the jump sizes
        
        # Create distribution on CPU to avoid MPS limitations with Cholesky decomposition
        jump_mean_cpu = self.jump_Mean_X.cpu()
        jump_cov_cpu = self.jump_Covariance_XX.cpu()
        self.jump_size_distribution = torch.distributions.MultivariateNormal(jump_mean_cpu, jump_cov_cpu)  # The jump size distribution

        self.Lambda_1 = 0.25 # Jump intensity coefficient 1
        self.Lambda_2 = 0.00 # Jump intensity coefficient 2
        # Setup 1: Lambda_1 = 0.00 , Lambda_2 = 2.00
        # Alternative setup (constant jump intensity): Lambda_1 = 0.25, Lambda_2 = 0.00


        # The terminal time in years
        self.terminal_time = TERMINAL_TIME  # e.g., 1 year
        # Roughly the number of trading days
        self.time_step_count = math.floor(self.terminal_time * TICKS_PER_SECOND)
        self.delta_t = float(self.terminal_time) / self.time_step_count

        self.MC_sample_size = MC_SAM_SIZE  # The integer M
        # Generate sample points for integration with respect to jump_size_distribution
        self.MC_sample_points_LMX = self.jump_size_distribution.sample((self.dim_L, self.MC_sample_size)).to(device)


        # Closed-form solution
        self.zeta = torch.trace(self.jump_Covariance_XX)
        sol = solve_ivp(lambda t, h: (h ** 2) / (2 * self.c_1 + h * float(self.zeta * self.Lambda_2)),
                        (self.terminal_time, 0.0),
                        [2 * self.c_2],
                        t_eval=np.flip(np.arange(0, self.time_step_count + 1) * self.delta_t),
                        method="RK45")
        self.h_closed_form = np.flip(sol.y[0])

        # Plot h_closed_form
        plt.plot(np.arange(0, self.time_step_count + 1) * self.delta_t, self.h_closed_form)
        plt.title('Closed-form solution h(t)')
        plt.xlabel('t')
        plt.ylabel('h(t)')
        plt.grid()
        plt.show()

        int_h = np.trapz(self.h_closed_form, np.arange(0, self.time_step_count + 1) * self.delta_t)
        self.f_0 = 0.5 * (torch.trace(self.Sigma_XW @ self.Sigma_XW.T) + self.Lambda_1 * self.zeta) * \
                torch.tensor(int_h, dtype=torch.float32, device=device)

    def V_0(self, x):
        return 0.5 * self.h_closed_form[0] * torch.sum(x ** 2) + self.f_0

    def sample(self, sample_size : int):
        delta_W_TBW = np.random.normal(size=(self.time_step_count, sample_size, self.dim_W)) * np.sqrt(self.delta_t)
        return torch.tensor(delta_W_TBW, dtype=torch.float32).to(device)

    def jump_intensity(self, l, t, x, u):
        # Input shape of u: (batch_size, dim_u)
        # Output shape: batch_size
        return self.Lambda_1 + self.Lambda_2 * torch.sum(u ** 2, dim=1, keepdim=True)

    def jump_intensity_u(self, l, t, x, u):
        """Derivative of jump intensity with respect to control"""
        # Output shape: (batch_size, dim_u)
        return 2 * self.Lambda_2 * u

    def f(self, t, x, u):
        # Output shape: (batch_size, 1)
        return self.c_1 * torch.sum(u ** 2, dim=1, keepdim=True)

    def f_x(self, t, x, u):
        # Output shape: (batch_size, dim_X)
        return torch.zeros(x.shape[0], self.dim_X, dtype=torch.float32, device=x.device)

    def f_u(self, t, x, u):
        # Output shape: (batch_size, dim_u)
        return 2 * self.c_1 * u

    def g(self, x):
        # Output shape: (batch_size, 1)
        return self.c_2 * torch.sum(x ** 2, dim=1, keepdim=True)

    def g_x(self, x):
        # Output shape: (batch_size, dim_X)
        return 2 * self.c_2 * x

    def drift(self, t, x, u):
        # Output shape: (batch_size, dim_X)
        # b(t, x, u) = u
        return u

    def drift_x(self, t, x, u):
        # Partial derivatives of each component of b with respect to x
        # Output shape: (batch_size, dim_X, dim_X)
        return torch.zeros(x.shape[0], self.dim_X, self.dim_X, dtype=torch.float32, device=x.device)

    def drift_u(self, t, x, u):
        # Partial derivatives of each component of b with respect to u
        # Output shape: (batch_size, dim_X, dim_u)
        # Output shape: (batch_size, dim_X, dim_u)
        return torch.eye(self.dim_X, self.dim_u, dtype=torch.float32, device=x.device).unsqueeze(0).repeat(x.shape[0], 1, 1)

    def diffusion(self, t, x, u):
        # Input shape of u: (batch_size, dim_X)
        # Output shape: (batch_size, dim_X, dim_W)
        return self.Sigma_XW.unsqueeze(0).repeat(x.shape[0], 1, 1)

    def diffusion_x(self, t, x, u):
        # Partial derivatives of each component of diffusion with respect to x
        # Input shape of u: (batch_size, dim_u)
        # Output shape: (batch_size, dim_X, dim_W, dim_X)
        return torch.zeros(x.shape[0], self.dim_X, self.dim_W, self.dim_X, dtype=torch.float32, device=x.device)

    def diffusion_u(self, t, x, u):
        # Partial derivatives of each component of diffusion with respect to u
        # Input shape of u: (batch_size, dim_u)
        # Output shape: (batch_size, dim_X, dim_W, dim_u)
        # diffusion_u(t, x, u)[b, j, k, :] = \partial diffusion^{j,k}(t, x, u) / \partial u
        return torch.zeros(x.shape[0], self.dim_X, self.dim_W, self.dim_u, dtype=torch.float32, device=x.device)

# Set the random seed for reproducibility
torch.manual_seed(42)
dim = 10
config = Config(dim=dim,
                sigma_XW= torch.rand(dim, dim, dtype=torch.float32, device=device),  # The diffusion coefficient Sigma
                jump_Covariance_XX= torch.diag(torch.rand(dim, dtype=torch.float32, device=device))  # The covariance matrix of the jump sizes
                )
initial_value_first_components = [-1.5, -1.0, -0.5]
print("first components of X_init: ", initial_value_first_components)
V_for_different_X_init = []
closed_form_V_for_different_X_init = []
std_for_different_X_init = []
print("testing for different X_init values")
for x_1 in initial_value_first_components:
    # Set the first component of X_init to x_1
    config.X_init[0] = x_1
    print(f"X_0 value: ({x_1}, 0, ..., 0)")
    closed_form_V = config.V_0(config.X_init).detach().cpu().numpy()
    print('Closed form V_0: ', closed_form_V)

    solver = Solver(config)
    solver.train()
    # generate the trajectory
    trajectory, cost_functional = solver.generate_trajectoy(256)

    # print('Trajectory shape after the plot: ', trajectory.shape)
    # compute the cost functional for each of the trajectories. get the mean and std
    print("cost functional shape: ", cost_functional.shape)
    print("cost functional: ", cost_functional)
    print('Mean: ', cost_functional.mean().item())
    print('Std: ', cost_functional.std().item())
    solver.plot_trajectory(trajectory)

    V_for_different_X_init.append(cost_functional.mean().item())
    closed_form_V_for_different_X_init.append(closed_form_V)
    std_for_different_X_init.append(cost_functional.std().item())

    # Make a heatmap showing the difference between FNNetU and closed_form_solver.u_star
    t_values = np.arange(0, config.time_step_count + 1) * config.delta_t
    x_values = torch.linspace(-2.5, 2.5, 256, dtype=torch.float32).unsqueeze(1).to(device)
    # Fill the extra dimensions with one
    x_values = torch.cat([x_values, torch.ones(x_values.shape[0], dim-1, dtype=torch.float32).to(device)], dim=1)
    trained_values = np.zeros((len(t_values), x_values.shape[0], dim))
    closed_form_values = np.zeros((len(t_values), x_values.shape[0], dim))
    for i in range(len(t_values)):
        trained_values[i, :, :] = solver.model.u_net((t_values[i], x_values)).detach().cpu().numpy()
        closed_form_values[i, :, :] = - (config.h_closed_form[i] * x_values.detach().cpu().numpy()) / (2 * config.c_1 + config.h_closed_form[i] * config.zeta.detach().cpu().numpy() * config.Lambda_2)

    value_difference_abs = np.sum(np.abs(trained_values - closed_form_values), axis=2)
    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(value_difference_abs, aspect='auto', extent=[-2.5, 2.5, 0, config.terminal_time], origin='lower', cmap='viridis')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('t')
    plt.grid()
    plt.show()


print("Cost Functional for different X_init values:")
print("Mean: \n", V_for_different_X_init)
print("Std: \n", std_for_different_X_init)

### plot the cost functional for different X_init values. add a transparent area for +-1 std
plt.figure()
plt.plot(initial_value_first_components, V_for_different_X_init, label='Mean Cost Functional')
plt.fill_between(initial_value_first_components,
                 np.array(V_for_different_X_init) - 0.2 * np.array(std_for_different_X_init),
                 np.array(V_for_different_X_init) + 0.2 * np.array(std_for_different_X_init),
                 alpha=0.2, label='0.2 Std Dev')
plt.plot(initial_value_first_components, closed_form_V_for_different_X_init, label='Closed Form Cost Functional', linestyle='--')
plt.title('Cost Functional for Different Initial Values of X0')
plt.xlabel('Initial Value of X0')
plt.ylabel('Cost Functional')
plt.legend()
plt.grid()
plt.show()

# save the plot
plt.savefig('cost_functional_2.png')
plt.close()