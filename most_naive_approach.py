import time
import numpy as np
import torch
import math
from collections import namedtuple
import matplotlib.pyplot as plt

LAMBDA = 0.05 # was 0.05

# Set default tensor type to float
torch.set_default_dtype(torch.float32)

# Check for available devices
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")



class Config(object):
    """Define the configs in the systems"""
    def __init__(self):
        super(Config, self).__init__()
        self.dim_X = 1              # The integer n
        self.dim_u = 1              # The dimension of U
        self.dim_W = 1              # The integer m
        self.dim_L = 1              # The integer l

        ###
        # training config:
        self.valid_size = 512
        self.batch_size = 256 # NOTE: how big should the batch size be?
        self.MC_sample_size = 256  # The integer M
        self.num_iterations = 1000
        self.logging_frequency = 100
        self.lr_values = [0.1, 0.01, 0.005]

        self.lr_boundaries = [int(0.2 * self.num_iterations), int(0.8 * self.num_iterations)]
        ###
        self.TARGET_MEAN_A = 1.1

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
        self.tics_per_unit_of_time = 250  # The number of tics for one year, e.g. 100 tics for one year means 1 tic is 1/100 year

        # Roughly the number of trading days
        self.time_step_count = math.floor(self.terminal_time * self.tics_per_unit_of_time)  # 20 trading days in a month. keep it small for testing.
        # print(f"Time step count: {self.time_step_count}")
        self.delta_t = float(self.terminal_time) / self.time_step_count
        # print(f"Delta t: {self.delta_t}")
        # Generate sample points for integration with respect to nu
        MC_sample_points_LMX = np.random.lognormal(mean=self.log_normal_mu[0], sigma=self.log_normal_sigma[0], size=(self.dim_L, self.MC_sample_size, self.dim_X)) - 1
        self.MC_sample_points_LMX = torch.tensor(MC_sample_points_LMX, dtype=torch.float32).to(device)

    def sample(self, sample_size: int):
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
        return torch.tensor(self.TARGET_MEAN_A, dtype=torch.float32, device=device)

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
                torch.einsum('blc,blcx->bx', sample_data.jump_mask_TBLC[i, :, :, :], sample_data.jump_sizes_BLCX) + \
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

# A simple object to contain the sample data
SampleData = namedtuple('SampleData', [
    'delta_W_TBW',    # torch.Tensor, shape (time_step_count, batch_size, dim_W)
    'jump_times_BLC', # torch.Tensor, (batch_size, dim_L, max_counts)
    'jump_mask_TBLC',  # torch.BoolTensor, (time_step_count, batch_size, dim_L, max_counts)
    'jump_sizes_BLCX'  # torch.Tensor, (batch_size, dim_L, max_counts, n)
])

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

    def Lambda(self, t):
        return torch.tensor(self.config.sigma(t)**2 + self.config.jump_intensity * self.config.jump_size_mean[0] ** 2 + self.config.jump_intensity * self.config.jump_size_std[0] ** 2,
                            dtype=torch.float32, device=device)
    def phi(self, t):
        I_phi = (self.config.mu(t) - self.config.rho(t)) **2 / self.Lambda(t) - 2 * self.config.rho(t)
        I_phi = torch.tensor(I_phi, dtype=torch.float32, device=device)
        return -1.0 * torch.exp(I_phi * (torch.tensor(self.config.terminal_time, dtype=torch.float32, device=device) - t))

    def psi(self, t):
        I_psi = (self.config.mu(t) - self.config.rho(t)) **2 / self.Lambda(t) - self.config.rho(t)
        I_psi = torch.tensor(I_psi, dtype=torch.float32, device=device)
        return self.config.a_in_cost() * torch.exp(I_psi * (self.config.terminal_time - t))

    def u_star(self, t_1, X_BX):
        # t is a scalar,
        # X is a tensor of shape (batch_size, dim_X)
        # output shape: (batch_size, dim_X)
        if not isinstance(t_1, torch.Tensor):
            t_1 = torch.tensor(t_1, dtype=torch.float32, device=device)
        # the mistake was here!
        ustar = (self.config.rho(t_1) - self.config.mu(t_1)) * (self.phi(t_1) * X_BX + self.psi(t_1)) / (self.phi(t_1) * self.Lambda(t_1))
        return ustar
    def get_solution(self):
        """
        should implement the cloed formula from 4.1
        """
        start_time = time.time()
        training_history = []
        validation_data = self.config.sample(self.batch_size)

        # compute Lambda(t) = sigma(t)**2 + mean_jump ** 2 + std_jump ** 2
        print("Lambda(0): ", self.Lambda(0).item())

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
                    u_star_at_t.append(self.u_star(t, X_BX).detach().cpu().numpy()[0,0])
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
                X_BX = X_BX + self.config.drift(t[i], X_BX, self.u_star(t[i], X_BX)) * self.config.delta_t + \
                torch.einsum('bxw,bw->bx', self.config.diffusion(t[i], X_BX, self.u_star(t[i], X_BX)), delta_W[i, :, :]) + \
                self.u_star(t[i], X_BX) * torch.einsum('blc,blcx->bx', jump_mask[i, :, :, :], jump_sizes) + \
                - self.config.jump_intensity[0] * self.config.jump_size_mean[0] * self.u_star(t[i], X_BX) * self.config.delta_t # NOTE: why with a minus sign? # REPLY: because this compensation term is intended to cancel out the jumps "on average" (over time and probability space)

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

class Solver(object):
    def __init__(self, x_0_value=1.0):
        self.config = Config()

        self.valid_size = self.config.valid_size
        self.batch_size = self.config.batch_size
        self.num_iterations = self.config.num_iterations
        self.logging_frequency = self.config.logging_frequency
        self.lr_values = self.config.lr_values
        self.lr_boundaries = self.config.lr_boundaries
        self.x_0_value = x_0_value

        self.model = WholeNet().to(device)  # Move model to the selected device
        print("y_intial: ", self.model.p_init.detach().cpu().numpy())
        print("x_0_value: ", x_0_value)

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
                    loss_, performance_, ratio_ = self.model(validation_data)
                    p_init_euclidean = np.linalg.norm(self.model.p_init.detach().cpu().numpy())
                    elapsed_time = time.time() - start_time
                    print(f"step: {step:5d}, loss: {loss_.item():.4e}, ||Y0||: {p_init_euclidean:.4e}, cost functional: {-performance_.item():.4e}, "
                          f"elapsed time: {int(elapsed_time):3d}, ratio: {ratio_.item():.4e}")
                    validation_history[step, 0] = step
                    validation_history[step, 1] = loss_.item()
                    validation_history[step, 2] = performance_.item()
            else:
                validation_history[step, 0] = None

            # Gradient descent
            self.optimizer.zero_grad()
            loss, performance, ratio = self.model(self.config.sample(self.batch_size))
            training_history[step, 0] = step
            training_history[step, 1] = loss.item()
            training_history[step, 2] = performance.item()
            loss.backward()
            self.optimizer.step()

            self.scheduler.step()

        self.training_history = training_history
        validation_history = validation_history[validation_history[:, 0] != None]  # Remove rows where step is None
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

    def generate_trajectoy(self, sample_size, x_0_value = 1.0):
        """Generate the trajectory of the model. start from the initial value of X0 and apply the control generated by the FFNetU"""

        # Generate sample data for simulation
        sample_data = self.config.sample(sample_size)
        delta_W_TBW, _, jump_mask_TBLC, jump_sizes_BLCX = sample_data

        X_BX = self.config.X_init.repeat(sample_size, 1) * x_0_value  # Shape: (sample_size, dim_X)
        trajectory = []
        trajectory.append(X_BX.detach().cpu().numpy())  # Store initial state
        t = np.arange(0, self.config.time_step_count) * self.config.delta_t

        for i in range(0, self.config.time_step_count):
            u_BU = self.model.u_net((t[i], X_BX))  # Access u_net through self.model

            X_BX = X_BX + self.config.drift(t[i], X_BX, u_BU) * self.config.delta_t + \
                torch.einsum('bxw,bw->bx', self.config.diffusion(t[i], X_BX, u_BU), delta_W_TBW[i, :, :])
            for l in range(self.config.dim_L):
                X_BX = X_BX + u_BU * torch.einsum('bc,bcx->bx', jump_mask_TBLC[i, :, l, :], jump_sizes_BLCX[:, l, :, :])
                X_BX = X_BX - self.config.jump_intensity[l] * (np.exp(self.config.log_normal_mu[l] + 0.5 * self.config.log_normal_sigma[l] ** 2) - 1)* u_BU * self.config.delta_t

            trajectory.append(X_BX.detach().cpu().numpy())

        return np.array(trajectory)

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
        plt.show()

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
        self.u_net = FNNetU()
        for _ in range(self.config.dim_L):
            self.r_nets.append(FNNetR())

    def forward(self, sample, x_0_value=1.0, training=True):
        delta_W_TBW, jump_times_BLC, jump_mask_TBLC, jump_sizes_BLCX = sample
        sample_size = delta_W_TBW.shape[1]
        jump_counts = jump_mask_TBLC.shape[3]
        t = np.arange(0, self.config.time_step_count) * self.config.delta_t
        X_BX = self.config.X_init.repeat(sample_size, 1) * x_0_value  # Shape: (sample_size, dim_X)

        for i in range(0, self.config.time_step_count):
            u_BU = self.u_net((t[i], X_BX))   # Shape: (sample_size, dim_u)

            # Update X
            X_BX = X_BX + self.config.drift(t[i], X_BX, u_BU) * self.config.delta_t + \
                torch.einsum('bxw,bw->bx', self.config.diffusion(t[i], X_BX, u_BU), delta_W_TBW[i, :, :])
            for l in range(self.config.dim_L):
                X_BX = X_BX + u_BU * torch.einsum('bc,bcx->bx', jump_mask_TBLC[i, :, l, :], jump_sizes_BLCX[:, l, :, :])
                X_BX = X_BX - self.config.jump_intensity[l] * (np.exp(self.config.log_normal_mu[l] + 0.5 * self.config.log_normal_sigma[l] ** 2) - 1)* u_BU * self.config.delta_t

        performance = torch.mean(self.config.g(X_BX))

        return -performance, performance, performance


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




def main():
    # Set the random seed for reproducibility
    torch.manual_seed(42)
    config = Config()
    print("starting the code")
    print("target mean A: ", config.TARGET_MEAN_A)

    # config.sample_stock_price(sample_size=10)
    x_0_values = np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
    V_for_different_x0 = []
    std_for_different_x0 = []
    for x_0 in x_0_values:
        print('\n\n\n x_0: ', x_0)
        # Initialize the solver with the initial value of X0
        solver = Solver(x_0_value=x_0)
        closed_form_solver = ClosedFormSolver(x_0_value=x_0)
        solver.train()
        # generate the trajectory
        trajectory = solver.generate_trajectoy(256, x_0_value=x_0)
        # plot the trajectory

        solver.plot_trajectory(trajectory)

        # print('Trajectory shape after the plot: ', trajectory.shape)
        # compute the cost functional for each of the trajectories. get the mean and std
        cost_functional = np.mean(np.sum((trajectory - config.TARGET_MEAN_A) ** 2, axis=2), axis=0)
        print("cost finctional shape: ", cost_functional.shape)
        print('Cost functional: ', cost_functional)
        print('Mean: ', np.mean(cost_functional))
        print('Std: ', np.std(cost_functional))
        V_for_different_x0.append(np.mean(cost_functional))
        std_for_different_x0.append(np.std(cost_functional))

        # Make a heatmap showing the difference between FNNetU and closed_form_solver.u_star
        t_values = np.linspace(0, config.terminal_time, 256)
        x_values = torch.linspace(0, 2, 256, dtype=torch.float32).unsqueeze(1).to(device)
        trained_values = np.zeros((len(t_values), len(x_values)))
        closed_form_values = np.zeros((len(t_values), len(x_values)))
        for i in range(len(t_values)):
            trained_values[i, :] = solver.model.u_net((t_values[i], x_values)).detach().cpu().numpy().flatten()
            closed_form_values[i, :] = closed_form_solver.u_star(t_values[i], x_values).detach().cpu().numpy().flatten()

        value_difference_abs = np.abs(trained_values - closed_form_values)
        # Plot the heatmap
        plt.figure(figsize=(8, 6))
        plt.imshow(value_difference_abs, aspect='auto', extent=[0, 2, 0, config.terminal_time], origin='lower', cmap='viridis')
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('t')
        plt.grid()
        plt.show()


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
    print(" trying the better time scaling")
    print(f" num_iterations = 200 ,  LAMBDA =  {LAMBDA}, B = 512, A = {config.TARGET_MEAN_A}")
    print("Cost Functional for different x_0 values:")
    print("Mean: \n", V_for_different_x0)
    print("Std: \n", std_for_different_x0)
    # save the plot
    plt.savefig('cost_functional_2.png')
    plt.close()


if __name__ == '__main__':
    main()