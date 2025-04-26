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
        self.valid_size = 513
        self.B = 256 # batch size
        self.num_iterations = 5000 # number of epochs to train
        self.logging_frequency = 200
        self.lr_values = [5e-3, 5e-3, 5e-3]
        
        self.lr_boundaries = [2000, 4000]
        self.config = Config()

        self.model = WholeNet().to(device) 
        self.y_init_1d = self.model.y_init_1d
        print("y_initial: ", self.y_init_1d.shape)  # (1, d)
        
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
        
        #### Validation data: ###
        dW = self.config.sample(self.valid_size)
        print("validation dW shape: ", dW.shape) # (512, 100, 25) (M = valid size, d,T)
        t_jump = self.config.sample_jump_times(self.valid_size)
        print("validation t_jump shape: ", t_jump.shape) # (512, 100, 25) (M = valid size, d,T)
        jump_size = self.config.sample_jump_sizes(self.valid_size, shape_to_sample=t_jump.shape)
        print("validation jump_size shape: ", jump_size.shape)
        valid_dW = dW.to(device)  # Ensure data is on the correct device
        valid_jump_times = t_jump.to(device)
        valid_jump_sizes = jump_size.to(device) 
        valid_support_points = self.config.sample_support_points(self.valid_size).to(device)  
        
        for step in range(self.num_iterations+1):
            # Custom learning rate adjustment
            if step in self.lr_boundaries:
                idx = self.lr_boundaries.index(step) + 1
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr_values[idx]
            
            if step % self.logging_frequency == 0:
                with torch.no_grad():
                    loss, cost, ratio = self.model(valid_dW, valid_jump_times, valid_jump_sizes, valid_support_points, training = False)
                    y_init_1d = self.y_init_1d.detach().cpu().numpy()[0][0]
                    elapsed_time = time.time() - start_time
                    training_history.append([step, cost.item(), y_init_1d, loss.item(), ratio.item()])
                    print(f"step: {step:5d}, loss: {loss.item():.4e}, Y0: {y_init_1d:.4e}, cost: {cost.item():.4e}, "
                          f"elapsed time: {int(elapsed_time):3d}, ratio: {ratio.item():.4e}")

            # sample new train data (we do this every iteration!)
            dW = self.config.sample(self.B).to(device)  
            jump_times = self.config.sample_jump_times(self.B).to(device)
            jump_sizes = self.config.sample_jump_sizes(self.B, shape_to_sample=jump_times.shape).to(device)
            support_points = self.config.sample_support_points(self.B).to(device)
            
            self.train_step(dW, jump_times, jump_sizes, support_points)  
            
            self.scheduler.step()
        
        self.training_history = training_history

    def train_step(self, train_data, jump_times, jump_sizes, support_points):
        """Updating the gradients"""
        self.optimizer.zero_grad()
        loss, _, _ = self.model(train_data, jump_times, jump_sizes, support_points)
        loss.backward()
        self.optimizer.step()

class WholeNet(torch.nn.Module):
    """Building the neural network architecture.
        a forward pass should return [loss, cost, ratio]
        
        naming convention:
        B = batch size
        M = number of monte carlo samples
        d = dimension of the process d = 100
        T = time horizon
        
    """
    def __init__(self):
        super(WholeNet, self).__init__()
        self.config = Config()
        
        # Initialize y_init as a parameter
        self.y_init_1d = torch.nn.Parameter(
            torch.randn(1, self.config.dim_y, dtype=torch.float32).to(device),  # float32 and correct device
            requires_grad = True # needed to find this value.
        )
        self.q_net = FNNetQ()
        self.u_net = FNNetU()
        self.r_net = FFNetR()

    def forward(self, dw_BdT, jump_times_BdN_j, jump_sizes_BdN_j, support_points_Bd, training = True):
        # Ensure input tensor is on the correct device
        
        dw_BdT = dw_BdT.to(device)
        #print("dw shape: ", dw.shape)  # (B, d, T)
        jump_times_BdN_j = jump_times_BdN_j.to(device)
        print("jump_times shape: ", jump_times_BdN_j.shape)  # (B, d, N_j)
        jump_sizes_BdN_j = jump_sizes_BdN_j.to(device)
        print("jump_sizes shape: ", jump_sizes_BdN_j.shape)
        # (B, d, N_j)
        support_points_Bd = support_points_Bd.to(device)
        print("support_points shape: ", support_points_Bd.shape)
        # (B, d)
        
        x_init = torch.ones(1, self.config.dim_x, dtype=torch.float32).to(device) * 0.0
        time_stamp = np.arange(0, self.config.num_time_interval) * self.config.delta_t
        all_one_vec = torch.ones(dw_BdT.shape[0], 1, dtype=torch.float32).to(device)
        x_Bd = torch.matmul(all_one_vec, x_init)
        y_Bd = torch.matmul(all_one_vec, self.y_init_1d)
        print("y shape: ", y_Bd.shape)  # (B, M)
        l = 0.0  # The cost functional
        H = 0.0  # The constraint term
        J = 0.0  # The jump term
        
        for t in range(0, self.config.num_time_interval):
            data = (time_stamp[t], x_Bd)
            print("data shape: ", time_stamp[t], x_Bd.shape)  # (B, M)
            u_Bd = self.u_net(data) 
            print("u shape: ", u_Bd.shape)  # (B, M)
            z = self.q_net(data)

            l = l + self.config.f_fn(time_stamp[t], x_Bd, u_Bd) * self.config.delta_t
            H = H + self.config.Hu_fn(time_stamp[t], x_Bd, y_Bd, z, u_Bd)
            b_ = self.config.b_fn(time_stamp[t], x_Bd, u_Bd)
            # print("b_ shape: ", b_.shape)  # (B, d)
            sigma_ = self.config.sigma_fn(time_stamp[t], x_Bd, u_Bd)
            f_ = self.config.Hx_fn(time_stamp[t], x_Bd, u_Bd, y_Bd, z)
            
            ### JUMP TERM for x update ###
            # we will do it with bit masking : first create a mask for jump times <= t
            mask_lower = (jump_times_BdN_j[:, :, :] <= time_stamp[t])
            
            x_Bd = x_Bd + b_ * self.config.delta_t + sigma_ * dw_BdT[:, :, t]
            y_Bd = y_Bd - f_ * self.config.delta_t + z * dw_BdT[:, :, t]

        delta = y_Bd + self.config.hx_tf(self.config.total_T, x_Bd)
        loss = torch.mean(torch.sum(delta**2, 1, keepdim=True) + LAMBDA * H)
        ratio = torch.mean(torch.sum(delta**2, 1, keepdim=True)) / torch.mean(H)

        l = l + self.config.h_fn(self.config.total_T, x_Bd)
        cost = torch.mean(l)

        return loss, cost, ratio

class FNNetQ(torch.nn.Module):
    """ Define the feedforward neural network """
    def __init__(self):
        super(FNNetQ, self).__init__()
        self.config = Config()
        # this is the larger network config
        # num_hiddens = [self.config.dim_x+10, self.config.dim_x+10, self.config.dim_x+10]
        
        num_hiddens = [self.config.dim_x + 10]
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
        
        # this is the bigger network. for now lets use one layer to speed up the training
        # num_hiddens = [self.config.dim_x+10, self.config.dim_x+10, self.config.dim_x+10]
        num_hiddens = [self.config.dim_x+10]
        
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
        
        # Initialize weights
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

class FFNetR(torch.nn.Module):
    """ Define the feedforward neural network """
    def __init__(self):
        super(FFNetR, self).__init__()
        self.config = Config()
        num_hiddens = [self.config.dim_x+10, self.config.dim_x+10, self.config.dim_x+10]
        
        # Create layer lists
        self.bn_layers = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(
                self.config.dim_x + 2,  # First layer input size
                momentum=0.99,
                eps=1e-6
            )
        ])
        self.dense_layers = torch.nn.ModuleList()
        # Hidden layers
        for i in range(len(num_hiddens)):
            # Input size for the first layer
            input_size = self.config.dim_x + 2 if i == 0 else num_hiddens[i-1]
            
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
            torch.nn.Linear(num_hiddens[-1], self.config.dim_x)
        )
        self.bn_layers.append(
            torch.nn.BatchNorm1d(
                self.config.dim_x,
                momentum=0.99,
                eps=1e-6
            )
            
            
        )
        

    def forward(self, inputs):
        
        """structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense -> bn
        ---- 
        note the inputs are (t, x, z)
        returns scalar value
        ----
        """
        t, x, z = inputs
        ts = torch.ones(x.shape[0], 1, dtype=torch.float32).to(x.device) * t 
        # double check z size
        print(z.shape)  # 
        # Ensure correct shape and device
        x = torch.cat([ts, x, z], dim=1)  # Concatenate along the feature dimension
        
        x = self.bn_layers[0](x)
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = self.bn_layers[i+1](x)
            x = torch.nn.functional.relu(x)
            
        x = self.dense_layers[-1](x)
        x = self.bn_layers[-1](x)
        # print("x shape: ", x.shape)
        # print("x: ", x)
        return x  # returns scalar value



class Config(object):
    """Define the configs in the systems"""
    def __init__(self):
        super(Config, self).__init__()
        self.dim_x = 100 # dimension of the process = d
        self.dim_y = 100 # dimension of the process = d
        self.dim_z = 100 # dimension of the process = d
        self.dim_u = 100 # dimension of the process = d
        
        self.lambda_jumps = 3
        self.jump_mean = 0.5
        self.jump_std = 0.5
        
        self.num_time_interval = 25
        self.total_T = 1.0
        self.delta_t = (self.total_T + 0.0) / self.num_time_interval
        self.sqrth = np.sqrt(self.delta_t)
        self.t_stamp = np.arange(0, self.num_time_interval) * self.delta_t

    ##### SAMPLING FUNCTIONS: for the random walk and jump process #####
    def sample(self, num_sample):
        # sample dW from a normal distribution
        dw_sample = normal.rvs(size=[num_sample, self.dim_x, self.num_time_interval]) * self.sqrth
        dw_sample_tensor = torch.tensor(dw_sample, dtype=torch.float32).to(device)  
        # print("sampled dW: ", dw_sample_tensor.shape)  
        return dw_sample_tensor # (B, d, num_time_interval)
    
    def sample_jump_times(self, num_sample):
        # sample jump times uniformly in [0, T]
        nr_jumps = np.random.poisson(self.lambda_jumps * self.total_T, size=num_sample)
        # print("nr_jumps: ", nr_jumps)  # (B,)
        # sample jump times uniformly in [0, T]. for each m sample nr_jumps[m] jump times. to keep the size const, the rest of the times are set to T
        max_jumps = np.max(nr_jumps)
        jump_times = np.zeros((num_sample, self.dim_x, max_jumps))
        jump_times = jump_times.astype(np.float32)
        
        for m in range(num_sample):
            jump_times[m, :, :nr_jumps[m]] = np.random.uniform(0, self.total_T, size=(self.dim_x, nr_jumps[m]))
            jump_times[m, :, nr_jumps[m]:] = self.total_T  # Fill the rest with T   
            
        # sort the jump times
        jump_times = np.sort(jump_times, axis=2)
        print("sampled jump times: ", jump_times.shape)  # (B, d, N_j)
        jump_times_tensor = torch.tensor(jump_times, dtype=torch.float32).to(device)  
        return jump_times_tensor
    
    def sample_jump_sizes(self, num_sample, shape_to_sample=None):
        # sample jump sizes from a normal distribution
        
        jump_sizes = np.random.normal(self.jump_mean, self.jump_std, size=(shape_to_sample))
        jump_sizes = jump_sizes.astype(np.float32)
        # print("sampled jump sizes: ", jump_sizes.shape)  #  (B, d, N_j)
        jump_sizes_tensor = torch.tensor(jump_sizes, dtype=torch.float32).to(device)
        return jump_sizes_tensor
    
    def sample_support_points(self, num_sample):
        # Sample points to calculate integrals w.r.t. \nu(dz)
        support_points = np.random.uniform(0, 1, size=(num_sample, self.dim_x))
        support_points_tensor = torch.tensor(support_points, dtype=torch.float32).to(device)  # Ensure tensor is on the correct device
        return support_points_tensor
        
    ##### COST FUNCTIONAL AND HAMILTONIAN #####
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