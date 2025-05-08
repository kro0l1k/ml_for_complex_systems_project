import numpy as np
import matplotlib.pyplot as plt

def sample_jump_process(T=1.0, lambda_rate=5.0, jump_mean=1.0, jump_std=0.5, seed=None):
    """
    Sample a compound Poisson jump process on [0, T].
    
    Parameters:
    -----------
    T : float
        The time horizon
    lambda_rate : float
        The rate parameter of the Poisson process (average number of jumps per unit time)
    jump_mean : float
        The mean of the jump size distribution
    jump_std : float
        The standard deviation of the jump size distribution
    seed : int or None
        Random seed for reproducibility
        
    Returns:
    --------
    times : np.ndarray
        The times at which jumps occur
    values : np.ndarray
        The cumulative values of the process at each time
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Sample the number of jumps from a Poisson distribution
    n_jumps = np.random.poisson(lambda_rate * T)
    
    # If no jumps, return empty arrays
    if n_jumps == 0:
        return np.array([0, T]), np.array([0, 0])
    
    # Sample the jump times uniformly on [0, T]
    jump_times = np.random.uniform(0, T, n_jumps)
    jump_times = np.sort(jump_times)  # Sort the jump times
    
    # Sample the jump sizes from a normal distribution
    jump_sizes = np.random.normal(jump_mean, jump_std, n_jumps)
    
    # Compute the process values (cumulative sum of jumps)
    process_values = np.cumsum(jump_sizes)
    
    # Include the starting point (0, 0)
    times = np.concatenate(([0], jump_times, [T]))
    values = np.concatenate(([0], process_values, [process_values[-1]]))
    
    return times, values

def sample_combined_process(T=1.0, lambda_rate=5.0, jump_mean=1.0, jump_std=0.5, diffusion_std=0.1, seed=None):
    """
    Sample a combined process consisting of a diffusion process and a jump process on [0, T].
    
    Parameters:
    -----------
    T : float
        The time horizon
    lambda_rate : float
        The rate parameter of the Poisson process (average number of jumps per unit time)
    jump_mean : float
        The mean of the jump size distribution
    jump_std : float
        The standard deviation of the jump size distribution
    diffusion_std : float
        The standard deviation of the diffusion process (Brownian motion)
    seed : int or None
        Random seed for reproducibility
        
    Returns:
    --------
    times : np.ndarray
        The times at which jumps occur
    values : np.ndarray
        The cumulative values of the combined process at each time
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Sample the number of jumps from a Poisson distribution
    n_jumps = np.random.poisson(lambda_rate * T)
    
    # If no jumps, return a pure diffusion process
    if n_jumps == 0:
        times = np.linspace(0, T, 1000)
        diffusion_values = np.cumsum(np.random.normal(0, diffusion_std * np.sqrt(T / 1000), len(times)))
        return times, diffusion_values
    
    # Sample the jump times uniformly on [0, T]
    jump_times = np.random.uniform(0, T, n_jumps)
    jump_times = np.sort(jump_times)  # Sort the jump times
    
    # Sample the jump sizes from a normal distribution
    jump_sizes = np.random.normal(jump_mean, jump_std, n_jumps)
    
    # Compute the jump process values (cumulative sum of jumps)
    jump_values = np.cumsum(jump_sizes)
    
    # Include the starting point (0, 0) for the jump process
    jump_times = np.concatenate(([0], jump_times, [T]))
    jump_values = np.concatenate(([0], jump_values, [jump_values[-1]]))
    
    # Generate the diffusion process
    diffusion_times = np.linspace(0, T, 1000)
    diffusion_values = np.cumsum(np.random.normal(0, diffusion_std * np.sqrt(T / 1000), len(diffusion_times)))
    
    # Combine the jump and diffusion processes
    combined_times = np.union1d(jump_times, diffusion_times)
    combined_values = np.interp(combined_times, jump_times, jump_values) + np.interp(combined_times, diffusion_times, diffusion_values)
    
    return combined_times, combined_values

def sample_multiple_combined_processes(m=10, T=1.0, lambda_rate=5.0, jump_mean=1.0, jump_std=0.5, diffusion_std=0.1, seed=None):
    """
    Sample multiple combined processes consisting of a diffusion process and a jump process on [0, T].
    
    Parameters:
    -----------
    m : int
        The number of paths to sample
    T : float
        The time horizon
    lambda_rate : float
        The rate parameter of the Poisson process (average number of jumps per unit time)
    jump_mean : float
        The mean of the jump size distribution
    jump_std : float
        The standard deviation of the jump size distribution
    diffusion_std : float
        The standard deviation of the diffusion process (Brownian motion)
    seed : int or None
        Random seed for reproducibility
        
    Returns:
    --------
    paths : list of tuples
        A list where each element is a tuple (times, values) representing a sampled path
    """
    if seed is not None:
        np.random.seed(seed)
    
    paths = []
    for i in range(m):
        times, values = sample_combined_process(T, lambda_rate, jump_mean, jump_std, diffusion_std, seed=np.random.randint(0, 1e6))
        paths.append((times, values))
    
    return paths

# Example usage
def plot_jump_process(times, values):
    """
    Plot a jump process.
    """
    plt.figure(figsize=(10, 6))
    plt.step(times, values, where='post')
    plt.scatter(times[1:-1], values[1:-1], color='red', zorder=3)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Sample Path of a Jump Process')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_combined_process(times, values):
    """
    Plot a combined process.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(times, values, label='Combined Process')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Sample Path of a Combined Process (Diffusion + Jump)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def plot_multiple_combined_processes(paths):
    """
    Plot multiple combined processes.
    """
    plt.figure(figsize=(10, 6))
    for i, (times, values) in enumerate(paths):
        plt.plot(times, values, label=f'Path {i+1}', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Sample Paths of Combined Processes (Diffusion + Jump)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

# Generate and plot multiple sample paths
paths = sample_multiple_combined_processes(m=10, T=100.0, lambda_rate=2.0, jump_mean=0., diffusion_std=0.2, seed=42)
plot_multiple_combined_processes(paths)