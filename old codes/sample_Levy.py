import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson

def simulate_levy_process(b, A, nu, T=1.0, n_steps=1000, n_samples=1, epsilon=0.1, max_jumps=1000, seed=None):
    """
    Simulate samples from a Lévy process with given Lévy triplet.
    
    Parameters:
    -----------
    b : float or np.ndarray
        Drift vector
    A : float or np.ndarray
        Diffusion coefficient (for 1D) or covariance matrix (for multi-D)
    nu : callable
        Lévy measure. Should take a size parameter and return random jumps
        according to the measure, or a tuple (rate, jump_dist) where rate is the
        intensity of a Poisson process and jump_dist is a callable that generates
        random jump sizes
    T : float
        Time horizon
    n_steps : int
        Number of time steps
    n_samples : int
        Number of sample paths to generate
    epsilon : float
        Threshold for separating large and small jumps
    max_jumps : int
        Maximum number of jumps to simulate for large jumps
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    times : np.ndarray
        Time points
    paths : np.ndarray
        Simulated paths of shape (n_samples, n_steps+1)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Handle 1D and multi-D cases
    is_1d = np.isscalar(b) or (isinstance(b, np.ndarray) and b.size == 1)
    
    if is_1d:
        b = float(b)
        if np.isscalar(A):
            sigma = np.sqrt(A)
        else:
            sigma = np.sqrt(A.item())
        
        # Initialize paths
        dt = T / n_steps
        times = np.linspace(0, T, n_steps + 1)
        paths = np.zeros((n_samples, n_steps + 1))
        
        # Simulate each path
        for i in range(n_samples):
            # Drift and diffusion (Brownian motion) component
            increments = b * dt + sigma * np.sqrt(dt) * np.random.randn(n_steps)
            
            # Jump component (large jumps)
            if callable(nu):
                # If nu is a function that generates jumps directly
                large_jumps = simulate_large_jumps(nu, T, epsilon, max_jumps)
            else:
                # If nu is a tuple (rate, jump_dist)
                rate, jump_dist = nu
                large_jumps = simulate_compound_poisson(rate, jump_dist, T, max_jumps)
            
            # Add large jumps to path
            for jump_time, jump_size in large_jumps:
                if jump_time <= T:  # Ensure jump is within time horizon
                    # Find the index of the closest time point after the jump
                    idx = np.searchsorted(times, jump_time)
                    if idx <= n_steps:  # Ensure jump is within time steps
                        paths[i, idx:] += jump_size
            
            # Calculate path by cumulating the increments
            paths[i, 1:] = np.cumsum(increments)
            
            # Add small jumps (approximation by a Brownian motion with adjusted variance)
            # This is an approximation, using the fact that small jumps behave like a Brownian motion
            small_jump_var = estimate_small_jump_variance(nu, epsilon)
            small_jump_sigma = np.sqrt(small_jump_var / T * dt)
            paths[i, 1:] += np.cumsum(small_jump_sigma * np.random.randn(n_steps))
    else:
        # For multi-dimensional case - implementation would follow similar logic
        # but handling vectors and matrices. Omitted for simplicity.
        raise NotImplementedError("Multi-dimensional Lévy process simulation not implemented.")
    
    return times, paths

def simulate_large_jumps(nu, T, epsilon, max_jumps):
    """
    Simulate large jumps for a Lévy process.
    
    Parameters:
    -----------
    nu : callable
        Function that takes a size parameter and returns random jumps according to the Lévy measure
    T : float
        Time horizon
    epsilon : float
        Threshold for large jumps
    max_jumps : int
        Maximum number of jumps to simulate
    
    Returns:
    --------
    List of tuples (jump_time, jump_size) for jumps larger than epsilon
    """
    # Generate potential jumps
    potential_jumps = nu(max_jumps)
    large_jumps = []
    
    # Filter for jumps larger than epsilon (in absolute value)
    large_indices = np.where(np.abs(potential_jumps) > epsilon)[0]
    n_large_jumps = len(large_indices)
    
    if n_large_jumps > 0:
        # Generate random jump times uniformly in [0, T]
        jump_times = np.random.uniform(0, T, n_large_jumps)
        
        # Combine jump times and sizes
        for i in range(n_large_jumps):
            large_jumps.append((jump_times[i], potential_jumps[large_indices[i]]))
        
        # Sort by jump time
        large_jumps.sort(key=lambda x: x[0])
    
    return large_jumps

def simulate_compound_poisson(rate, jump_dist, T, max_jumps):
    """
    Simulate a compound Poisson process.
    
    Parameters:
    -----------
    rate : float
        Intensity of the Poisson process
    jump_dist : callable
        Function that takes a size parameter and returns random jump sizes
    T : float
        Time horizon
    max_jumps : int
        Maximum number of jumps to simulate
    
    Returns:
    --------
    List of tuples (jump_time, jump_size)
    """
    # Generate Poisson random number for jump count
    n_jumps = min(poisson.rvs(rate * T), max_jumps)
    
    jumps = []
    if n_jumps > 0:
        # Generate jump times (uniformly distributed in [0, T])
        jump_times = np.sort(np.random.uniform(0, T, n_jumps))
        
        # Generate jump sizes
        jump_sizes = jump_dist(n_jumps)
        
        # Combine jump times and sizes
        for i in range(n_jumps):
            jumps.append((jump_times[i], jump_sizes[i]))
    
    return jumps

def estimate_small_jump_variance(nu, epsilon):
    """
    Estimate the variance of small jumps for approximation by Brownian motion.
    This is a placeholder and would need to be adapted to the specific Lévy measure.
    
    Parameters:
    -----------
    nu : callable or tuple
        Lévy measure or tuple (rate, jump_dist)
    epsilon : float
        Threshold for small jumps
    
    Returns:
    --------
    float
        Estimated variance of small jumps
    """
    # This is a simplified approximation
    # In practice, this would involve integrating x^2 over the Lévy measure for |x| < epsilon
    # Here we return a placeholder value
    return 0.1 * epsilon**2

# Example usage for a specific Lévy process:

def normal_inverse_gaussian_measure(alpha, beta, delta):
    """
    Create a Normal Inverse Gaussian (NIG) Lévy measure.
    
    Parameters:
    -----------
    alpha : float
        Tail heaviness parameter
    beta : float
        Asymmetry parameter
    delta : float
        Scale parameter
    
    Returns:
    --------
    Tuple (rate, jump_dist) for compound Poisson approximation
    """
    def jump_dist(size):
        # Simplified approximation - in practice would simulate from NIG distribution
        return np.random.standard_t(df=3, size=size) * delta / alpha
    
    # Rate parameter for the Poisson process (approximation)
    rate = delta * alpha
    
    return (rate, jump_dist)

def simulate_nig_process(alpha, beta, delta, mu, T=1.0, n_steps=1000, n_samples=1, seed=None):
    """
    Simulate a Normal Inverse Gaussian (NIG) Lévy process.
    
    Parameters:
    -----------
    alpha : float
        Tail heaviness parameter
    beta : float
        Asymmetry parameter
    delta : float
        Scale parameter
    mu : float
        Location parameter
    T : float
        Time horizon
    n_steps : int
        Number of time steps
    n_samples : int
        Number of sample paths
    seed : int, optional
        Random seed
    
    Returns:
    --------
    times : np.ndarray
        Time points
    paths : np.ndarray
        Simulated paths
    """
    # Set the Lévy triplet for NIG
    b = mu  # Simplified drift
    A = 0.0  # No Brownian component in pure NIG
    nu = normal_inverse_gaussian_measure(alpha, beta, delta)
    
    return simulate_levy_process(b, A, nu, T, n_steps, n_samples, epsilon=0.01, seed=seed)

def plot_levy_process_paths(times, paths, title="Lévy Process Simulation"):
    """
    Plot simulated Lévy process paths.
    
    Parameters:
    -----------
    times : np.ndarray
        Time points
    paths : np.ndarray
        Simulated paths
    title : str
        Plot title
    """
    plt.figure(figsize=(10, 6))
    for i in range(paths.shape[0]):
        plt.plot(times, paths[i], lw=1)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.ylim(-1, 1)  # Set y-axis limits

    plt.grid(True)
    plt.show()

# Example: Simulate and plot a Normal Inverse Gaussian (NIG) Lévy process
if __name__ == "__main__":
    
    # NIG parameters
    alpha = 10000.5   # Tail heaviness
    beta = 1000.0    # Symmetry
    delta = 1000000.0   # Scale
    mu = 0.0      # Location
    
    # Simulation parameters
    T = 100.0       # Time horizon
    n_steps = 1000000  # Time steps
    n_samples = 5  # Number of paths
    
    # Simulate NIG process
    times, paths = simulate_nig_process(alpha, beta, delta, mu, T, n_steps, n_samples, seed=42)
    
    # Plot the result
    plot_levy_process_paths(times, paths, title=f"Normal Inverse Gaussian Lévy Process (α={alpha}, β={beta}, δ={delta}, μ={mu})")