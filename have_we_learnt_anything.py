import numpy as np
import matplotlib.pyplot as plt

TARGET_MEAN_A = 1.1

def main():
    
    our_algorithm_results =    [np.float32(0.031073641), np.float32(0.018550556), np.float32(0.009279302), np.float32(0.0038568017), np.float32(0.0040464764), np.float32(0.0024469951), np.float32(0.008800172), np.float32(0.021375414)]
    our_alg_std = [np.float32(0.0052901735), np.float32(0.01924247), np.float32(0.016293189), np.float32(0.005592685), np.float32(0.03364823), np.float32(0.007405035), np.float32(0.022660717), np.float32(0.045441333)]
    x0_values = np.array([0.9, 0.95, 1.0, 1.02, 1.05, 1.1, 1.15, 1.2])

    
    lower_bound = np.array(our_algorithm_results) - np.array(our_alg_std)
    lower_bound[lower_bound < 0] = 0  # Ensure lower bound is not negative
    
    ### plot the cost functional for different x_0 values. add a transparent area for +-1 std
    plt.figure()
    plt.plot(x0_values, our_algorithm_results, label='Mean Cost Functional')
    plt.fill_between(x0_values, 
                     lower_bound,
                     np.array(our_algorithm_results) + np.array(our_alg_std), 
                     alpha=0.2, label='1 Std Dev')
    plt.title('Cost Functional for Different Initial Values of X0, and learnt control')
    plt.xlabel('Initial Value of X0')
    plt.ylabel('Cost Functional')
    plt.legend()
    plt.grid()
    plt.show()
   
    # save the plot
    plt.savefig('cost_functional_2.png')
    plt.close()
    

if __name__ == '__main__':
    main()