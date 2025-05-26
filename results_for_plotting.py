import numpy as np

x0_values = np.array([0.9, 0.95, 1.0, 1.02, 1.05, 1.1, 1.15, 1.2])

# results for M = 100
# our_algorithm_results = np.array([0.031538915, 
#                                   0.016092524,
#                                   0.01146256, 
#                                   0.004226983, 
#                                   0.0022009457, 
#                                   0.0039236657, 
#                                   0.005980258, 
#                                   0.019229744])

# Results for M = 50
# our_algorithm_results = np.array([0.030980885, 
#                                   0.015923709, 
#                                   0.010522552, 
#                                   0.006315307,
#                                   0.0018191382, 
#                                   0.0014189442,
#                                   0.009686759,
#                                   0.015931945])
# results for large B, large M to compute (X_T -a)**2
our_algorithm_results =  np.array( [np.float32(0.030942392), np.float32(0.017765235), np.float32(0.008979618), np.float32(0.0052980892), np.float32(0.0013526499), np.float32(0.0026162344), np.float32(0.008263119), np.float32(0.020444945)]
)
    
std_our_algg = np.array( [np.float32(0.006415405), np.float32(0.008366791), np.float32(0.020514537), np.float32(0.004286777), np.float32(0.0025299753), np.float32(0.0068915472), np.float32(0.005344459), np.float32(0.028364524)]
)
# ran for M = 100
# true_values = np.array([0.03659518733146482, 0.01963405121084947, 0.01116546668904079, 0.010694841515995835, 0.010992423650850482, 0.016943291072830702, 0.027179290188189938, 0.042433316633389395])


# for M = 1000
true_values = np.array([0.03207735203787802, 0.021498543625317196, 0.01593552436185347, 0.012522989224872437, 0.012249100500414382, 0.017893475098775966, 0.025976433414267603, 0.040595823872162894])
import matplotlib.pyplot as plt

# Set up the plot with professional styling
plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.size': 14})

# Plot both datasets
plt.plot(x0_values, true_values, 'o-', linewidth=2.5, markersize=8, 
            label='True Values', color='#2E86AB', markerfacecolor='white', 
            markeredgewidth=2)
plt.plot(x0_values, our_algorithm_results, 's-', linewidth=2.5, markersize=8, 
            label='Algorithm Results', color="#700A41", markerfacecolor='white', 
            markeredgewidth=2)
# Add error bars for the algorithm results - shaded region beteen mean - std and mean + std
# plt.fill_between(x0_values,
#                     our_algorithm_results , 
#                     our_algorithm_results, 
#                     color='#A23B72', alpha=0.2, label='Standard Deviation Range')


# Formatting for academic presentation
plt.xlabel('Initial Value ($x_0$)', fontsize=16, fontweight='bold')
plt.ylabel('Error Metric', fontsize=16, fontweight='bold')
plt.title('Algorithm Performance vs True Values', fontsize=18, fontweight='bold', pad=20)

# Grid and styling
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=14, frameon=True, fancybox=True, shadow=True)

# Set axis limits with padding
plt.xlim(0.88, 1.22)
plt.ylim(-0.005, 0.045)

# Tick formatting
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Tight layout for better spacing
plt.tight_layout()
plt.show()
