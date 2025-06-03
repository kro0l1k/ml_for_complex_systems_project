import numpy as np

x0_values = np.array([100, 110, 120, 130, 140, 150, 160])

our_algorithm_results =     [np.float32(1585.0469), np.float32(847.426), np.float32(353.0327), np.float32(106.16183), np.float32(0), np.float32(349.6094), np.float32(839.44543)]

true_values =  [859.918319015969, 637.9473402889879, 269.64025440241966, 348.8934937943913, 626.4771555870955, 1200.1056346502692, 1984.0407278353455]

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
#plt.xlim(0.88, 1.22)
#plt.ylim(-0.005, 0.045)

# Tick formatting
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Tight layout for better spacing
plt.tight_layout()
plt.show()
