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
# settled on  num_iterations = 200 ,  LAMBDA =  0.05, B = 512, A = 1.1
our_algorithm_results =    [np.float32(0.031073641), np.float32(0.018550556), np.float32(0.009279302), np.float32(0.0038568017), np.float32(0.0040464764), np.float32(0.0024469951), np.float32(0.008800172), np.float32(0.021375414)]
our_alg_std = [np.float32(0.0052901735), np.float32(0.01924247), np.float32(0.016293189), np.float32(0.005592685), np.float32(0.03364823), np.float32(0.007405035), np.float32(0.022660717), np.float32(0.045441333)]
# our_algorithm_results = np.array( [np.float32(0.031412832), np.float32(0.018074248), np.float32(0.011914069), np.float32(0.0040811445), np.float32(0.0026358007), np.float32(0.0020323012), np.float32(0.0070558796), np.float32(0.021787804)])
# vals_for_u0 = np.array( [0.03155143, 0.016037878, 0.005783389, 0.0031538783, 0.00078718434, 0.0010494824, 0.0065703387, 0.017349776])
#  LAMBDA = 0.001, B = 256, A = 1.2
# our_algorithm_results = np.array(  [np.float32(0.031412832), np.float32(0.018074248), np.float32(0.011914069), np.float32(0.0040811445), np.float32(0.0026358007), np.float32(0.0020323012), np.float32(0.0070558796), np.float32(0.021787804)] )
# for A = 1.2
# our_algorithm_results = np.array( [np.float32(0.07748832), np.float32(0.052262686), np.float32(0.03385166), np.float32(0.02337157), np.float32(0.015910007), np.float32(0.005850268), np.float32(0.0020789257), np.float32(0.013073487)]
# )
# std_our_algg = np.array( [np.float32(0.006415405), np.float32(0.008366791), np.float32(0.020514537), np.float32(0.004286777), np.float32(0.0025299753), np.float32(0.0068915472), np.float32(0.005344459), np.float32(0.028364524)])
# ran for M = 100

# for M = 256
true_values = np.array( [0.023964712845498816, 0.010325144818675835, 0.002278464330915918, 0.0010168161701622671, 0.00020701913938189323, 0.0032401091163525887, 0.011496853325030325, 0.025134232140540808])
# true_values_new = np.array( [0.022397732276267887, 0.009587086648211082, 0.0028601783180159483, 0.0013082264232344358, 0.0007534595305754106, 0.004602796383914046, 0.014335846425772774, 0.028928644990341375])
# for M = 1000
# true_values = np.array([0.023149361134447453, 0.009755122156570376, 0.0024464571194930088, 0.0009795407322678073, 0.0002289617547097965, 0.003261895807451209, 0.01155618357448535, 0.024793297469137034])
# for A = 1.2
# true_values = np.array( [0.062395374576003804, 0.038884960950503, 0.021898599144130504, 0.01631074293357849, 0.009159145685796347, 0.002007531882674692, 0.00032921513393286637, 0.0038824295834506044])


# more physical approach:
# x0_values = np.array([1.0, 1.01,  1.02, 1.03, 1.04, 1.05, 1.06])
# true_values = np.array([0.0026165481913986506, 0.001534379514795969, 0.0009831782837829105, 0.00042183499580209947, 0.00020595951017073072, 0.00020419537122024032, 0.00038109061774971244])
# our_algorithm_results = np.array( [np.float32(0.0071333777), np.float32(0.005329078), np.float32(0.0070927935), np.float32(0.007179645), np.float32(0.0031143844), np.float32(0.0017217874), np.float32(0.0031677773)]
# )

# our_algorithm_results = np.array( [np.float32(0.017480899), np.float32(0.009430798), np.float32(0.031805538), np.float32(0.0030702353), np.float32(0.0045361393), np.float32(0.0052178325), np.float32(0.004597137)])


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
