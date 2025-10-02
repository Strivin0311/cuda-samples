import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

# read data
df = pd.read_csv('benchmark.csv', index_col='Kernel')

# deal with missing data
df = df.replace(-1.0, np.nan)

# 转置数据以便分析
df_transposed = df.T
df_transposed.index = df_transposed.index.astype(int)

# process line plot
line_data = df_transposed.reset_index().melt(id_vars=['index'], 
                                           value_vars=df_transposed.columns,
                                           var_name='Kernel', 
                                           value_name='Time (ms)')
line_data.rename(columns={'index': 'Size'}, inplace=True)

# pick some typical sizes
selected_sizes = [1, 512, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216]
bar_data = df_transposed.loc[selected_sizes].reset_index().melt(id_vars=['index'],
                                                              value_vars=df_transposed.columns,
                                                              var_name='Kernel',
                                                              value_name='Time (ms)')
bar_data.rename(columns={'index': 'Size'}, inplace=True)

# get colormap
colors = sns.color_palette("husl", 8)  # 8 kernels
custom_cmap = ListedColormap(colors)

# create figure and ax
fig, ax1 = plt.subplots(1, 1, figsize=(16, 12))

# set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12

# draw line plot for each kernel
for i, kernel in enumerate(df.index):
    kernel_data = line_data[line_data['Kernel'] == kernel]
    ax1.plot(kernel_data['Size'], kernel_data['Time (ms)'], 
             label=f'Kernel {kernel}', linewidth=2.5, color=colors[i], marker='o', markersize=4)

ax1.set_xscale('log')
# ax1.set_yscale('log')
ax1.set_xlabel('Input Size', fontsize=14, fontweight='bold')
ax1.set_ylabel('Time (ms)', fontsize=14, fontweight='bold')
ax1.set_title('Reduce Kernel Performance: Time vs Input Size', fontsize=16, fontweight='bold', pad=20)
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fancybox=True, shadow=True)
ax1.grid(True, alpha=0.3)

# tight layout
plt.tight_layout()

# save figure
plt.savefig('benchmark.png', dpi=300, bbox_inches='tight')