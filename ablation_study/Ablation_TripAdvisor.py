# import matplotlib.pyplot as plt
# import numpy as np

# # Data for MAE
# conditions = ['Lack of global attention', 'Lack of Global-CL', 'D-MGAC']
# MAE = np.array([0.7714, 0.7922, 0.7274])

# # Data for RMSE
# RMSE = np.array([1.0375, 1.0824, 0.9858])

# # Colors for each condition
# colors = ['b', 'g', 'r']

# # Bar width
# bar_width = 0.3  # Adjust this value to make bars thinner or thicker

# # Font sizes
# label_fontsize = 18
# title_fontsize = 16
# tick_fontsize = 14
# condition_fontsize = 12  # Font size for conditions (x-axis labels)

# # Plot MAE
# plt.figure(figsize=(10, 6))
# bars = plt.bar(conditions, MAE, color=colors, width=bar_width)
# plt.xlabel('', fontsize=label_fontsize)
# plt.ylabel('MAE', fontsize=label_fontsize)
# plt.title('TripAdvisor dataset', fontsize=title_fontsize)
# plt.ylim(0.7, 0.85)  # Set the y-axis range for MAE
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.xticks(fontsize=condition_fontsize, rotation=45)  # Increase font size and rotate x-axis labels if needed
# plt.yticks(fontsize=tick_fontsize)
# plt.savefig('MAE_chart.png')
# plt.show()

# # Plot RMSE
# plt.figure(figsize=(10, 6))
# bars = plt.bar(conditions, RMSE, color=colors, width=bar_width)
# plt.xlabel('', fontsize=label_fontsize)
# plt.ylabel('RMSE', fontsize=label_fontsize)
# plt.title('TripAdvisor dataset', fontsize=title_fontsize)
# plt.ylim(0.9, 1.2)  # Set the y-axis range for RMSE
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.xticks(fontsize=condition_fontsize, rotation=45)  # Increase font size and rotate x-axis labels if needed
# plt.yticks(fontsize=tick_fontsize)
# plt.savefig('RMSE_chart.png')
# plt.show()






import matplotlib.pyplot as plt
import numpy as np

# Data for MAE
conditions = ['Lack of global attention', 'Lack of Global-CL', 'D-MGAC']
MAE = np.array([0.7714, 0.7922, 0.7274])

# Data for RMSE
RMSE = np.array([1.0375, 1.0824, 0.9858])

# Colors for each condition
colors = ['b', 'g', 'r']

# Bar width
bar_width = 0.3  # Adjust this value to make bars thinner or thicker

# Font sizes
label_fontsize = 20
title_fontsize = 18
tick_fontsize = 14

# Plot MAE
plt.figure(figsize=(10, 6))
bars = plt.bar(conditions, MAE, color=colors, width=bar_width)
plt.xlabel('', fontsize=label_fontsize)
plt.ylabel('MAE', fontsize=label_fontsize)
plt.title('TripAdvisor dataset', fontsize=title_fontsize)
plt.ylim(0.7, 0.85)  # Set the y-axis range for MAE
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.savefig('MAE_chart.png')
plt.show()

# Plot RMSE
plt.figure(figsize=(10, 6))
bars = plt.bar(conditions, RMSE, color=colors, width=bar_width)
plt.xlabel('', fontsize=label_fontsize)
plt.ylabel('RMSE', fontsize=label_fontsize)
plt.title('TripAdvisor dataset', fontsize=title_fontsize)
plt.ylim(0.9, 1.2)  # Set the y-axis range for RMSE
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.savefig('RMSE_chart.png')
plt.show()
