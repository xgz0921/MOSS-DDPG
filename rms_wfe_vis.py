import numpy as np
import matplotlib.pyplot as plt
#Visualize the mean and standard deviation of the test results of models in RMS wavefront error.
file_path = 'Path_to_training_history_directory'# Example: 'Training History/2024_07_04/12-09-27_12_m_0.15_bound_40000_stp_128_unts_0.00025_lrA_0.0025_LrC_8192_bs_0.06_noise_0.001_nsmin/'

wfe_decreasing = np.load(file_path+'WFE_record_decreasing_ra.npy')
wfe_decreasing = np.sqrt(np.sum(wfe_decreasing**2, axis=1))

wfe_cf_decreasing = np.load(file_path+'WFE_record_cf_decreasing_ra.npy')
wfe_cf_decreasing = np.sqrt(np.sum(wfe_cf_decreasing**2, axis=1))

wfe_training = np.load(file_path+'WFE_record_uniform_ra.npy')
wfe_training = np.sqrt(np.sum(wfe_training**2, axis=1))

wfe_cf_training = np.load(file_path+'WFE_record_cf_uniform_ra.npy')
wfe_cf_training = np.sqrt(np.sum(wfe_cf_training**2, axis=1))

wfe_normal = np.load(file_path+'WFE_record_normal_ra.npy')
wfe_normal = np.sqrt(np.sum(wfe_normal**2, axis=1))

wfe_cf_normal = np.load(file_path+'WFE_record_cf_normal_ra.npy')
wfe_cf_normal = np.sqrt(np.sum(wfe_cf_normal**2, axis=1))

# Data for the plots with the new sequence (A at the top)
data_without_cf = [wfe_training, wfe_normal, wfe_decreasing]
labels = ['A', 'B', 'C']
data_with_cf = [wfe_cf_training, wfe_cf_normal, wfe_cf_decreasing]

# Calculate means and standard deviations
means_without_cf = [np.mean(data) for data in data_without_cf]
std_without_cf = [np.std(data) for data in data_without_cf]

means_with_cf = [np.mean(data) for data in data_with_cf]
std_with_cf = [np.std(data) for data in data_with_cf]

# Reverse the order to have A at the top
labels = labels[::-1]
means_without_cf = means_without_cf[::-1]
std_without_cf = std_without_cf[::-1]
means_with_cf = means_with_cf[::-1]
std_with_cf = std_with_cf[::-1]

# Create horizontal bar plot
y = np.arange(len(labels))  # Label locations
height = 0.4  # Height of the bars

fig, ax = plt.subplots(figsize=(9, 4))

# Plot data without CF
bars1 = ax.barh(y - height/2, means_without_cf, height, xerr=std_without_cf, label='MOSS-DDPG', capsize=4)

# Plot data with CF
bars2 = ax.barh(y + height/2, means_with_cf, height, xerr=std_with_cf, label='Parabolic Maximization', capsize=4)

# Add labels, title, and custom y-axis tick labels
ax.set_xlabel('Average $RMS_{WFE}$',fontsize = 14)
ax.set_yticks(y)
ax.tick_params(axis='x', labelsize=14)
ax.set_yticklabels(labels,fontsize =14)
ax.axvline(x=0.0349, color='r', linestyle='--',label = 'Diffraction-limited')
ax.legend(fontsize = 14)

plt.tight_layout()
# Display the plot
plt.savefig(file_path+'avg_rms_wfe_random_test.pdf',dpi = 200)

