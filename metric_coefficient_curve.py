from DSAO_env import *
# This file is to plot an image metric-coefficient curve for each individual mode.
env = MOSSDDPG_Env()
env.reset()
env.cr.c = np.zeros(n_modes_all)
co_test = np.linspace(-1,1,41)
metrics = []
env.ab.c = np.zeros(n_modes_all)
env.abSet()
flat = env.crSet()
for m in range(len(abDM)):
    metrics_m = []
    R_effs_m = []
    for i in range(co_test.shape[0]):
        env.ab.c = np.zeros(n_modes_all)
        env.ab.c[m] = co_test[i].copy()
        env.abSet()
        metric = env.crSet()
        metrics_m.append(metric/flat)
    metrics.append(metrics_m)
metrics = np.array(metrics)


#%%
plt.figure(figsize=(15, 10))
co_test = np.linspace(-1,1,41)

colors = [
    '#e6194b',  # Red
    '#3cb44b',  # Green
    '#ffe119',  # Yellow
    '#4363d8',  # Blue
    '#f58231',  # Orange
    '#911eb4',  # Purple
    '#46f0f0',  # Cyan
    '#f032e6',  # Magenta
    '#bcf60c',  # Lime
    '#fabebe',  # Pink
    '#008080',  # Teal
    '#e6beff',  # Lavender
    '#800000',  # Maroon
    '#808000',  # Olive
    '#800080',  # Purple
    '#000080',  # Navy
    '#008080',  # Teal
    '#008000'   # Green
]

assert len(colors) >= 12, "Not enough colors defined for all modes."

for mode in range(12):
    plt.plot(co_test, metrics[mode, :], label=f'Mode {mode + 4}', color=colors[mode])
    plt.yscale('log')
from matplotlib.ticker import LogLocator
#plt.ylim(1e-3,1)
log_locator = LogLocator(subs='all')  # 'all' to place ticks at every integer power of the base and all of its integer multiples
plt.gca().yaxis.set_major_locator(log_locator)

# Optionally, adjust the appearance of ticks and labels
plt.minorticks_on()  # Turn on minor ticks, which are off by default for log scale
plt.grid(True, which="both", ls="--", linewidth=0.5)  # Optional: add a grid for better readability

plt.xlabel('Zernike Coefficient / $\mu$m',fontsize = 12)
plt.ylabel('Relative Image Metric Value',fontsize = 12)
plt.legend()
plt.tight_layout()
plt.show()