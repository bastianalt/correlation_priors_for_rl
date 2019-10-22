import xyzpy as xyz
import matplotlib.pyplot as plt
import seaborn as sns

# style settings
plt.style.use('seaborn')
plt.style.use('nips.mplstyle')
style = dict(pg=['xkcd:moss green', r'PG mean'],
             dir_sparse=['xkcd:gold', r'Dir ($\alpha=10^{-3}$) mean'],
             dir_uniform=['xkcd:rust red', r'Dir ($\alpha=1$) mean'])

# load data
file = '../results/modellearning/result.h5'
data_full = xyz.load_ds(file)

# extract relevant block and average over MC runs
data = data_full.sel(metrics='hellinger', env='10x10_TunnelWorldCorner').mean('seed')

# create figure
handles = []
for method in data.coords['methods'].values:
    # x-axis
    x = data.coords['nData'].values

    # extract mean and std
    std = data['std'].sel(methods=method)
    mean = data['mean'].sel(methods=method)

    # add plot
    color, name = style[method]
    plt.plot(x, mean, marker='o', color=color)
    plt.fill_between(x, mean-std, mean+std, alpha=0.2, edgecolor=None, facecolor=color)

# set labels
plt.xlabel('number of transitions')
plt.ylabel('Hellinger distance')

# save and show
plt.savefig('../evaluation/plots/transition_model/transition_model.pdf', bbox_inches='tight')
plt.show()
