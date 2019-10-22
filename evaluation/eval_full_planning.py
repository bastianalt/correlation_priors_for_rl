import numpy as np
import xyzpy as xyz
import matplotlib.pyplot as plt
import seaborn as sns
from simulations.sim_full_planning import generateEnvironment

# style settings
plt.style.use('seaborn')
plt.style.use('nips.mplstyle')
style = dict(pg_mean=['xkcd:moss green', r'PG mean'],
             pg_sampling=['xkcd:moss green', r'PG sampling'],
             dir_sparse_mean=['xkcd:gold', r'Dir ($\alpha=10^{-3}$) mean'],
             dir_sparse_sampling=['xkcd:gold', r'Dir ($\alpha=10^{-3}$) sampling'],
             dir_uniform_mean=['xkcd:rust red', r'Dir ($\alpha=1$) mean'],
             dir_uniform_sampling=['xkcd:rust red', r'Dir ($\alpha=1$) sampling'])

# load data
file = '../results/fullplanning/result.h5'
data_full = xyz.load_ds(file)

# extract relevant block and normalize values
env = data_full.attrs['env']
mdp = generateEnvironment(env)
max_val = mdp.policyIteration()[0].mean()
data = data_full['value'].sel(planningUpdateFreq=10) / max_val

# create figure
plt.figure(figsize=(5, 2))
handles = []
for policy in data.coords['policies'].values:
    # skip sampling policies
    # if 'sampling' in policy:
    #     continue

    # x-axis
    x = data.coords['time'].values

    # compute mean and std
    mean = data.sel(policies=policy).mean('seed')
    std = data.sel(policies=policy).std('seed')

    # select line style
    if 'mean' not in policy:
        params = dict(linestyle='', marker='o', markersize=2, markevery=0.02)
    else:
        params = dict()
    color, name = style[policy]

    # add plot
    h, = plt.plot(x, mean, c=color, label=name, **params)
    handles.append(h)
    if 'sampling' not in policy:
        plt.fill_between(x, mean-std, mean+std, alpha=0.3, facecolor=color, edgecolor=None)

# set labels
plt.xlabel('number of transitions')
plt.ylabel('normalized value')

# save and show
plt.savefig('../evaluation/plots/fullplanning/fullplanning.pdf', bbox_inches='tight')
plt.show()

# export legend
plt.figure()
plt.figlegend(handles=handles, ncol=3)
plt.savefig('../evaluation/plots/fullplanning/legend.pdf', bbox_inches='tight')



