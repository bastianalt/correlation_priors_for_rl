import pickle
import numpy as np
import matplotlib.pyplot as plt


def visualize(file):
    with open(file, 'rb') as f:
        setup, results = pickle.load(f)
        results = np.asarray(results)
        sweep = setup[-1]

    # ---------- Visualization ---------- #
    mean = results.mean(axis=0)
    std = results.std(axis=0)
    cmap = plt.get_cmap('tab10')
    for env, envName in enumerate(sweep['envs']):
        for div, divName in enumerate(sweep['divergences']):
            fig, ax = plt.subplots()
            for m, mName in enumerate(sweep['methods']):
                mu = mean[env, m, div, :]
                s = std[env, m, div, :]
                ax.semilogx(sweep['nTrajs'], mu, label=mName, color=cmap.colors[m], marker='o')
                plt.fill_between(sweep['nTrajs'], mu - s, mu + s, alpha=0.2, color=cmap.colors[m])
                plt.title(envName)
                plt.legend()
                plt.xlabel('# trajectories')
                plt.ylabel(divName)
                plt.grid(True)
            plt.savefig('../evaluation/plots/imitationlearning/' + envName + '_' + divName + '.pdf', bbox_inches='tight')
            plt.show()


if __name__ == '__main__':
    file = '../results/imitationlearning/result.pkl'
    visualize(file)
