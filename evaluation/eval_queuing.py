import numpy as np
import matplotlib.pyplot as plt
from packages.utils.figure_configuration_nips import figure_configuration_nips
import sys
sys.path.append('..')

def evaluate(path, file_name, file_extension):
    figure_configuration_nips()

    # ------------- Evaluate -------------------
    #Load all files
    all_data = np.load(path + file_name + file_extension)
    results = all_data['results'].tolist()
    conf_env = all_data['conf_env'].tolist()
    conf_mdp = all_data['conf_mdp'].tolist()
    conf_transition_model = all_data['conf_transition_model'].tolist()

    #Set properties
    nMC = conf_mdp['nMC']
    nEpisodes = conf_mdp['nEpisodes']
    nTransitionModels = len(conf_transition_model)

    #Container for results
    value_e = np.zeros((nMC, nTransitionModels, nEpisodes))
    for t in range(nTransitionModels):
        for n in range(nMC):
            values = results[n][0]
            # Get all experiments for envoirnment e and transition model t
            experiments = list(
                filter(lambda experiment: experiment['Tmodel_name'] == conf_transition_model[t][0], values))
            # Only evaluate results if they are present
            if experiments:
                for experiment in experiments:
                    value_e[n, t, experiment['episode']] = np.mean(experiment['value'])
            print(str(n) + str(t)) #Print MC number and transition model number


    #------------- Visualize -------------------
    cmap = plt.get_cmap('tab10')
    colors = cmap.colors[0:3] + cmap.colors[0:3]
    line_specs = ['-', '--', ':', '-', '--', ':']
    label = [r'PG', r'Dir $\alpha=1$', r'Dir $\alpha=10^{-3}$']

    # Plot only mean results
    # label=[]
    for t in range(3):
        mu = np.mean(value_e, axis=0)[t, :]
        s = np.sqrt(np.var(value_e, axis=0)[t, :])
        plt.plot(mu, line_specs[t], color=colors[t])
        plt.fill_between(range(len(mu)), mu - s, mu + s, color=colors[t], alpha=0.2)
        # label.append(conf_transition_model[t][0]) #Native labels

    plt.ylim((-10, -6))
    # plt.title('Posterior Mean Model')
    plt.legend(label, loc='upper left', bbox_to_anchor=(.335, .515))
    plt.xlabel('episode')
    plt.ylabel('value')
    plt.savefig('../evaluation/plots/queuing/queuing_learning_mean.pdf')
    plt.show()

    # Plot only sampling results
    # label=[]
    for t in range(3, 6):
        mu = np.mean(value_e, axis=0)[t, :]
        s = np.sqrt(np.var(value_e, axis=0)[t, :])
        plt.plot(mu, line_specs[t], color=colors[t])
        plt.fill_between(range(len(mu)), mu - s, mu + s, color=colors[t], alpha=0.2)
        # label.append(conf_transition_model[t][0]) #Native labels

    plt.ylim((-10, -6))
    # plt.title('Posterior Sampling Model')
    plt.legend(label)
    plt.xlabel('episode')
    plt.ylabel('value')
    plt.savefig('../evaluation/plots/queuing/queuing_learning_sampling.pdf')
    plt.show()
