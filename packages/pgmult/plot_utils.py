import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.stats import multivariate_normal


def data_plot(X):
    M, K = np.shape(X)

    fig, ax = plt.subplots()
    ind = np.arange(K)  # the x locations for the groups
    width = 0.7 / M  # the width of the bars
    for m in range(M):
        ax.bar(ind - width / M + (m) * width, X[m, :], width,
               color='C' + str(m), label=str(m))

    ax.set_title('Counts')
    ax.set_xlabel('k')
    ax.set_ylabel('N_k')

    # Add some text for labels, title and custom x-axis tick labels, etc.

    # ax.set_xticks(ind)
    # ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5'))
    ax.legend()

    plt.show()

def data_GT_plot(X,Pi_GT):
    """
    Plots a bar plot of the realtive frequency and Ground Truth

    Parameters
    ----------
    X: [MxK] Count matrix
    Pi_GT: [MxK] relative GT frequency

    Returns
    -------

    """
    M, K = np.shape(X)
    X=X/np.tile(np.sum(X,axis=1).reshape(-1,1),(1,K))
    fig, ax = plt.subplots()
    ind = np.arange(K)  # the x locations for the groups
    width = 0.7 / M  # the width of the bars
    for m in range(M):
        ax.bar(ind - width / M + (m) * width, X[m, :], width,
               color='C' + str(m), label=str(m))
        ax.plot(ind - width / M + (m) * width, Pi_GT[m, :], marker="D", linestyle="", alpha=0.8, color="r")

    ax.set_title('Frequency')
    ax.set_xlabel('k')
    ax.set_ylabel('Pi_k')

    # Add some text for labels, title and custom x-axis tick labels, etc.

    # ax.set_xticks(ind)
    # ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5'))
    ax.legend()

    plt.show()

def bar_plot(Pi_samples, Pi_GT=False):
    """
    Plots quantiles and posterior mean for Pi_sameples. Ground truth plotted with diamonds.
    Parameters
    ----------
    Pi_samples [NxMxK] N samples for the K categories and M barplots


    """
    N, M, K = np.shape(Pi_samples)
    Pi_post_mean = np.mean(Pi_samples, axis=0)

    Pi_post_quantile_low = Pi_post_mean - np.quantile(Pi_samples, 0.05, axis=0)
    Pi_post_quantile_up = np.quantile(Pi_samples, 0.95, axis=0) - Pi_post_mean

    fig, ax = plt.subplots()
    ind = np.arange(K)  # the x locations for the groups
    width = 0.7 / M  # the width of the bars
    for m in range(M):
        ax.bar(ind - width / M + (m) * width, Pi_post_mean[m, :], width,
               yerr=np.array([Pi_post_quantile_low[m, :], Pi_post_quantile_up[m, :]]),
               color='C' + str(m), label=str(m))
        if not isinstance(Pi_GT, bool):
            ax.plot(ind - width / M + (m) * width, Pi_GT[m, :], marker="D", linestyle="", alpha=0.8, color="r")

    ax.set_title('Posterior')
    ax.set_xlabel('k')
    ax.set_ylabel('Pi')

    # Add some text for labels, title and custom x-axis tick labels, etc.

    # ax.set_xticks(ind)
    # ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5'))
    ax.legend()

    plt.show()


def mcmc_plot(Pi_samples):
    """
    Plots iteration series

    Parameters
    ----------
    Pi_samples [NxMxK]  N samples for the K categories and M barplots
    """
    N, M, K = np.shape(Pi_samples)
    for m in range(M):
        plt.figure(m)
        for k in range(K):
            plt.plot(Pi_samples[:, m, k])
        plt.show()


def gaussian_plot(Psi_samples, GT=False, ):
    """
    Plots Scatter plots for the Gaussian distributed samples. Ground truth as contour plot.
    Parameters
    ----------
    Psi_samples: [NxMxK-1]  N samples for the K categories and M barplots
    GT: Ground truth tupple mu_0_k_GT, Sigma_0_GT, Psi_GT

    Returns
    -------

    """
    N, M, K_prime = np.shape(Psi_samples)

    if M == 2:
        style.use('fivethirtyeight')
        for k in range(K_prime):
            if not isinstance(GT, bool):
                mu_0_k_GT, Sigma_0_GT, Psi_GT = GT
                x = np.linspace(-5, 5, 500)
                y = np.linspace(-5, 5, 500)
                X, Y = np.meshgrid(x, y)

                pos = np.array([X.flatten(), Y.flatten()]).T

                rv = multivariate_normal(mu_0_k_GT[:, k], Sigma_0_GT)

                fig = plt.figure(figsize=(10, 10))
                ax0 = fig.add_subplot(111)
                ax0.contour(X, Y, rv.pdf(pos).reshape(500, 500))

            ax0 = fig.add_subplot(111)
            ax0.scatter(Psi_samples[:, 0, k], Psi_samples[:, 1, k])
            if not isinstance(GT, bool):
                ax0.scatter(Psi_GT[0, k], Psi_GT[1, k], color="red")

            plt.show()
