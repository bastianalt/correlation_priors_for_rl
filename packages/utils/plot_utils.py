import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.stats import multivariate_normal


def bar_plot(Pi_samples, Pi_GT=False):
    """

    Parameters
    ----------
    Pi_samples [NxMxK] N samples for the K categories and M barplots


    """
    N, M, K = np.shape(Pi_samples)
    Pi_post_mean = np.mean(Pi_samples, axis=0)

    Pi_post_quantile_low = Pi_post_mean - np.quantile(Pi_samples, 0.1, axis=0)
    Pi_post_quantile_up = np.quantile(Pi_samples, 0.90, axis=0) - Pi_post_mean

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

    Parameters
    ----------
    Pi_samples
    Pi_GT

    Returns
    -------

    """
    N, M, K = np.shape(Pi_samples)
    for m in range(M):
        plt.figure(m)
        for k in range(K):
            plt.plot(Pi_samples[:, m, k])
        plt.show()


def gaussian_plot(Psi_samples, GT=False, ):
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
