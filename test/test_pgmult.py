import numpy as np
from packages.pgmult.pgmult import pgmult
from packages.utils.utils import stick_breaking, inv_stick_breaking
import matplotlib.pyplot as plt

from packages.utils.plot_utils import bar_plot, mcmc_plot, gaussian_plot
#TODO Burn in (Thinning?)
#TODO Compare with Lidermann implementation ?
#TODO Derive the full conditionals using a fully correlated Prior
#TODO Add simplex plot
K_prime = 4
M = 2

# Create GT data
Pi_vague = np.ones((M, K_prime + 1)) / (K_prime + 1)
#mu_0_k_GT = inv_stick_breaking(Pi_vague)+2
#R=np.random.randn(M,M)
#Sigma_0_GT = np.eye(M)#R.dot(R.T)
mu_0_k_GT = inv_stick_breaking(Pi_vague)
R=np.random.randn(M,M)*1e2
Sigma_0_GT = R.dot(R.T)
Sigma_0_GT=np.eye(M)
PgMultObj_GT = pgmult(mu_0_k_GT, Sigma_0_GT)
N_m = np.array(np.ones(M) * 1000)
X, Pi_GT = PgMultObj_GT.create_prior_data(N_m)
Psi_GT=inv_stick_breaking(Pi_GT)
#X = PgMultObj_GT.create_model_data(N_m, Pi_GT)

# Inference with vague prior
Pi_vague = np.ones((M, K_prime + 1)) / (K_prime + 1)
mu_0_k = inv_stick_breaking(Pi_vague) * 6e2
R_0=np.random.randn(M,M)*1e2
Sigma_0 = R_0.dot(R_0.T)
PgMultObj= pgmult(mu_0_k, Sigma_0)
N_samples=1000
samples = PgMultObj.sample_posterior(N_samples, X)
#Pi_samples = np.array([stick_breaking(samples[0][n,:,:]) for n in range(N_samples)])
Pi_samples=np.array([PgMultObj_GT.sample_prior() for _ in range(1000)])

# %% Plots
bar_plot(Pi_samples, Pi_GT=Pi_GT)
mcmc_plot(Pi_samples)
gaussian_plot(samples[0],GT=(mu_0_k_GT,Sigma_0_GT,Psi_GT))

plt.hist(Pi_samples[:,0,0],bins=100)
plt.show()




