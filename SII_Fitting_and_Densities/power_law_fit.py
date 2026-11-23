from scipy.optimize import least_squares
import emcee
import corner
import numpy as np

"""For this, I am following the MCMC tutorial written by Imad Pasha at 
https://prappleizer.github.io/Tutorials/MCMC/MCMC_Tutorial.html"""



def model(theta, z):
    """Power law model for the density redshift relation"""
    a, p = theta
    model = a * (1 + z) ** p
    return model

def log_likelihood(theta, n_e_data, z, n_e_data_err_upper, n_e_data_err_lower):
    """Compute the log-likelihood for the data given the model, taking asymmetric 
    errors into account"""
    n_e_model = model(theta, z)

    residual = n_e_data - n_e_model
    n_e_error = np.where(residual > 0, n_e_data_err_lower, n_e_data_err_upper)

    ratio = (n_e_data - n_e_model)/n_e_error

    log_likelihood = -0.5 * np.sum(ratio ** 2)

    return log_likelihood

def log_prior(theta):
    """Constrain priors"""
    a, p = theta
    if a < 1e4 and p > -5 and p < 5:
        return 0
    else:
        return -np.inf

def log_probability(theta, n_e_data, z, n_e_data_err_upper, n_e_data_err_lower):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, n_e_data, z, n_e_data_err_upper, n_e_data_err_lower)

def main(data):

    initial_guess = np.array([10, 1.5]) #a, p
    proposal = initial_guess + 1e-4 * np.random.randn(2)

    sampler = emcee.EnsembleSampler(50, 2, log_probability, args = (data))

    print("Running burn-in...")
    proposal, _, _ = sampler.run_mcmc(proposal, 100)
    sampler.reset()

    print("Running production...")
    pos, prob, state = sampler.run_mcmc(proposal, 5000)

    return sampler, pos, prob, state

