import emcee
import corner
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd 
from IPython.display import display, Math

"""For this, I am following the MCMC tutorial written by Imad Pasha at 
https://prappleizer.github.io/Tutorials/MCMC/MCMC_Tutorial.html"""

#Import data
plt.rcParams["text.usetex"] = True

dir = Path(__file__).resolve().parent
dataframe = pd.read_csv(dir / "n_e_vs_z_T_e_15000_binned_data.csv", comment = "#")

lit_source = dataframe["Literature"]
z = dataframe["z_median"].to_numpy(dtype=float)
n_e_data = dataframe["n_e_median"].to_numpy(dtype=float)
n_e_data_err_lower = (dataframe["n_e_median"] - dataframe["n_e_ci16"]).to_numpy(dtype=float)
n_e_data_err_upper = (dataframe["n_e_ci84"] - dataframe["n_e_median"]).to_numpy(dtype=float)

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

    nwalkers = 50
    ndim = 2

    initial_guess = np.array([10, 1.5]) #a, p
    proposal = initial_guess + 1e-4 * np.random.randn(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args = (data))

    print("Running burn-in...")
    proposal, _, _ = sampler.run_mcmc(proposal, 100)
    sampler.reset()

    print("Running production...")
    pos, prob, state = sampler.run_mcmc(proposal, 5000, progress = True)

    fig, axes = plt.subplots(2, figsize=(8, 6), sharex=True)
    samples = sampler.get_chain()
    labels = ["a", "p"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    
    axes[-1].set_xlabel("step number")
    #plt.savefig(dir / "MCMCwalkers.png")
    #plt.show()

    tau = sampler.get_autocorr_time()
    print(tau)

    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    print(flat_samples.shape)

    fig = corner.corner(flat_samples, labels=labels, 
                        plot_datapoints=True,quantiles=[0.16, 0.5, 0.84])
    plt.savefig(dir / "MCMCcornerplot.png")
    #plt.show()

    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = r"\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])
        print(f"{labels[i]} = {mcmc[1]:.3f} " 
              f"-{q[0]:.3f}/+{q[1]:.3f}")

    return sampler, pos, prob, state

data = (n_e_data, z, n_e_data_err_upper, n_e_data_err_lower)
sampler, _, _, _ = main(data)

#Plot all data
dataframe["Literature"] = dataframe["Literature"].str.strip()

own = dataframe[dataframe["Literature"] == "This work"]
literature = dataframe[dataframe["Literature"] != "This work"]

fig, ax = plt.subplots()

# This work: viridis colour by redshift bin
bins = own["z_bin"].dropna().unique()
bin_colours = dict(zip(bins, plt.cm.viridis(np.linspace(0, 1, len(bins)))))

for z_bin, group in own.groupby("z_bin", sort=False):
    ax.errorbar(group["z_median"], group["n_e_median"],
        xerr=[group["z_median"] - group["z_ci16"],
            group["z_ci84"] - group["z_median"]],
        yerr=[group["n_e_median"] - group["n_e_ci16"],
            group["n_e_ci84"] - group["n_e_median"]],
        fmt="o", color=bin_colours[z_bin], ecolor = "black",
        markeredgecolor="black", markersize = 10, capsize=3, zorder = 2)

# Literature: distinct colour and marker by source
sources = literature["Literature"].dropna().unique()
markers = ["s", "^", "D", "v", "P", "X", "*", "<", ">"]

for source, marker in zip(sources, markers):
    group = literature[literature["Literature"] == source]
    ax.errorbar(group["z_median"], group["n_e_median"],
                
        xerr=[group["z_median"] - group["z_ci16"], 
              group["z_ci84"] - group["z_median"]],

        yerr=[group["n_e_median"] - group["n_e_ci16"],
            group["n_e_ci84"] - group["n_e_median"]],

        fmt=marker, color = "darkorange", ecolor = "black",
        markeredgecolor="black", markersize = 10, capsize=3, label=source, zorder = 0)

flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
a_fit, p_fit = np.median(flat_samples, axis=0)

z_fit = np.linspace(dataframe["z_median"].min(),
                    dataframe["z_median"].max(), 1000)
ax.plot(
    z_fit,
    model((a_fit, p_fit), z_fit),
    color="black",
    linewidth=2, linestyle = "--", zorder = 1)

ax.set_xlabel(r"$z$")
ax.set_ylabel(r"$n_e$ (cm$^{-3}$)")
ax.set_yscale("log")
ax.legend()
plt.savefig(dir / "Master_density_plot.png")
plt.show()
