import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

dir = Path(__file__).resolve().parent
data = pd.read_csv(dir / "n_e_data_Te_15000.csv")

no_bins = 4
labels = [f"bin {i + 1}" for i in range(no_bins)]
data["z_bin"] = pd.cut(data["redshift"], bins = no_bins, labels = labels)

for label, subset in data.groupby("z_bin", observed=True):
    print(label, len(subset))

rng = np.random.default_rng(42)

def bootstrap_median(group, column, n_boot=10_000):
    values = group[column].to_numpy(dtype=float)

    if column == "redshift":
        err_low = err_high = group["redshift_err"].to_numpy(dtype=float)
    elif column == "n_e":
        err_low = group["n_e_err_lower"].to_numpy(dtype=float)
        err_high = group["n_e_err_upper"].to_numpy(dtype=float)
    else:
        raise ValueError(f"No uncertainty columns defined for {column}")

    # Resample objects, then perturb each selected measurement by its error.
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    draws = rng.standard_normal(size=indices.shape)
    errors = np.where(draws < 0, err_low[indices], err_high[indices])
    perturbed = values[indices] + draws * errors

    boot_medians = np.median(perturbed, axis=1)
    low, high = np.percentile(boot_medians, [16, 84])
    return np.median(values), low, high

results = []

for z_bin, group in data.groupby("z_bin", observed=True):
    row = {"z_bin": z_bin, "count": len(group)}

    for column, name in [("redshift", "z"), ("n_e", "n_e")]:
        median, low, high = bootstrap_median(group, column)
        row[f"{name}_median"] = median
        row[f"{name}_ci16"] = low
        row[f"{name}_ci84"] = high

    results.append(row)

summary = pd.DataFrame(results).set_index("z_bin")
print(summary)

#Extract arrays from DataFrame
z_median = summary["z_median"].to_numpy()
z_error_lower = summary["z_ci16"].to_numpy()
z_error_upper = summary["z_ci84"].to_numpy()

n_e_median = summary["n_e_median"].to_numpy()
n_e_error_lower = summary["n_e_ci16"].to_numpy()
n_e_error_upper = summary["n_e_ci84"].to_numpy()

#Fit a power laws to the medians
def power_law(z, a, p):
    n_e = a * (1 + z) ** p
    return n_e



colors = plt.cm.viridis(np.linspace(0, 1, no_bins))
color_by_bin = dict(zip(labels, colors))

for z_bin, group in data.groupby("z_bin", observed=True):
    color = color_by_bin[str(z_bin)]
    plt.scatter(group["redshift"], group["n_e"], color=color, alpha=0.35, s=20)
    plt.errorbar(
        group["redshift"],
        group["n_e"],
        xerr=group["redshift_err"],
        yerr=[group["n_e_err_lower"], group["n_e_err_upper"]],
        fmt="none",
        ecolor=color,
        alpha=0.35,
    )

for z_bin, row in summary.iterrows():
    plt.scatter(
        row["z_median"],
        row["n_e_median"],
        color=color_by_bin[str(z_bin)],
        s=50,
        marker="D",
    )
    plt.errorbar(
        row["z_median"],
        row["n_e_median"],
        xerr=[[row["z_median"] - row["z_ci16"]], [row["z_ci84"] - row["z_median"]]],
        yerr=[[row["n_e_median"] - row["n_e_ci16"]], [row["n_e_ci84"] - row["n_e_median"]]],
        fmt="none",
        ecolor=color_by_bin[str(z_bin)],
        capsize=5,
    )

plt.xlabel(r"$z$")
plt.ylabel(r"$n_e$ (cm$^{-3}$)")
plt.yscale("log")
plt.title(rf"$T_e$ = 15000 K (Topping/PyNeb limits, Topping SNR>5)")
plt.savefig(dir / "n_e_vs_z_Te_15000_binned.png", dpi=300)
plt.show()


