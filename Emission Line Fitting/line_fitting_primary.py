import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, join
from astropy.modeling import models,fitting
from scipy.optimize import least_squares
import re
from pathlib import Path

class LineFitting:

    """This class will read in a redshift of a target and the target flux from a certain disperser
    (either G140M, G235M, G395M) and will output the fitted spectrum for the [SII]6716,31 doublet
    along with a continuum fitted to a linear function of the form a*wavelength + b."""

    def __init__(self, file):

        filename = Path(file).name
        match = re.search(r"spec1d_(\d+)_(g\d+m)_", filename)

        if match is None:
            raise ValueError(f"Can't extract target ID and disperser from {filename}")
        
        self.target_id = int(match.group(1))
        self.disperser = match.group(2)
        catalogue = Table.read(r"C:\Users\drcla\OneDrive\MPhys Project\Emission Line Fitting\excels_cat_internal_v2_0p5asec_120424_miri.fits")
        self.redshift = catalogue[catalogue["targetid"] == self.target_id]["z_excels_v1"][0]

        flux_data = np.loadtxt(file)
        self.wavelengths = flux_data[:,0]
        self.flux = flux_data[:,1]
        self.flux_err = flux_data[:,2]

    def sii_mask(self):
        #Cut out the area of the spectrum surrounding the [SII] doublet
        upper_lim = 6850*(1+self.redshift)
        lower_lim = 6600*(1+self.redshift)
        mask = (self.wavelengths < upper_lim) & (self.wavelengths > lower_lim)
        return mask

    def spectrum_model(self, amp6716, amp6731, sigma, redshift):
        #Fit a pair of Gaussians to the [SII] doublet and linear expression 
        #to the continuum surrounding the doublet
        continuum = models.Polynomial1D(1)
        sii6716 = models.Gaussian1D(amplitude = amp6716, mean = 6716 * (1 + redshift), stddev = sigma)
        sii6731 = models.Gaussian1D(amplitude = amp6731, mean = 6731 * (1 + redshift), stddev = sigma) 
        model = continuum + sii6716 + sii6731
        return model

    def model_fit(self):
        mask = self.sii_mask()
        fit = fitting.LevMarLSQFitter()
        #Best guess
        initial_guess = self.spectrum_model(1, 1, 20, self.redshift)
        fitted_model = fit(initial_guess, self.wavelengths[mask], self.flux[mask], maxiter=1000)
        return fitted_model

    def plot_fit(self):
        mask = self.sii_mask()
        #Plot empirical data
        plt.plot(self.wavelengths[mask], self.flux[mask], color = "limegreen")
        plt.fill_between(self.wavelengths[mask], self.flux[mask] - self.flux_err[mask], 
                        self.flux[mask] + self.flux_err[mask], alpha=0.5, color = "limegreen") 
        
        #Plot fitted data
        fitted_model = self.model_fit()
        fitted_data = fitted_model(self.wavelengths[mask])
        plt.plot(self.wavelengths[mask], fitted_data, color = "mediumpurple")

        plt.xlabel(f"Lab-frame wavelength (Angstrom)")
        plt.ylabel(f"Flux (erg/s/cm^-2/Angstrom)")
        plt.title(f"S[II] doublet of Target {self.target_id} \n with disperser {self.disperser}")
        plt.savefig(f"Testfittedspectrum.png")
        plt.show()

        print(self.redshift)

l = LineFitting(r"C:\Users\drcla\OneDrive\MPhys Project\Emission Line Fitting\spec1d_fluxcal\spec1d_40081_g395m_final.txt")
l.plot_fit()


