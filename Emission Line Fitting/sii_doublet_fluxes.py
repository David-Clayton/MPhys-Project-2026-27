import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, join
from astropy.modeling import models,fitting
from scipy.optimize import least_squares
import re
from pathlib import Path

class SIILineFitting:

    """
    This class will calculate the flux of both lines of the [SII]6716,31 doublet along with errors. 
    
    It will read in the redshift and raw spectroscopic data from a target combined with a NIRSpec 
    disperser (G140M, G235M, or G395M), and fit the data to a model with a Levenberg-Marquardt least-
    squared fitting algorithm. The model in question is a pair of Gaussians to model the doublet and
    a linear polynomial to model the continuum around the doublet.
    """

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
        self.flux = 1e20 * flux_data[:,1] #Scale it up to improve fitter's numerical precision
        self.flux_err = 1e20 * flux_data[:,2]

    def sii_mask(self):
        #Cut out the area of the spectrum surrounding the [SII] doublet
        upper_lim = 6850*(1+self.redshift)
        lower_lim = 6600*(1+self.redshift)
        mask = (self.wavelengths < upper_lim) & (self.wavelengths > lower_lim)
        return mask

    def spectrum_model(self, amp6716, amp6731, sigma, redshift):
        """Create a theoretical model of the spectrum around the [SII] doublet as
        a pair of Gaussians for the doublet and a linear function to model the 
        continuum.""" 
        continuum = models.Polynomial1D(1)
        sii6716 = models.Gaussian1D(amplitude = amp6716, mean = 6716 * (1 + redshift), stddev = sigma)
        sii6731 = models.Gaussian1D(amplitude = amp6731, mean = 6731 * (1 + redshift), stddev = sigma) 
        #Fix peaks in place in order to fix redshifts
        #sii6716.mean.fixed = True
        #sii6731.mean.fixed = True
        model = continuum + sii6716 + sii6731
        return model

    def model_fit(self):
        """Use Astropy Lev-Mar Least Squares fitter to fit the model to the data"""
        mask = self.sii_mask()
        fitter = fitting.LevMarLSQFitter(calc_uncertainties=True)
        #Initial guess of fit parameters
        initial_guess = self.spectrum_model(1, 1, 20, self.redshift)
        #Fit model to masked data
        fit_to_model = fitter(initial_guess, self.wavelengths[mask], self.flux[mask], maxiter=1000)
        #print(fit_to_model.parameters)
        """
        fit_to_model.parameters is as follows:
        index 0: continuum.c0  c0 = continuum intercept
        index 1: continuum.c1  c1 = continuum slope
        index 2: sii6716.amplitude
        index 3: sii6716.mean
        index 4: sii6716.stddev
        index 5: sii6731.amplitude
        index 6: sii6731.mean
        index 7: sii6731.stddev
        """
        #print((fit_to_model.parameters[3] / 6716) - 1)
        #print((fit_to_model.parameters[6] / 6731) - 1) #Fit's best estimate of redshifts
        #print(fitter.fit_info["param_cov"])
        return fit_to_model, fitter

    def plot_fit(self):
        """Plot empirical data with resultant fitted data from Lev-Mar method"""
        mask = self.sii_mask()
        #Plot empirical data
        plt.plot(self.wavelengths[mask], self.flux[mask], color = "limegreen")
        #plt.errorbar(self.wavelengths[mask], self.flux[mask], yerr = self.flux_err[mask], color = "limegreen")
        plt.fill_between(self.wavelengths[mask], self.flux[mask] - self.flux_err[mask], 
                        self.flux[mask] + self.flux_err[mask], alpha=0.5, color = "limegreen") 
        
        #Plot fitted data
        fitted_model = self.model_fit()[0]
        #Expand wavelength data for less janky curve
        x_data = np.linspace(np.min(self.wavelengths[mask]), np.max(self.wavelengths[mask]), 1000)   
        fitted_data = fitted_model(x_data)

        plt.plot(x_data, fitted_data, color = "mediumpurple")
        plt.axvline(6716 * (1+self.redshift), color = "darkblue")
        plt.axvline(6731 * (1+self.redshift), color = "darkblue")
        plt.xlabel(f"Lab-frame wavelength (Angstrom)")
        plt.ylabel(r"Flux ($10^{-20}$erg/s/$cm^{-2}$/Angstrom)")
        plt.title(f"Target {self.target_id} \n with disperser {self.disperser}. z = {self.redshift}")
        plt.savefig(f"Testfittedspectrum.png")
        plt.show()

    def calculate_fluxes(self):
        """Calculate the fluxes and uncertainties on the two [SII] emission lines"""
        #Extract the variances of the parameters from the fit's covariance matrix
        fit_parameters, fitter = self.model_fit()
        stddevs = np.sqrt(np.diag(fitter.fit_info['param_cov']))
        c0, c1, amp6716, mean6716, stddev6716, amp6731, mean6731, stddev6731 = fit_parameters.parameters
        c0_err, c1_err, amp6716_err, mean6716_err, stddev6716_err, amp6731_err, mean6731_err, stddev6731_err = stddevs
        #[SII]6716
        flux6716 = amp6716 * stddev6716 * np.sqrt(2 * np.pi)
        #Analytic error propagation of flux expression
        flux6716_err = flux6716 * np.sqrt((amp6716_err/amp6716)**2 + (stddev6716_err/stddev6716)**2) 

        #[SII]6731
        flux6731 = amp6731 * stddev6731 * np.sqrt(2 * np.pi)
        flux6731_err = flux6731 * np.sqrt((amp6731_err/amp6731)**2 + (stddev6731_err/stddev6731)**2) 

        #print(f"flux for [SII]6716 = {flux6716} +- {flux6716_err}")
        #print(f"flux for [SII]6731 = {flux6731} +- {flux6731_err}")

l = SIILineFitting(r"C:\Users\drcla\OneDrive\MPhys Project\Emission Line Fitting\spec1d_fluxcal\spec1d_40081_g395m_final.txt")
l.plot_fit()
l.calculate_fluxes()


