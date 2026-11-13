import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table 
from astropy.modeling import models,fitting
import re
from pathlib import Path
import pyneb as pn
import pandas as pd

class SIILineFittingandDensities:

    """
    This class will calculate the densities of the targets from the 
    flux of both lines of the [SII]6716,31 doublet along with errors. 
    
    It will read in the redshift and raw spectroscopic data from a target combined with a NIRSpec 
    disperser (G140M, G235M, or G395M), and fit the data to a model with a Levenberg-Marquardt least-
    squared fitting algorithm. The model in question is a pair of Gaussians to model the doublet and
    a linear polynomial to model the continuum around the doublet.

    This class will use a Monte-Carlo to derive errors on the best estimates on the 
    fluxes and redshifts, with ~100-1000 realisations of each spectrum based on the EXCELS data.

    It will then use the [SII]6716,31 fluxes to calculate the electron density of the target using PyNeb's 
    S2.getTemDen() functionality.

    """

    @staticmethod
    def sii_mask(redshift, wavelengths):
            #Cut out the area of the spectrum surrounding the [SII] doublet
            upper_lim = 6850*(1+redshift)
            lower_lim = 6600*(1+redshift)
            mask = (wavelengths < upper_lim) & (wavelengths > lower_lim)
            return mask

    def __init__(self, file):

        filename = Path(file).name
        match = re.search(r"spec1d_(\d+)_(g\d+m)_", filename)

        if match is None:
            raise ValueError(f"Can't extract target ID and disperser from {filename}")
        
        self.target_id = int(match.group(1))
        self.disperser = match.group(2)

        #Temporary Linux path
        sii_fitting_dir = Path(__file__).resolve().parent
        catalogue_path = sii_fitting_dir / "excels_cat_internal_v2_0p5asec_120424_miri.fits"
        catalogue = Table.read(catalogue_path)

        #Windows path
        #catalogue = Table.read(r"C:\Users\drcla\OneDrive\MPhys Project\SII Fitting\excels_cat_internal_v2_0p5asec_120424_miri.fits")
        self.redshift = catalogue[catalogue["targetid"] == self.target_id]["z_excels_v1"][0]

        flux_data = np.loadtxt(file)
        self.wavelengths = flux_data[:,0]
        self.flux = 1e20 * flux_data[:,1] #Scale it up to improve fitter's numerical precision
        self.flux_err = 1e20 * flux_data[:,2]

        #Check for presence of doublet in masked area of spectrum
        #A chip gap will return a flux of 0 in the gap
        mask = self.sii_mask(self.redshift, self.wavelengths)

        self.valid = True

        #Conditional for if spectrum is sufficiently within masked doublet range
        #Limits of my doublet mask
        upper_mask_limit = 6850 * (1 + self.redshift)
        lower_mask_limit = 6600 * (1 + self.redshift)
        mask_width = upper_mask_limit - lower_mask_limit

        #Limits of data spectrum
        upper_data_limit = np.max(self.wavelengths)
        lower_data_limit = np.min(self.wavelengths)

        overlap = max(0, min(upper_mask_limit, upper_data_limit) - max(lower_mask_limit, lower_data_limit))
        overlap_ratio = overlap / mask_width
        #Fraction of doublet range in overlap - less than 90% of doublet in spectrum
        if overlap_ratio == 0 :
            print(f"SII doublet is not in range")
            self.valid = False
            return 

        #Reject if more than 10% of the doublet window is out of range
        elif overlap_ratio < 0.9:
            print(f"{overlap_ratio * 100}% of SII doublet window is out of range")
            self.valid = False
            return        

        #Conditional for if doublet overlaps with chip gap
        masked_flux = self.flux[mask]
        #Bad pixels are either in a chip gap, or otherwise have NaN fluxes for whatever reason
        bad_pixels = (masked_flux == 0) | np.isnan(masked_flux)

        #Reject doublet measurement if the chip gap or other issues affects more than 10% of pixels in the mask
        if np.count_nonzero(bad_pixels) > 0.1 * len(masked_flux):
            print(f"Doublet interrupted by chip gap")
            self.valid = False
            return

        #Cut out the area of the spectrum surrounding the [SII] doublet to get just the continuum
        continuum_mask = self.flux[~mask]
        #Residuals of estimated continuum subtracted spectra
        residual = continuum_mask - np.median(continuum_mask)
        #NMAD of residuals with normalisation constant assumed to be 1.4826
        nmad = 1.4826 * np.median(np.abs(residual - np.median(residual)))
        scale_factor = nmad / np.median(self.flux_err[~mask])
        #Multiply flux errors by the scale factor
        self.flux_err = scale_factor * self.flux_err

    def spectrum_model(self, amp6716, amp6731, sigma, redshift, cont_const, cont_lin):
        """Create a theoretical model of the spectrum around the [SII] doublet as
        a pair of Gaussians for the doublet and a linear function to model the 
        continuum.""" 
        continuum = models.Polynomial1D(1, c0 = cont_const, c1 = cont_lin)
        sii6716 = models.Gaussian1D(amplitude = amp6716, mean = 6716 * (1 + redshift), stddev = sigma)
        sii6731 = models.Gaussian1D(amplitude = amp6731, mean = 6731 * (1 + redshift), stddev = sigma) 
        model = continuum + sii6716 + sii6731

        #Tie linewidths together - they should physically be equal
        model.stddev_2.tied = lambda m: m.stddev_1
        #Tie redshifts together - ditto
        model.mean_2.tied = lambda m: m.mean_1 * (6731 / 6716)

        return model

    def model_fit(self):
        """Use Astropy Lev-Mar Least Squares fitter to fit the model to the data"""
        if not self.valid:
            return 
    
        mask = self.sii_mask(self.redshift, self.wavelengths)
        fitter = fitting.LevMarLSQFitter(calc_uncertainties=True)
        #Initial guess of fit parameters
        initial_guess = self.spectrum_model(np.max(self.flux[mask]), np.max(self.flux[mask]), 5 * (1 + self.redshift), self.redshift, np.median(self.flux[mask]) , 0)
        #Fit model to masked data
        fit_to_model = fitter(initial_guess, self.wavelengths[mask], self.flux[mask], weights = 1 / self.flux_err[mask], maxiter=1000)
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

    def plot_fit(self, output_file, residual_file):
        """Plot empirical data with resultant fitted data from Lev-Mar method"""
        if not self.valid:
            return
        mask = self.sii_mask(self.redshift, self.wavelengths)
        #Plot empirical data
        plt.plot(self.wavelengths[mask], self.flux[mask], color = "limegreen")
        #plt.errorbar(self.wavelengths[mask], self.flux[mask], yerr = self.flux_err[mask], color = "limegreen")
        plt.fill_between(self.wavelengths[mask], self.flux[mask] - self.flux_err[mask], 
                        self.flux[mask] + self.flux_err[mask], alpha=0.5, color = "limegreen") 
        
        #Plot fitted data
        fit_results = self.model_fit()

        if fit_results is None:
            return

        fitted_model = fit_results[0]

        #Expand wavelength data for less janky curve
        x_data = np.linspace(np.min(self.wavelengths[mask]), np.max(self.wavelengths[mask]), 1000)   
        fitted_data = fitted_model(x_data)

        plt.plot(x_data, fitted_data, color = "mediumpurple")
        #plt.axvline(6716 * (1+self.redshift), color = "darkblue")
        #plt.axvline(6731 * (1+self.redshift), color = "darkblue")
        plt.xlabel(r"Lab-frame wavelength ($\AA$)")
        plt.ylabel(r"Flux ($10^{-20}$erg/s/$cm^{-2}$/$\AA$)")
        plt.title(f"Target {self.target_id} \n with disperser {self.disperser}. z = {self.redshift}")
        plt.savefig(output_file)
        plt.close()

        #Residuals
        resid_fit_data = fitted_model(self.wavelengths[mask])
        resid = resid_fit_data - self.flux[mask]
        plt.plot(self.wavelengths[mask], resid, color = "orange")
        plt.axvline(6716 * (1+self.redshift), color = "darkblue")
        plt.axvline(6731 * (1+self.redshift), color = "darkblue")
        plt.axhline(0, color = "blue", ls = ":")
        plt.xlabel(r"Lab-frame wavelength ($\AA$)")
        plt.ylabel(r"Residual ($10^{-20}$erg/s/$cm^{-2}$/$\AA$)")
        plt.savefig(residual_file)
        plt.close()

    def chi_squared_test(self, fit_to_model): 
        """Test the goodness of fit of the model to the data using a chi-squared test. Reject if the reduced chi-squared is not within the set tolerance value of 1.0"""

        mask = self.sii_mask(self.redshift, self.wavelengths)

        observed_flux = self.flux[mask]
        fitted_flux = fit_to_model(self.wavelengths[mask])
        flux_err = self.flux_err[mask]

        residuals = (observed_flux - fitted_flux) / flux_err
        chi_squared = np.sum(residuals**2)

        number_of_data_points = len(observed_flux)
        number_of_free_parameters = number_of_free_parameters = sum(not (getattr(fit_to_model, name).tied)
        for name in fit_to_model.param_names)
        degrees_of_freedom = number_of_data_points - number_of_free_parameters

        reduced_chi_squared = chi_squared / degrees_of_freedom

        return reduced_chi_squared 

    def compute_density(self, flux6716, flux6731, T_e = 15000):
        """Calculate the electron density from the computed fluxes"""

        emission_line_ratio = flux6716 / flux6731
        #Atomic class
        S2 = pn.Atom('S', 2)

        n_e = S2.getTemDen(emission_line_ratio, tem = T_e, wave1 = 6716, wave2 = 6731)

        return n_e

    def monte_carlo(self, no_realisations = 500):
        """Run a Monte Carlo on however many realisations of the masked spectrum"""

        if not self.valid:
            return None
        
        mask = self.sii_mask(self.redshift, self.wavelengths)
        wavelengths = self.wavelengths[mask]
        flux = self.flux[mask]
        flux_err = self.flux_err[mask]

        flux6716_gauss_dist = np.zeros(no_realisations)
        flux6731_gauss_dist = np.zeros(no_realisations)
        redshift_gauss_dist = np.zeros(no_realisations)

        n_e_gauss_dist = np.zeros(no_realisations)
    

        #Initial guess of model for all N realisations
        fit_results = self.model_fit()

        if fit_results is None:
            return None
        
        model_guess, fitter = fit_results
        #Check if tying worked
        #print(model_guess.stddev_1.value, model_guess.stddev_2.value)
        #print(model_guess.mean_1.value, (6716/6731) * model_guess.mean_2.value)

        #Check chi-squared of fit to see if we should go through with the Monte Carlo
        reduced_chi_squared = self.chi_squared_test(model_guess)
        print(f"Reduced chi-squared for target {self.target_id} with disperser {self.disperser} ={reduced_chi_squared}" )

        if reduced_chi_squared > 2.0:
            print(f"Chi-squared test failed for target {self.target_id} with disperser {self.disperser}.")
            self.valid = False
            return None

        #Run MC
        for i in range(no_realisations):
            realisation = np.random.normal(loc = flux, scale = flux_err)

            initial_guess = model_guess.copy()
            fit_to_model = fitter(initial_guess, wavelengths, realisation, weights = 1 / self.flux_err[mask], maxiter=100)

            amp6716, mean6716, sigma6716, amp6731, mean6731, sigma6731 = fit_to_model.parameters[2:]

            #Add to data arrays

            flux6716_gauss_dist[i] = amp6716 * sigma6716 * np.sqrt(2 * np.pi)
            flux6731_gauss_dist[i] = amp6731 * sigma6731 * np.sqrt(2 * np.pi)

            redshift_gauss_dist[i] = mean6716 / 6716 - 1

            #Maximum density limit 

            if flux6716_gauss_dist[i]/flux6731_gauss_dist[i] < 0.45:
                n_e_gauss_dist[i] = 2000 #Topping 2025

            elif flux6716_gauss_dist[i]/flux6731_gauss_dist[i] > 1.45:
                n_e_gauss_dist[i] = 10 #cm^-3

            else:
                n_e_gauss_dist[i] = self.compute_density(flux6716_gauss_dist[i], flux6731_gauss_dist[i])


        #Take medians and errors from distributions
        flux6716_median = np.nanmedian(flux6716_gauss_dist)
        flux6716_error_upper = np.nanpercentile(flux6716_gauss_dist, 84) - flux6716_median
        flux6716_error_lower = flux6716_median - np.nanpercentile(flux6716_gauss_dist, 16)

        flux6716 = np.nanmedian(flux6716_gauss_dist)
        flux6716_error = (flux6716_error_upper + flux6716_error_lower) / 2

        flux6731_median = np.nanmedian(flux6731_gauss_dist)
        flux6731_error_upper = np.nanpercentile(flux6731_gauss_dist, 84) - flux6731_median
        flux6731_error_lower = flux6731_median - np.nanpercentile(flux6731_gauss_dist, 16)
        
        flux6731 = np.nanmedian(flux6731_gauss_dist)
        flux6731_error = (flux6731_error_upper + flux6731_error_lower) / 2

        redshift_median = np.nanmedian(redshift_gauss_dist)
        redshift_error_upper = np.nanpercentile(redshift_gauss_dist, 84) - redshift_median
        redshift_error_lower = redshift_median - np.nanpercentile(redshift_gauss_dist, 16)
        
        redshift = np.nanmedian(redshift_gauss_dist)
        redshift_error = (redshift_error_upper + redshift_error_lower) / 2

        n_e_median = np.nanmedian(n_e_gauss_dist)
        n_e_error_upper = np.nanpercentile(n_e_gauss_dist, 84) - n_e_median
        n_e_error_lower = n_e_median - np.nanpercentile(n_e_gauss_dist, 16)

        n_e = n_e_median

        print(f"Density = {n_e} + {n_e_error_upper} -{n_e_error_lower} cm^-3")

        return (flux6716, flux6716_error, flux6731, flux6731_error,
                redshift, redshift_error, n_e, n_e_error_upper, n_e_error_lower)





        

        


