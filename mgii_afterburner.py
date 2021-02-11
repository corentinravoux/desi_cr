import numpy as np
import fitsio,os
from desispec.io import read_spectra
from astropy.table import Table
from prospect.plotframes import create_model
from scipy.optimize import curve_fit
from scipy.ndimage.filters import gaussian_filter
from desispec.coaddition import coadd_cameras


tile = "80607"
spectra_path = f"/global/cfs/cdirs/desi/spectro/redux/blanc/tiles/{tile}/deep"
spectro = 0
exposure = f"{tile}-deep"

spectra_name = os.path.join(spectra_path,f"coadd-{spectro}-{exposure}.fits")
zbest_name = os.path.join(spectra_path,f"zbest-{spectro}-{exposure}.fits")


lambda_width = 250 # Width of the fit
qso_target = False # If True only analyze QSO targets
gaussian_smoothing_fit = None # Apply a gaussian smoothing of the flux for the fit
template_dir = None # Add the redrock templates path if the desi environment is not loaded
archetypes_dir = None # Add a archetypes directory to use the redrock-archetypes fit
add_linear_term = False # Add a linear term to the fit
name = f"{tile}_{spectro}"

max_sigma = 200
min_sigma = 10
min_deltachi2 = 25
min_signifiance_A = 4
min_A = 0.0


def get_spectra(spectra_name,zbest_name,lambda_width,qso_target = True,
                template_dir = None,archetypes_dir = None):


    spectra = read_spectra(spectra_name)
    if 'brz' not in spectra.bands:
        spectra = coadd_cameras(spectra)
    zbest = Table.read(zbest_name, 'ZBEST')
    if archetypes_dir is not None:
        model_wave, model_flux = create_model(spectra, zbest, archetype_fit=True,
                                              archetypes_dir=archetypes_dir,
                                              template_dir=template_dir)
    else:
        model_wave, model_flux = create_model(spectra, zbest, archetype_fit=False,
                                              archetypes_dir=None,
                                              template_dir=template_dir)

    fiber_status = spectra.fibermap["FIBERSTATUS"]
    target_id = spectra.fibermap["TARGETID"]
    sv1_desi_target = spectra.fibermap["SV1_DESI_TARGET"]
    wavelength = spectra.wave['brz']
    flux = spectra.flux['brz']
    ivar_flux = spectra.ivar['brz']

    redshift_redrock = zbest["Z"]
    spec_type = zbest["SPECTYPE"]


    fiber_ok = fiber_status == 0
    redrock_galaxies = (spec_type == "GALAXY")

    mgii_peak_1 = 2803.5324
    mgii_peak_2 = 2796.3511
    mean_mgii_peak = (mgii_peak_1 + mgii_peak_2)/2
    non_visible_peak = (redshift_redrock+1) * mean_mgii_peak < np.min(wavelength) + lambda_width/2
    non_visible_peak |= (redshift_redrock+1) * mean_mgii_peak > np.max(wavelength) - lambda_width/2

    galaxies_to_test = redrock_galaxies & fiber_ok & (~non_visible_peak)

    if(qso_target):
        galaxies_to_test &= ((sv1_desi_target&4) != 0)

    print("nb galaxies to test: ",len(galaxies_to_test[galaxies_to_test==True]))

    target_id_to_test = target_id[galaxies_to_test]
    redshift_redrock_to_test = redshift_redrock[galaxies_to_test]
    flux_to_test = flux[galaxies_to_test]
    ivar_flux_to_test = ivar_flux[galaxies_to_test]
    model_flux_to_test = model_flux[galaxies_to_test]

    print("Flux and RR best fit obtained")

    return(target_id_to_test,redshift_redrock_to_test,flux_to_test,
           ivar_flux_to_test,model_flux_to_test,wavelength)



def fig_mgii_line(target_id,redshift_redrock,flux,ivar_flux,model_flux,
                  wavelength,lambda_width,add_linear_term = True,
                  gaussian_smoothing_fit=None,mask_mgii=None):

    mgii_peak_1 = 2803.5324
    mgii_peak_2 = 2796.3511
    mean_mgii_peak = (mgii_peak_1 + mgii_peak_2)/2
    mgii_peak_observed_frame = (redshift_redrock+1) * mean_mgii_peak

    if(add_linear_term):
        fit_results = np.zeros((target_id.shape[0],11))
    else:
        fit_results = np.zeros((target_id.shape[0],9))

    for i in range(len(flux)):
        centered_wavelenght = wavelength - mgii_peak_observed_frame[i]
        mask_wave = np.abs(centered_wavelenght) < lambda_width/2
        if(mask_mgii) is not None:
            mask_wave &= np.abs(centered_wavelenght) > mask_mgii/2
        flux_centered = flux[i][mask_wave]
        model_flux_centered = model_flux[i][mask_wave]
        sigma_flux_centered = (1 /np.sqrt(ivar_flux[i]))[mask_wave]
        if(gaussian_smoothing_fit is not None):
            flux_centered = gaussian_filter(flux_centered,gaussian_smoothing_fit)
        if(add_linear_term):
            fit_function = lambda x,A,sigma,B,C : A * np.exp(-1.0 * (x)**2 / (2 * sigma**2)) + B + C * x
            try:
                popt, pcov = curve_fit(fit_function,
                                       xdata=centered_wavelenght[mask_wave],
                                       ydata=flux_centered, sigma=sigma_flux_centered,
                                       p0=[1.0,lambda_width/2,np.mean(flux_centered),0.0],
                                       bounds = ([-np.inf,-np.inf,-np.inf,-0.01],
                                       [np.inf,np.inf,np.inf,0.01]))
            except RuntimeError:
                print("Fit not converged")
                popt = np.full((4),0)
                pcov = np.full((4,4),0)
            fit_results[i][3:6] = popt[0:3]
            fit_results[i][6:9] = np.diag(pcov)[0:3]
            fit_results[i][9] = popt[3]
            fit_results[i][10] = np.diag(pcov)[3]


        else:
            fit_function = lambda x,A,sigma,B : A * np.exp(-1.0 * (x)**2 / (2 * sigma**2)) + B
            try:
                popt, pcov = curve_fit(fit_function,
                                       xdata=centered_wavelenght[mask_wave],
                                       ydata=flux_centered,
                                       sigma=sigma_flux_centered,
                                       p0=[1.0,lambda_width/2,np.mean(flux_centered)])
            except RuntimeError:
                print("Fit not converged")
                popt = np.full((3),0)
                pcov = np.full((3,3),0)
            fit_results[i][3:6] = popt
            fit_results[i][6:9] = np.diag(pcov)


        chi2_gauss = (np.sum(((flux_centered - fit_function(centered_wavelenght[mask_wave],*popt))/sigma_flux_centered)**2))
        chi2_RR = (np.sum(((flux_centered - model_flux_centered)/sigma_flux_centered)**2))

        fit_results[i][0] = chi2_gauss
        fit_results[i][1] = chi2_RR
        fit_results[i][2] = chi2_RR - chi2_gauss


    return(fit_results)

def create_mask_fit(fit_results,
                    max_sigma = None,
                    min_sigma = None,
                    min_deltachi2 = None,
                    min_A = None,
                    min_signifiance_A = None):
    mask = np.full(fit_results.shape[0],True)
    if(max_sigma is not None):
        mask &= np.abs(fit_results[:,4]) < max_sigma # sigma < max_sigma
    if(min_sigma is not None):
        mask &= np.abs(fit_results[:,4]) > min_sigma # sigma > min_sigma
    if(min_deltachi2 is not None):
        mask &= np.abs(fit_results[:,2]) > min_deltachi2 # deltachi2 > min_deltachi2
    if(min_A is not None):
        mask &= fit_results[:,3] > min_A # A > min_A
    if(min_signifiance_A is not None):
        mask &= np.abs(fit_results[:,3]) > min_signifiance_A * np.sqrt(np.abs(fit_results[:,6])) # A > min_signifiance_A * sigma(A)
    return(mask)





def mgii_fitter(spectra_name,zbest_name,lambda_width,
                qso_target = True,
                add_linear_term = True,
                gaussian_smoothing_fit=None,
                template_dir = None,
                archetypes_dir = None,
                max_sigma = None,
                min_sigma = None,
                min_deltachi2 = None,
                min_A = None,
                min_signifiance_A = None):

    target_id,redshift_redrock,flux,ivar_flux,model_flux,wavelength =get_spectra(spectra_name,
                                                                                 zbest_name,
                                                                                 lambda_width,
                                                                                 qso_target = qso_target,
                                                                                 template_dir = template_dir,
                                                                                 archetypes_dir = archetypes_dir)
    fit_results = fig_mgii_line(target_id,redshift_redrock,
                                flux,ivar_flux,model_flux,
                                wavelength,lambda_width,
                                add_linear_term = add_linear_term,
                                gaussian_smoothing_fit=gaussian_smoothing_fit)
    mask_fit = create_mask_fit(fit_results,
                               max_sigma = max_sigma,
                               min_sigma = min_sigma,
                               min_deltachi2 = min_deltachi2,
                               min_A = min_A,
                               min_signifiance_A = min_signifiance_A)
    return(mask_fit,fit_results,target_id)

def save_fit(target_id,fit_results,name_fits,header=None):
    a = fitsio.FITS(name_fits,"rw",clobber=True)
    h = {}
    h["TARGET_ID"] = target_id
    h["CHI2_GAUSS"] = fit_results[:,0]
    h["CHI2_RR"] = fit_results[:,1]
    h["DELTA_CHI2"] = fit_results[:,2]
    h["A"] = fit_results[:,3]
    h["VAR_A"] = fit_results[:,6]
    h["SIGMA"] = fit_results[:,4]
    h["VAR_SIGMA"] = fit_results[:,7]
    h["B"] = fit_results[:,5]
    h["VAR_B"] = fit_results[:,8]
    if(fit_results.shape[1] > 9):
        h["C"] = fit_results[:,9]
        h["VAR_C"] = fit_results[:,10]
    a.write(h,header=header)





if __name__ == "__main__":

    (mask_fit,fit_results,target_id) = mgii_fitter(spectra_name,zbest_name,lambda_width,
                                                    qso_target = qso_target,
                                                    add_linear_term = add_linear_term,
                                                    gaussian_smoothing_fit=gaussian_smoothing_fit,
                                                    template_dir = template_dir,
                                                    archetypes_dir = archetypes_dir,
                                                    max_sigma = max_sigma,
                                                    min_sigma = min_sigma,
                                                    min_deltachi2 = min_deltachi2,
                                                    min_A = min_A,
                                                    min_signifiance_A = min_signifiance_A)

    target_id_fit = target_id[mask_fit]
    fit_results_fit = fit_results[mask_fit]
    header = {"max_sigma" : max_sigma,"min_sigma" : min_sigma, "min_deltachi2" : min_deltachi2,
              "min_signifiance_A": min_signifiance_A,
              "min_A" : min_A}
    save_fit(target_id_fit,fit_results_fit,f"positive_results_fit_{name}.fits",header=header)
