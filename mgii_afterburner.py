import numpy as np
import fitsio,os
from desispec.io import read_spectra
from astropy.table import Table
from prospect.plotframes import create_model
from scipy.optimize import curve_fit
from scipy.ndimage.filters import gaussian_filter
from desispec.coaddition import coadd_cameras
import matplotlib.pyplot as plt
import itertools
from matplotlib.lines import Line2D

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





def fit_all_mgii(lambda_width,spectra_path,spectro_list,
                 exposure,qso_target=True,
                 gaussian_smoothing_fit=None,
                 template_dir=None,
                 archetypes_dir = None,
                 add_linear_term = True,
                 mask_mgii=None):
    if(type(spectro_list) == list):
        all_fit_results  = []
        all_target_id  = []
        if(mask_mgii is not None):
            all_fit_results_mask = []
        for i in range(len(spectro_list)):
            spectra_name = os.path.join(spectra_path,f"coadd-{spectro_list[i]}-{exposure}.fits")
            zbest_name = os.path.join(spectra_path,f"zbest-{spectro_list[i]}-{exposure}.fits")
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
            all_fit_results.append(fit_results)
            all_target_id.append(target_id)
            if(mask_mgii is not None):
                fit_results_mask = fig_mgii_line(target_id,redshift_redrock,
                                                flux,ivar_flux,model_flux,
                                                wavelength,lambda_width,
                                                add_linear_term = add_linear_term,
                                                gaussian_smoothing_fit=gaussian_smoothing_fit,
                                                mask_mgii=mask_mgii)
                all_fit_results_mask.append(fit_results_mask)

        all_fit_results = np.concatenate(all_fit_results)
        all_target_id = np.concatenate(all_target_id)
        if(mask_mgii is not None):
            all_fit_results_mask = np.concatenate(all_fit_results_mask)

    elif(spectro_list == "all"):
        spectra_name = os.path.join(spectra_path,f"coadd-{exposure}.fits")
        zbest_name = os.path.join(spectra_path,f"zbest-{exposure}.fits")
        all_target_id,redshift_redrock,flux,ivar_flux,model_flux,wavelength =get_spectra(spectra_name,
                                                                                         zbest_name,
                                                                                         lambda_width,
                                                                                         qso_target = qso_target,
                                                                                         template_dir = template_dir,
                                                                                         archetypes_dir = archetypes_dir)
        all_fit_results = fig_mgii_line(target_id,redshift_redrock,
                                        flux,ivar_flux,model_flux,
                                        wavelength,lambda_width,
                                        add_linear_term = add_linear_term,
                                        gaussian_smoothing_fit=gaussian_smoothing_fit)
        if(mask_mgii is not None):
            all_fit_results_mask = fig_mgii_line(target_id,redshift_redrock,
                                            flux,ivar_flux,model_flux,
                                            wavelength,lambda_width,
                                            add_linear_term = add_linear_term,
                                            gaussian_smoothing_fit=gaussian_smoothing_fit,
                                            mask_mgii=mask_mgii)
    else:
        raise ValueError("Please give a list of spectro number or all if exposures are merged")

    argsort = np.flip(np.argsort(all_fit_results[:,2]))
    all_target_id = all_target_id[argsort]
    all_fit_results = all_fit_results[argsort,:]
    if(mask_mgii is not None):
        all_fit_results_mask = all_fit_results_mask[argsort,:]
    else:
        all_fit_results_mask = None
    return(all_target_id,all_fit_results,all_fit_results_mask)



def fit_all_double_mgii(lambda_width,spectra_path,spectro_list,
                        exposure,width_mask_mgii,qso_target=True,
                        gaussian_smoothing_fit=None,
                        template_dir=None,
                        archetypes_dir = None,
                        add_linear_term = True):
    if(type(spectro_list) == list):
        all_fit_results  = []
        all_fit_results_mask  = []
        all_target_id  = []
        for i in range(len(spectro_list)):
            spectra_name = os.path.join(spectra_path,f"coadd-{spectro_list[i]}-{exposure}.fits")
            zbest_name = os.path.join(spectra_path,f"zbest-{spectro_list[i]}-{exposure}.fits")
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
            fit_results_mask = fig_mgii_line(target_id,redshift_redrock,
                                            flux,ivar_flux,model_flux,
                                            wavelength,lambda_width,
                                            add_linear_term = add_linear_term,
                                            gaussian_smoothing_fit=gaussian_smoothing_fit,
                                            mask_mgii=width_mask_mgii)
            all_fit_results.append(fit_results)
            all_fit_results_mask.append(fit_results_mask)
            all_target_id.append(target_id)

        all_fit_results = np.concatenate(all_fit_results)
        all_target_id = np.concatenate(all_target_id)

    elif(spectro_list == "all"):
        spectra_name = os.path.join(spectra_path,f"coadd-{exposure}.fits")
        zbest_name = os.path.join(spectra_path,f"zbest-{exposure}.fits")
        all_target_id,redshift_redrock,flux,ivar_flux,model_flux,wavelength =get_spectra(spectra_name,
                                                                                         zbest_name,
                                                                                         lambda_width,
                                                                                         qso_target = qso_target,
                                                                                         template_dir = template_dir,
                                                                                         archetypes_dir = archetypes_dir)
        all_fit_results = fig_mgii_line(target_id,redshift_redrock,
                                        flux,ivar_flux,model_flux,
                                        wavelength,lambda_width,
                                        add_linear_term = add_linear_term,
                                        gaussian_smoothing_fit=gaussian_smoothing_fit)
    else:
        raise ValueError("Please give a list of spectro number or all if exposures are merged")

    argsort = np.flip(np.argsort(all_fit_results[:,2]))
    all_target_id = all_target_id[argsort]
    all_fit_results = all_fit_results[argsort,:]
    return(all_target_id,all_fit_results)




def read_fit(name_fits):
    h = fitsio.FITS(name_fits,"r")[1]
    target_id = h["TARGET_ID"][:]
    fit_results = np.zeros((target_id.shape[0],9))
    fit_results[:,0] = h["CHI2_GAUSS"][:]
    fit_results[:,1] = h["CHI2_RR"][:]
    fit_results[:,2] = h["DELTA_CHI2"][:]
    fit_results[:,3] = h["A"][:]
    fit_results[:,6] = h["VAR_A"][:]
    fit_results[:,4] = h["SIGMA"][:]
    fit_results[:,7] = h["VAR_SIGMA"][:]
    fit_results[:,5] = h["B"][:]
    fit_results[:,8] = h["VAR_B"][:]
    if(len(h) > 9):
        fit_results[:,9] = h["C"]
        fit_results[:,10] = h["VAR_C"]
    return(target_id,fit_results)



def plot_scatters(all_fit_results,fit_results_vi,fit_results_fit,mindeltachi2,minsigma,maxsigma,min_error_amplitude_A,nameout=None):

    dict_plot = {}
    dict_plot[r"$\Delta\chi^{2}$"] = (all_fit_results[:,2],fit_results_vi[:,2],
                              fit_results_fit[:,2],0,100,mindeltachi2)
    dict_plot[r"$\sigma$"] = (np.abs(all_fit_results[:,4]),np.abs(fit_results_vi[:,4]),
                          np.abs(fit_results_fit[:,4]),0,100,minsigma,maxsigma)
    dict_plot[r"$A/\sigma(A)$"] = (all_fit_results[:,3]/np.sqrt(all_fit_results[:,6]),
                             fit_results_vi[:,3]/np.sqrt(fit_results_vi[:,6]),
                             fit_results_fit[:,3]/np.sqrt(fit_results_fit[:,6]),
                             0,8,min_error_amplitude_A)
    fig,ax=plt.subplots(1,3,figsize=(10,5),sharex=True)
    color = ["C0","C1","k"]
    i = 0
    for elt in itertools.combinations(list(dict_plot.keys()),2):
        ax[i].scatter(dict_plot[elt[0]][0],dict_plot[elt[1]][0],color=color[0],label="all")
        ax[i].scatter(dict_plot[elt[0]][1],dict_plot[elt[1]][1],color=color[1])
        ax[i].scatter(dict_plot[elt[0]][2],dict_plot[elt[1]][2], s=60, facecolors='none', edgecolors=color[2])

        ax[i].set_xlim([dict_plot[elt[0]][3],dict_plot[elt[0]][4]])
        ax[i].set_ylim([dict_plot[elt[1]][3],dict_plot[elt[1]][4]])
        ax[i].axvline(dict_plot[elt[0]][5],color="k")
        if(len(dict_plot[elt[0]])>6): ax[i].axvline(dict_plot[elt[0]][6],color="k")
        ax[i].axhline(dict_plot[elt[1]][5],color="k")
        if(len(dict_plot[elt[1]])>6): ax[i].axhline(dict_plot[elt[1]][6],color="k")
        ax[i].set_xlabel(elt[0])
        ax[i].set_ylabel(elt[1])
        i = i +1

    ax[1].set_title(nameout + " scatter plots")
    legend = ["all","VI","fit"]
    legend_elements = [Line2D([0], [0], color=color[i], lw=1, label=legend[i]) for i in range(len(legend))]
    ax[2].legend(handles=legend_elements,loc = "upper right")
    if(nameout is not None):
        fig.savefig(nameout)

def plot_target_id(target_id,fit_results,name,
                   spectra_path,exposure,
                   spectro_list,lambda_width,
                   qso_target=True,
                   template_dir=None,
                   archetypes_dir = None,
                   add_linear_term = True,
                   gaussian_smoothing_plot=1):



    mgii_peak_1 = 2803.5324
    mgii_peak_2 = 2796.3511
    mean_mgii_peak = (mgii_peak_1 + mgii_peak_2)/2

    if(type(spectro_list) == list):
        all_target_id,all_model_flux,all_flux,all_redshift_redrock  = [],[],[],[]
        for i in range(len(spectro_list)):
            spectra_name = os.path.join(spectra_path,f"coadd-{spectro_list[i]}-{exposure}.fits")
            zbest_name = os.path.join(spectra_path,f"zbest-{spectro_list[i]}-{exposure}.fits")
            targ_id,redshift_redrock,flux,ivar_flux,model_flux,wavelength =get_spectra(spectra_name,
                                                                                         zbest_name,
                                                                                         lambda_width,
                                                                                         qso_target = qso_target,
                                                                                         template_dir = template_dir,
                                                                                         archetypes_dir = archetypes_dir)
            all_target_id.append(targ_id)
            all_model_flux.append(model_flux)
            all_flux.append(flux)
            all_redshift_redrock.append(redshift_redrock)

        all_target_id = np.concatenate(all_target_id)
        all_model_flux = np.concatenate(all_model_flux)
        all_flux = np.concatenate(all_flux)
        all_redshift_redrock = np.concatenate(all_redshift_redrock)

    elif(spectro_list == "all"):
        spectra_name = os.path.join(spectra_path,f"coadd-{exposure}.fits")
        zbest_name = os.path.join(spectra_path,f"zbest-{exposure}.fits")
        all_target_id,all_redshift_redrock,all_flux,ivar_flux,all_model_flux,wavelength =get_spectra(spectra_name,
                                                                                                 zbest_name,
                                                                                                 lambda_width,
                                                                                                 qso_target = qso_target,
                                                                                                 template_dir = template_dir,
                                                                                                 archetypes_dir = archetypes_dir)
    all_mgii_peak_observed_frame = (all_redshift_redrock+1) * mean_mgii_peak
    for i in range(len(target_id)):
        mask_target_id = all_target_id == target_id[i]
        centered_wavelenght = wavelength - all_mgii_peak_observed_frame[mask_target_id]
        mask_wave = np.abs(centered_wavelenght) < lambda_width/2

        if(add_linear_term):
            popt = np.zeros((4))
            popt[0:3] = fit_results[i][3:6]
            popt[3]  = fit_results[i][9]
            fit_function = lambda x,A,sigma,B,C : A * np.exp(-1.0 * (x)**2 / (2 * sigma**2)) + B + C * x
        else:
            popt = fit_results[i][3:6]
            fit_function = lambda x,A,sigma,B : A * np.exp(-1.0 * (x)**2 / (2 * sigma**2)) + B
        x = np.linspace(np.min(centered_wavelenght[mask_wave]),
                        np.max(centered_wavelenght[mask_wave]),1000)
        y = fit_function(x, *popt)


        plt.figure(figsize = (20,5))
        flux_plot = np.transpose(gaussian_filter(all_flux[mask_target_id],gaussian_smoothing_plot))

        plt.plot(centered_wavelenght,flux_plot,'C1')
        plt.plot(centered_wavelenght,np.transpose(all_model_flux[mask_target_id]),'k')
        plt.plot(x,y,'g')
        plt.title(f"TARGET_ID = {target_id[i]}")
        plt.xlabel("Observed wavelength centered on MgII peak")
        plt.ylabel("Flux")
        plt.legend(["Spectrum","Redrock model","MgII gaussian fit",])
        plt.savefig(f"{name}_{target_id[i]}.png")
        plt.close()




def plot_list(all_target_id_masked,all_results,name,gaussian_smoothing_plot):

    for i in range(len(all_target_id_masked)):
        target_id = all_target_id_masked[i]
        plt.figure(figsize = (20,5))
        plt.plot(all_results[target_id]["wave"],gaussian_filter(all_results[target_id]["flux"],gaussian_smoothing_plot),'C1')
        plt.plot(all_results[target_id]["wave"],all_results[target_id]["mfl"],'k')
        plt.plot(all_results[target_id]["x"],all_results[target_id]["y"],'g')
        plt.title(f"TARGET_ID = {target_id}")
        plt.xlabel("Observed wavelength centered on MgII peak")
        plt.ylabel("Flux")
        plt.legend(["Spectrum","Redrock model","MgII gaussian fit",])
        plt.savefig(f"{name}_{target_id}.png")
        plt.close()





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
