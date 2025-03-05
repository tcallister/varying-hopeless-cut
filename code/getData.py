import numpy as np
import h5py
import json
import astropy.cosmology as cosmo
import astropy.units as u
from astropy.cosmology import Planck15
import sys, os
import pickle
dirname = os.path.dirname(__file__)

def loadInjections(ifar_threshold,hopeless_cut=None):

    # Read injection file
    mockDetections = h5py.File('./../input/endo3_bbhpop-LIGO-T2100113-v12.hdf5','r')

    # Total number of trial injections (detected or not)
    nTrials = mockDetections.attrs['total_generated']

    # Read out IFARs and SNRs from search pipelines
    ifar_1 = mockDetections['injections']['ifar_gstlal'][()]
    ifar_2 = mockDetections['injections']['ifar_pycbc_bbh'][()]
    ifar_3 = mockDetections['injections']['ifar_pycbc_hyperbank'][()]
    optimal_snr = mockDetections['injections']['optimal_snr_net'][()]

    # Determine which events pass IFAR threshold (O3) or SNR threshold (O1/O2)
    detected_O3 = (ifar_1>ifar_threshold) + (ifar_2>ifar_threshold) + (ifar_3>ifar_threshold)
    if hopeless_cut:
        passes_hopeless_cut = (optimal_snr>hopeless_cut)
    else:
        passes_hopeless_cut = np.full(optimal_snr.shape,True)

    detected_full = detected_O3*passes_hopeless_cut
    print(detected_O3.size,np.where(detected_O3==True)[0].size,np.where(passes_hopeless_cut==True)[0].size,np.where(detected_full==True)[0].size)

    # Get properties of detected sources
    m1_det = np.array(mockDetections['injections']['mass1_source'][()])[detected_full]
    m2_det = np.array(mockDetections['injections']['mass2_source'][()])[detected_full]
    s1x_det = np.array(mockDetections['injections']['spin1x'][()])[detected_full]
    s1y_det = np.array(mockDetections['injections']['spin1y'][()])[detected_full]
    s1z_det = np.array(mockDetections['injections']['spin1z'][()])[detected_full]
    s2x_det = np.array(mockDetections['injections']['spin2x'][()])[detected_full]
    s2y_det = np.array(mockDetections['injections']['spin2y'][()])[detected_full]
    s2z_det = np.array(mockDetections['injections']['spin2z'][()])[detected_full]
    z_det = np.array(mockDetections['injections']['redshift'][()])[detected_full]

    # This is dP_draw/(dm1*dm2*dz*ds1x*ds1y*ds1z*ds2x*ds2y*ds2z)
    precomputed_p_m1m2z_spin = np.array(mockDetections['injections']['sampling_pdf'][()])[detected_full]

    # In general, we'll want either dP_draw/(dm1*dm2*dz*da1*da2*dcost1*dcost2) or dP_draw/(dm1*dm2*dz*dchi_eff*dchi_p).
    # In preparation for computing these quantities, divide out by the component draw probabilities dP_draw/(ds1x*ds1y*ds1z*ds2x*ds2y*ds2z)
    # Note that injections are uniform in spin magnitude (up to a_max = 0.998) and isotropic, giving the following:
    dP_ds1x_ds1y_ds1z = (1./(4.*np.pi))*(1./0.998)/(s1x_det**2+s1y_det**2+s1z_det**2)
    dP_ds2x_ds2y_ds2z = (1./(4.*np.pi))*(1./0.998)/(s2x_det**2+s2y_det**2+s2z_det**2)
    precomputed_p_m1m2z = precomputed_p_m1m2z_spin/dP_ds1x_ds1y_ds1z/dP_ds2x_ds2y_ds2z

    return m1_det,m2_det,s1x_det,s1y_det,s1z_det,s2x_det,s2y_det,s2z_det,z_det,precomputed_p_m1m2z,nTrials

def getInjections(ifar_threshold,hopeless_cut=None):

    # Load
    m1_det,m2_det,s1x_det,s1y_det,s1z_det,s2x_det,s2y_det,s2z_det,z_det,p_draw_m1m2z,nTrials = loadInjections(ifar_threshold,hopeless_cut)

    # Derived parameters
    q_det = m2_det/m1_det
    a1_det = np.sqrt(s1x_det**2 + s1y_det**2 + s1z_det**2)
    a2_det = np.sqrt(s2x_det**2 + s2y_det**2 + s2z_det**2)
    cost1_det = s1z_det/a1_det
    cost2_det = s2z_det/a2_det

    # Draw probabilities for component spin magnitudes and tilts
    p_draw_a1a2cost1cost2 = (1./2.)**2*(1./0.998)**2*np.ones(a1_det.size)

    # Also compute factors of dVdz that we will need to reweight these samples during inference later on
    dVdz = 4.*np.pi*Planck15.differential_comoving_volume(z_det).to(u.Gpc**3*u.sr**(-1)).value

    # Store and save
    injectionDict = {
            'm1':m1_det,
            'm2':m2_det,
            'z':z_det,
            's1z':s1z_det,
            's2z':s2z_det,
            'a1':a1_det,
            'a2':a2_det,
            'cost1':cost1_det,
            'cost2':cost2_det,
            'dVdz':dVdz,
            'p_draw_m1m2z':p_draw_m1m2z,
            'p_draw_a1a2cost1cost2':p_draw_a1a2cost1cost2,
            'nTrials':nTrials
            }

    for key in injectionDict:
        if key!='nTrials':
            injectionDict[key] = np.array(injectionDict[key])

    return injectionDict

def getSamples(sample_limit=1000,bbh_only=True):

    """
    Function to load and preprocess BBH posterior samples for use in numpyro likelihood functions.
    
    Parameters
    ----------
    sample_limit : int or None
        If specified, will randomly downselect posterior samples, returning N=sample_limit samples per event (default None)
    bbh_only : bool
        If true, will exclude samples for BNS, NSBH, and mass-gap events (default True)

    Returns
    -------
    sampleDict : dict
        Dictionary containing posterior samples
    """

    # Dicts with samples:
    sampleFile = os.path.join(dirname,"./../../get-lvk-data/sampleDict_FAR_1_in_1_yr.pickle")
    sampleDict = np.load(sampleFile,allow_pickle=True)

    non_bbh = ['GW170817','S190425z','S190426c','S190814bv','S190917u','S200105ae','S200115j']
    if bbh_only:
        for event in non_bbh:
            print("Removing ",event)
            sampleDict.pop(event)

    event_names = list(sampleDict.keys())
    for event in event_names:
        if event[0]=="G":
            sampleDict.pop(event)

    for event in sampleDict:

        inds_to_keep = np.random.choice(np.arange(sampleDict[event]['m1'].size),size=sample_limit,replace=True)
        for key in sampleDict[event].keys():
            if key!='downselection_Neff':
                sampleDict[event][key] = sampleDict[event][key][inds_to_keep]

    return sampleDict

if __name__=="__main__":

    getInjections(1.,6.)
    getInjections(1.,7.)
    getInjections(1.,8.)
