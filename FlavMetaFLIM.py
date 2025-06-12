# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 11:01:28 2025

FlavMetaFLIM.py - FLIM Analysis Pipeline for FAD metabolic imaging

@author: narain karedla
"""

# Standard library imports
import numpy as np
import os
import pickle
import glob
from typing import Dict, List, Tuple, Optional, Any

# Third-party imports
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

# PlantSeg imports for cell segmentation
from plantseg.predictions.functional.predictions import unet_predictions
from plantseg.segmentation.functional.segmentation import mutex_ws

# Custom module imports
from PTU_ScanRead import PTU_ScanRead, Process_Frame, mHist
from FLIM_fitter import Calc_mIRF, FluoFit, DistFluoFit, PatternMatchIm

#%%
# ============================================================================
# CONFIGURATION AND DATA LOADING SECTION
# ============================================================================

# File path configuration
# filename = r'D:\Collabs\fromYuexuan\Yuexuan_ptu_testfile.ptu'  # Input PTU file path
filename = r'D:\Collabs\fromKaitlyn\Kaitlyn\data\OTB5\RawImage1.ptu'
res_file = filename[:-4] + '_FLIM_data.pkl'  # Cached processed data file
flag_allframes = True # flag for combining all frames together. In this case the data is processed as one single frame

# Analysis parameters
cnum = 1  # Number of PIE (Pulsed Interleaved Excitation) cycles, default value
max_cell_Area = 5000  # Maximum cell area threshold in pixels for filtering

# Data loading: Check if preprocessed data exists to avoid reprocessing
if os.path.exists(res_file):
    """
    Load previously processed FLIM data from pickle file.
    This saves significant processing time for repeated analysis.
    """
    print(f"Loading cached FLIM data from: {res_file}")
    pklname = glob.glob(res_file)
    with open(pklname[0], 'rb') as f:
        FLIM_data = pickle.load(f)
    
    # Extract photon stream data arrays
    im_sync = FLIM_data['im_sync']      # Sync pulse timestamps
    im_tcspc = FLIM_data['im_tcspc']    # Time-correlated single photon counting data
    im_chan = FLIM_data['im_chan']      # Detector channel information
    im_line = FLIM_data['im_line']      # Line scan coordinates
    im_frame = FLIM_data['im_frame']    # Frame numbers
    im_col = FLIM_data['im_col']        # Column coordinates
    head = FLIM_data['head']            # Header information from PTU file
    # timeF = FLIM_data['time']         # Optional: timing information
    
    # Update global namespace with all loaded data
    globals().update(FLIM_data)
else:
    """
    Read and process raw PTU file data.
    This step extracts photon stream data from the binary PTU format.
    """
    print(f"Processing raw PTU file: {filename}")
    head, im_sync, im_tcspc, im_chan, im_line, im_col, im_frame = PTU_ScanRead(filename)
     
# ============================================================================
# PARAMETER SETUP AND INITIALIZATION
# ============================================================================

# TCSPC (Time-Correlated Single Photon Counting) parameters
resolution = 0.2  # ns - temporal resolution for TCSPC histogram binning
dind = np.unique(im_chan)  # Array of unique detector channel indices
nFrames = head['ImgHdr_MaxFrames']  # Total number of frames in the dataset



# PIE (Pulsed Interleaved Excitation) configuration
if 'PIENumPIEWindows' in head:
    cnum = head['PIENumPIEWindows']  # Number of PIE cycles from header
    print(f"PIE cycles detected: {cnum}")


#%%

auto_det = 0;  # Detector ID for autofluorescence channel detection
auto_PIE = 1;  # Laser pulse number for autofluorescence channel (PIE window)
flag_win = False; # Flag for window based lifetime estimation
if flag_win is True:
    win_size = 8
    setp = 2


if flag_allframes is True:
    tag, tau, tcspc_pix = Process_Frame(
        im_sync, im_col, im_line, im_chan,
        im_tcspc, head, cnum=cnum, resolution=resolution
    )
    
    nx,ny,ch,p = np.shape(tag)
    
    Resolution = max(head['MeasDesc_Resolution'] * 1e9, resolution)  # resolution of 0.256 ns to calculate average lifetimes
    chDiv = np.ceil(1e-9 * Resolution / head['MeasDesc_Resolution'])
    SyncRate = 1.0 / head['MeasDesc_GlobalResolution']
    Ngate = round(head['MeasDesc_GlobalResolution'] / head['MeasDesc_Resolution'] * (head['MeasDesc_Resolution'] / Resolution / cnum) * 1e9)
    tmpCh = np.ceil(head['MeasDesc_GlobalResolution'] / head['MeasDesc_Resolution']) # total number of channels in the original tcspc histogram
    
    idx = im_chan == auto_det
    if len(idx) > 0:  # Check if there are photons to process
        tcspc_im = mHist((im_tcspc[idx] / chDiv).astype(np.int64) - int((auto_PIE-1)*tmpCh/cnum/chDiv),
                         np.arange(Ngate))[0]  # tcspc histograms for all the pixels at once!
    else:
        print(f"Warning: No photons found for channel {auto_det} - skipping FLIM analysis")

    tcspcIRF = Calc_mIRF(head, tcspc_im[np.newaxis,:,np.newaxis]);
    tmpi = np.where((tcspcIRF/np.max(tcspcIRF))<(10**-4))[1]
    tcspcIRF[:,tmpi,:]=0
    
    tau0 = np.array([0.3, 1.7, 6.0]) # initial guesses        
    taufit, A, _, zfit, patterns, _, _, _, _ = FluoFit(np.squeeze(tcspcIRF), tcspc_im, \
                                                        np.floor(head['MeasDesc_GlobalResolution']*10**9/cnum + 0.5), \
                                                        resolution, tau0, flag_ml=True)    
    patterns = patterns/np.sum(patterns,axis=0)    # normalized patterns
    if flag_win is False: # pixel by pixel
        #Amp = np.zeros((nx,ny,len(taufit)+1))
        Amp,_ = PatternMatchIm(tcspc_pix[:,:,:,0], patterns, mode='PIRLS')
        
    

 