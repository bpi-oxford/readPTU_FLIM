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
from scipy.ndimage import gaussian_filter

# PlantSeg imports for cell segmentation
from plantseg.predictions.functional.predictions import unet_predictions
from plantseg.segmentation.functional.segmentation import mutex_ws

# Custom module imports
from PTU_ScanRead import PTU_ScanRead, Process_Frame, mHist, cim
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
#%% extra definitions

 
# ============================================================================
# VIGNETTE CORRECTION FUNCTION
# ============================================================================

def vignette_correction(int_im, blur_factor=6, lower_percentile=2, upper_percentile=98):
    """
    Apply vignette correction to intensity image following MATLAB algorithm.
    
    This function corrects for illumination non-uniformity by:
    1. Creating a blurred background estimate using Gaussian filter
    2. Normalizing the original image by this background
    3. Applying percentile-based contrast enhancement
    4. Final normalization to [0,1] range
    
    MATLAB equivalent:
    bblur = imgaussfilt(int_im, min(size(int_im))/6);
    bblur = bblur ./ max(bblur(:));
    I2 = int_im ./ bblur;
    lb = prctile(I2(:), 2);
    ub = prctile(I2(:), 98);
    I2(I2 < lb) = lb;
    I2(I2 > ub) = ub;
    I2 = I2 - lb;
    int2 = I2 ./ max(I2(:));
    
    Parameters
    ----------
    int_im : ndarray
        Input intensity image
    blur_factor : float, optional (default=6)
        Gaussian blur factor = min(image_size) / blur_factor
    lower_percentile : float, optional (default=2)
        Lower percentile for contrast adjustment (removes dark outliers)
    upper_percentile : float, optional (default=98)
        Upper percentile for contrast adjustment (removes bright outliers)
        
    Returns
    -------
    int2 : ndarray
        Vignette-corrected and normalized intensity image [0,1]
    bblur : ndarray
        Gaussian-blurred background estimate
    """
    
    print(f"Applying vignette correction...")
    print(f"  - Image shape: {int_im.shape}")
    print(f"  - Blur factor: 1/{blur_factor} of min image dimension")
    
    # Calculate Gaussian blur sigma: min(image_size) / blur_factor
    sigma = min(int_im.shape) / blur_factor
    print(f"  - Gaussian sigma: {sigma:.2f} pixels")
    
    # Create blurred background (excitation floor estimate)
    # MATLAB: bblur = imgaussfilt(int_im, min(size(int_im))/6);
    bblur = gaussian_filter(int_im, sigma=sigma)
    
    # Normalize blurred background to [0,1] with max=1
    # MATLAB: bblur = bblur ./ max(bblur(:));
    bblur = bblur / np.max(bblur)
    
    # Vignette correction: divide by normalized background
    # MATLAB: I2 = int_im ./ bblur;
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    I2 = int_im / (bblur + epsilon)
    
    # Calculate percentile bounds for contrast enhancement
    # MATLAB: lb = prctile(I2(:), 2); ub = prctile(I2(:), 98);
    lb = np.percentile(I2, lower_percentile)  # Lower bound (2nd percentile)
    ub = np.percentile(I2, upper_percentile)  # Upper bound (98th percentile)
    
    print(f"  - Intensity range before clipping: [{np.min(I2):.3f}, {np.max(I2):.3f}]")
    print(f"  - Percentile bounds: [{lb:.3f}, {ub:.3f}]")
    
    # Clip values to percentile bounds
    # MATLAB: I2(I2 < lb) = lb; I2(I2 > ub) = ub;
    I2 = np.clip(I2, lb, ub)
    
    # Subtract lower bound (baseline correction)
    # MATLAB: I2 = I2 - lb;
    I2 = I2 - lb
    
    # Final normalization to [0,1]
    # MATLAB: int2 = I2 ./ max(I2(:));
    int2 = I2 / np.max(I2) if np.max(I2) > 0 else I2
    
    print(f"  - Final intensity range: [{np.min(int2):.3f}, {np.max(int2):.3f}]")
    
    return int2, bblur

#%%

auto_det = 0;  # Detector ID for autofluorescence channel detection
auto_PIE = 1;  # Laser pulse number for autofluorescence channel (PIE window)
flag_win = True; # Flag for window based lifetime estimation


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
        Amp,Z = PatternMatchIm(tcspc_pix[:,:,:,0], patterns, mode='PIRLS')
        int_im = np.sum(tcspc_pix[:,:,:,0],axis=2)
        
    if flag_win is True:  # sliding window approach
        win_size = 8  # Size of the sliding window (8x8 pixels)
        step = 2     # Step size for window movement
        
        print(f"Processing with sliding window: {win_size}x{win_size}, step={step}")
        
        # Calculate output dimensions
        n_win_x = (nx - win_size) // step + 1
        n_win_y = (ny - win_size) // step + 1
        
        print(f"Original image: {nx}x{ny}, Windows: {n_win_x}x{n_win_y}")
        
        # Initialize output arrays for windowed analysis
        Amp_win = np.zeros((n_win_x, n_win_y, len(taufit) + 1))
        Z_win = np.zeros((n_win_x, n_win_y, tcspc_pix.shape[2]))
        
        # Create tcspc_win by sliding window aggregation
        tcspc_win = np.zeros((n_win_x, n_win_y, tcspc_pix.shape[2]))
        int_im = np.zeros((n_win_x, n_win_y))
        print("Aggregating TCSPC data with sliding windows...")
        for i in range(n_win_x):
            for j in range(n_win_y):
                # Define window boundaries
                x_start = i * step
                x_end = x_start + win_size
                y_start = j * step
                y_end = y_start + win_size
                
                # Aggregate TCSPC data within the window (sum all pixels)
                tcspc_win[i, j, :] = np.sum(tcspc_pix[x_start:x_end, y_start:y_end, :, 0], axis=(0, 1))
                int_im[i,j] = np.sum(tcspc_win[i, j, :])
        
        print("Running pattern matching on windowed data...")
        # Apply pattern matching to the aggregated windowed data
        Amp_win, Z_win = PatternMatchIm(tcspc_win, patterns, mode='PIRLS')
        
        # Store windowed results
        Amp = Amp_win
        Z = Z_win
        print(f" Windowed analysis completed: {n_win_x}x{n_win_y} windows processed")

    # ============================================================================
    # AMPLITUDE VISUALIZATION
    # ============================================================================
    
    # normalize the Amps
    Amp = Amp/np.sum(Amp[:,:,1:],axis=2,keepdims=True)
    # Apply vignette correction to intensity image
    print("\nApplying vignette correction to intensity image...")
    int2, bblur = vignette_correction(int_im)
   
    print(f"Creating amplitude plots for {len(taufit)} lifetime components using cim...")
    
    # Create custom colormap (similar to MATLAB's viridis)
    import matplotlib.cm as cm
    custom_cmap = cm.viridis(np.linspace(0, 1, 64))[:, :3]  # 64x3 RGB array
    
    # Display each amplitude component using cim in separate figures
    # Note: cim() function creates its own figure layout and doesn't work well with subplots
    
    print("Displaying Background/Offset Amplitude...")
    plt.figure(figsize=(8, 6))
    cim(Amp[:, :, 0], int2**0.5, [np.min(Amp[:, :, 0]), np.max(Amp[:, :, 0])], 'v', custom_cmap)
    plt.suptitle('Background/Offset Amplitude', fontsize=14)
    
    print("Displaying lifetime component amplitudes...")
    for i, tau in enumerate(taufit):
        plt.figure(figsize=(8, 6))
        amp_data = Amp[:, :, i + 1]
        cim(amp_data, int2**0.5, [np.min(amp_data), np.max(amp_data)], 'v', custom_cmap)
        plt.suptitle(f'Component {i+1} Amplitude: τ = {tau:.2f} ns', fontsize=14)
    
    print(f" All {len(taufit)+1} amplitude components displayed using cim function")
    
    # Print amplitude statistics
    print(f"\nAmplitude Statistics:")
    print(f"Background: min={np.min(Amp[:,:,0]):.3f}, max={np.max(Amp[:,:,0]):.3f}, mean={np.mean(Amp[:,:,0]):.3f}")
    for i, tau in enumerate(taufit):
        amp_comp = Amp[:, :, i + 1]
        print(f"Component {i+1} (τ={tau:.2f}ns): min={np.min(amp_comp):.3f}, max={np.max(amp_comp):.3f}, mean={np.mean(amp_comp):.3f}")

   
    # Create average lifetime image (amplitude-weighted)
    tau_avg = np.zeros_like(int_im)
    for i, tau_val in enumerate(taufit):
        tau_avg += Amp[:, :, i + 1] * tau_val
    
    # Avoid division by zero
    intensity_safe = np.where(int2 > 0, int2, 1)
    tau_avg = np.where(int2 > 0, tau_avg / intensity_safe, 0)
       
    # Display FLIM image using cim function
    print("\nDisplaying FLIM image using cim function with vignette-corrected intensity...")
    
    # Create custom colormap (similar to MATLAB's jet)
    import matplotlib.cm as cm
    custom_cmap = cm.jet(np.linspace(0, 1, 64))[:, :3]  # 64x3 RGB array

    # Display lifetime image with vignette-corrected intensity overlay
   
    plt.figure(figsize=(10, 8))
    cim(tau_avg, int2**0.75, [1.5, 4], 'v', custom_cmap)
    plt.suptitle('FLIM Image: Lifetime with Vignette-Corrected Intensity Overlay', fontsize=14)
    plt.show()

   

 
 
 