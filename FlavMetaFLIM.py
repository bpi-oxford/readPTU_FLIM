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
from PTU_ScanRead import PTU_ScanRead, Process_Frame, mHist2
from FLIM_fitter import Calc_mIRF, FluoFit, DistFluoFit

# ============================================================================
# CONFIGURATION AND DATA LOADING SECTION
# ============================================================================

# File path configuration
# filename = r'D:\Collabs\fromYuexuan\Yuexuan_ptu_testfile.ptu'  # Input PTU file path
filename = r'D:\Collabs\fromKaitlyn\FLIM_liver\data\OTB5\RawImage1.ptu'
res_file = filename[:-4] + '_FLIM_data.pkl'  # Cached processed data file

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