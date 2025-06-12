# readPTU_FLIM Library
### For demo: Jupyter Notebook to work with PicoQuant PTU files and FLIM analysis
**`The library provides the capability to handle all PicoQuant's TCSPC Harps in T2 as well as T3 mode with advanced FLIM analysis capabilities.`**

- PicoQuant uses a bespoke file format called PTU to store data from time-tag-time-resolved (TTTR) measurements.<br/>
- Current file format (.ptu) and is subsequent to the former .pt2 or .pt3 and can handle both T2 and T3 acquisition modes for a variety of TCSPC devices (MultiHarp, HydraHarp, TimeHarp, PicoHarp, etc.) <br/>
- At the moment, the library was tested for FLIM data obtained using MultiHarp, HydraHarp and PicoHarp. <br/>
- Includes advanced cell segmentation using PlantSeg and comprehensive fluorescence lifetime fitting capabilities.

### **`What is available`**
- ✅ PTU file reading and photon stream processing
- ✅ FLIM data analysis with multi-exponential fitting
- ✅ Cell segmentation using deep learning (U-Net) models via PlantSeg
- ✅ Distributed fluorescence lifetime analysis
- ✅ Parameter optimization for segmentation quality
- ✅ Comprehensive TCSPC analysis tools
- ✅ Support for PIE (Pulsed Interleaved Excitation) data
- ✅ Multi-frame and multi-channel analysis

### **`What is not available !`**
- Option to save data after processing in original library (now available in extended modules) <br/>

#### !!! Help is only an email away for MATLAB implementation of the same library.

## 🚀 Quick Start with Anaconda

### Prerequisites
- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Python 3.8 or higher

### 1. Create and Activate Conda Environment

```bash
# Create a new conda environment
conda create -n ptu_flim python=3.9

# Activate the environment
conda activate ptu_flim
```

### 2. Install Core Dependencies

#### Core Scientific Computing Packages
```bash
# Install from conda-forge for better compatibility
conda install -c conda-forge numpy matplotlib scipy numba scikit-learn pandas
```

#### Advanced Analysis and Optimization
```bash
# Install optimization and curve fitting
conda install -c conda-forge scikit-optimize lmfit
```

#### Progress Bars and Utilities
```bash
# Install tqdm for progress bars
conda install -c conda-forge tqdm
```

#### Fast Histogram Computing
```bash
# Install fast-histogram for efficient histogram computation
conda install -c conda-forge fast-histogram
```

#### GPU Acceleration (Optional but Recommended)
```bash
# Update setuptools and pip first
python -m pip install -U setuptools pip

# Install CuPy for GPU acceleration (requires CUDA-compatible GPU)
pip install cupy-core

# Note: For full CUDA support, you may need a specific CuPy version
# Check your CUDA version with: nvidia-smi
# Then install appropriate version, e.g., for CUDA 11.x:
# pip install cupy-cuda11x
```

### 3. Install PlantSeg for Cell Segmentation (Optional but Recommended)

PlantSeg provides state-of-the-art cell segmentation capabilities:

```bash
# Install PlantSeg dependencies
conda install -c conda-forge pytorch torchvision torchaudio
pip install plantseg
```

### 4. Alternative: Install All Dependencies at Once

Create an `environment.yml` file:

```yaml
name: ptu_flim
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - numpy
  - matplotlib
  - scipy
  - numba
  - scikit-learn
  - pandas
  - tqdm
  - fast-histogram
  - scikit-optimize
  - lmfit
  - pytorch
  - torchvision
  - torchaudio
  - pip
  - pip:
    - plantseg
```

Then install:
```bash
conda env create -f environment.yml
conda activate ptu_flim
python -m pip install -U setuptools pip
pip install cupy-core
```

### 5. Verify Installation

Test your installation with:

```python
# Test core dependencies
import numpy as np
import matplotlib.pyplot as plt
import scipy
import numba
from sklearn.model_selection import ParameterGrid
import tqdm
from fast_histogram import histogramdd

# Test FLIM modules
from readPTU_FLIM import PTUreader
from FLIM_fitter import FluoFit, DistFluoFit, Calc_mIRF
from PTU_ScanRead import PTU_ScanRead, Process_Frame

# Test PlantSeg (if installed)
try:
    from plantseg.predictions.functional.predictions import unet_predictions
    from plantseg.segmentation.functional.segmentation import mutex_ws
    print("✅ PlantSeg successfully imported")
except ImportError:
    print("⚠️ PlantSeg not available - cell segmentation features disabled")

# Test CuPy (if installed)
try:
    import cupy as cp
    print(f"✅ CuPy successfully imported - GPU acceleration available")
    print(f"   CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
    print(f"   Available GPU memory: {cp.cuda.Device().mem_info[1] / 1e9:.1f} GB")
except ImportError:
    print("⚠️ CuPy not available - using CPU-only computations")
except Exception as e:
    print(f"⚠️ CuPy installed but GPU not accessible: {e}")

print("✅ All core dependencies successfully imported!")
```

## 📦 Package Dependencies

### Core Dependencies (Required)
- **numpy**: Numerical operations and array handling
- **matplotlib**: Plotting and visualization
- **scipy**: Scientific computing and optimization
- **numba**: Just-in-time compilation for performance
- **fast-histogram**: Fast histogram computation
- **tqdm**: Progress bars

### Analysis Dependencies (Required)
- **scikit-learn**: Machine learning utilities (parameter grid search, LinearRegression for PatternMatchIm)
- **pandas**: Data manipulation and analysis
- **lmfit**: Non-linear least-squares minimization
- **scikit-optimize**: Optimization algorithms

### Advanced Features (Optional)
- **plantseg**: Deep learning-based plant cell segmentation
- **pytorch**: Deep learning framework (required for PlantSeg)
- **cupy-core**: GPU-accelerated computing (NumPy-like API on CUDA)

### Basic Usage Example

```python
# Import the library
from readPTU_FLIM import PTUreader
import numpy as np
from matplotlib import pyplot as plt

# Basic PTU file reading
ptu_file = PTUreader('your_file.ptu', print_header_data=False)

# Get FLIM data stack (requires PTU file)
# flim_data_stack, intensity_image = ptu_file.get_flim_data_stack()

# Advanced usage with full pipeline
from PTU_ScanRead import PTU_ScanRead
head, im_sync, im_tcspc, im_chan, im_line, im_col, im_frame = PTU_ScanRead('your_file.ptu')
```

## 📚 Comprehensive Usage Examples

### 1. Basic PTU File Reading

```python
from readPTU_FLIM import PTUreader
import numpy as np
import matplotlib.pyplot as plt

# Read PTU file and display header information
ptu_file = PTUreader('Test_FLIM_image_daisyPollen_PicoHarp_2.ptu', print_header_data=True)

# Access raw photon data
sync = ptu_file.sync        # Macro photon arrival time
tcspc = ptu_file.tcspc      # Micro photon arrival time (TCSPC time bin resolution)
channel = ptu_file.channel  # Detection channel of TCSPC unit (≤8 for PQ hardware)
special = ptu_file.special  # Special event markers (Frame, LineStart, LineStop, etc.)
```

### 2. FLIM Data Stack Generation

```python
# Generate FLIM data stack from raw TTTR data
flim_data_stack, intensity_image = ptu_file.get_flim_data_stack()

# Data dimensions:
# flim_data_stack: (pixX, pixY, spectral_detection_channel, tcspc_bins)
# intensity_image: (pixX, pixY)

print(f"FLIM data shape: {flim_data_stack.shape}")
print(f"Intensity image shape: {intensity_image.shape}")

# Extract data from specific channel
channel_1_data = flim_data_stack[:, :, 0, :]  # Channel 1 TCSPC histograms

# Plot intensity image
plt.figure(figsize=(8, 6))
plt.imshow(intensity_image, cmap='viridis')
plt.colorbar(label='Photon Counts')
plt.title('FLIM Intensity Image')
plt.xlabel('X Pixel')
plt.ylabel('Y Pixel')
plt.show()
```

### 3. Advanced PTU Scanning and Processing

```python
from PTU_ScanRead import PTU_ScanRead, Process_Frame
from FLIM_fitter import Calc_mIRF, FluoFit, DistFluoFit

# Advanced PTU file processing with frame-by-frame analysis
filename = 'your_multi_frame_data.ptu'
head, im_sync, im_tcspc, im_chan, im_line, im_col, im_frame = PTU_ScanRead(filename)

# Process specific frame
frame_id = 0
frame_indices = np.where(np.array(im_frame) == frame_id)[0]

# Extract frame data
frame_sync = np.array(im_sync)[frame_indices]
frame_col = np.array(im_col)[frame_indices]
frame_line = np.array(im_line)[frame_indices]
frame_chan = np.array(im_chan)[frame_indices]
frame_tcspc = np.array(im_tcspc)[frame_indices]

# Process frame to get tag, tau, and TCSPC pixel data
tag, tau, tcspc_pix = Process_Frame(frame_sync, frame_col, frame_line,
                                   frame_chan, frame_tcspc, head,
                                   cnum=1, resolution=0.2)

print(f"Processed frame shape - Tag: {tag.shape}, Tau: {tau.shape}")
```

### 4. Cell Segmentation Pipeline

```python
from sklearn.model_selection import ParameterGrid
from plantseg.predictions.functional.predictions import unet_predictions
from plantseg.segmentation.functional.segmentation import mutex_ws
import pickle
import os

# Configuration for cell segmentation
filename = 'cell_data.ptu'
res_file = filename[:-4] + '_FLIM_data.pkl'
max_cell_Area = 5000  # Maximum cell area threshold

# Load or process FLIM data
if os.path.exists(res_file):
    with open(res_file, 'rb') as f:
        FLIM_data = pickle.load(f)
    im_sync = FLIM_data['im_sync']
    im_tcspc = FLIM_data['im_tcspc']
    im_chan = FLIM_data['im_chan']
    im_line = FLIM_data['im_line']
    im_col = FLIM_data['im_col']
    im_frame = FLIM_data['im_frame']
    head = FLIM_data['head']
else:
    head, im_sync, im_tcspc, im_chan, im_line, im_col, im_frame = PTU_ScanRead(filename)

# Channel configuration
mem_PIE = 2    # Membrane stain PIE window
mem_det = 2    # Membrane detector
auto_PIE = 1   # Autofluorescence PIE window
auto_det = 1   # Autofluorescence detector

# Process frame for segmentation
nz = 0  # Frame number
frame_indices = np.where(np.array(im_frame) == nz)[0]
tag, tau, tcspc_pix = Process_Frame(im_sync[frame_indices], im_col[frame_indices],
                                   im_line[frame_indices], im_chan[frame_indices],
                                   im_tcspc[frame_indices], head, cnum=1, resolution=0.2)

# Create membrane image for segmentation
pos = np.argmax(np.sum(tcspc_pix[:, :, :, mem_det*mem_PIE-1], axis=(0, 1)))
nCh = pos + int(np.ceil(5/0.2))  # 5ns gate width
img_mem = np.sum(tcspc_pix[:, :, pos:pos + nCh-1, mem_det*mem_PIE-1], axis=2)

# Normalize for neural network
img_np_scaled = (img_mem - np.min(img_mem)).astype(float)
img_np_scaled /= np.max(img_np_scaled)

# Apply U-Net segmentation
pred = unet_predictions(img_np_scaled[np.newaxis, :, :],
                       "confocal_2D_unet_ovules_ds2x",
                       'pioneering-rhino',
                       patch=[1, 512, 512])

# Parameter optimization for segmentation
param_grid = {
    "beta": [round(x, 1) for x in np.arange(0.5, 0.95, 0.05)],
    "post_minsize": [round(x, 1) for x in np.arange(190, 210, 10)],
}
params = list(ParameterGrid(param_grid))

# Test different parameters
results = []
for param in params:
    mask = mutex_ws(pred, superpixels=None, beta=param["beta"],
                   post_minsize=param["post_minsize"], n_threads=6)
    results.append({
        "beta": param["beta"],
        "post_minsize": param["post_minsize"],
        "mask": mask,
        "pred": pred[0, :, :]
    })

# Select best segmentation (most cells detected)
cell_counts = [len(np.unique(r['mask'])) for r in results]
best_idx = np.argmax(cell_counts)
best_mask = results[best_idx]['mask'][0]

print(f"Best segmentation: {cell_counts[best_idx]} cells detected")
print(f"Optimal parameters: β={results[best_idx]['beta']}, min_size={results[best_idx]['post_minsize']}")
```

### 5. Fluorescence Lifetime Fitting

```python
from FLIM_fitter import FluoFit, DistFluoFit, Calc_mIRF

# Calculate instrumental response function
tcspc_data = tcspc_pix[:, :, :, auto_det*auto_PIE-1]  # Autofluorescence channel
tcspcIRF = Calc_mIRF(head, tcspc_data[np.newaxis, :, :, np.newaxis])

# Filter noise from IRF
noise_threshold = 10**-4
tmpi = np.where((tcspcIRF/np.max(tcspcIRF)) < noise_threshold)[1]
tcspcIRF[:, tmpi, :] = 0

# Fit fluorescence lifetimes for each cell
ncells = len(np.unique(best_mask)) - 1  # Excluding background
tauCell = np.zeros((ncells, 3))  # Assuming 3 exponential components
ACell = np.zeros_like(tauCell)

resolution = 0.2  # ns
pulse_period = np.floor(head['MeasDesc_GlobalResolution']*1e9 + 0.5)

for cell_id in range(ncells):
    # Extract cell-specific TCSPC data
    cell_mask = (best_mask == cell_id + 1)
    cell_tcspc = np.sum(tcspc_data[cell_mask, :], axis=0)
    
    if np.sum(cell_tcspc) > 100:  # Minimum photon count threshold
        # Initial lifetime guesses
        tau0 = np.array([0.5, 2.0, 5.0])  # ns
        
        # Perform multi-exponential fitting
        taufit, A, _, _, _, _, _, _, _ = FluoFit(
            np.squeeze(tcspcIRF),
            cell_tcspc,
            pulse_period,
            resolution,
            tau0
        )
        
        # Store results
        tauCell[cell_id, :] = taufit
        ACell[cell_id, :] = A/np.sum(A)  # Normalized amplitudes
        
        print(f"Cell {cell_id+1}: τ = {taufit} ns, A = {ACell[cell_id, :]}")

# Alternative: Distributed lifetime fitting
for cell_id in range(min(5, ncells)):  # First 5 cells as example
    cell_mask = (best_mask == cell_id + 1)
    cell_tcspc = np.sum(tcspc_data[cell_mask, :], axis=0)
    
    if np.sum(cell_tcspc) > 100:
        cx, tau_dist, offset, shift, fit_curve, t_axis, error = DistFluoFit(
            cell_tcspc,
            pulse_period,
            resolution,
            np.squeeze(tcspcIRF),
            shift=(-10, 10),
            N=100,
            bild=1  # Show plots
        )
        
        print(f"Cell {cell_id+1} distributed fitting:")
        print(f"  Dominant lifetimes: {tau_dist[cx > 0.1*np.max(cx)]} ns")
        print(f"  Color shift: {shift} channels")
        print(f"  Fit error: {error}")
```

### 6. Pattern Matching Analysis

```python
from FLIM_fitter import PatternMatchIm, PatternMatch

# Example: Pattern matching for basis function decomposition
# This is useful for analyzing FLIM data with known basis functions

# Generate example FLIM data and basis functions
nx, ny, n_time = 64, 64, 256
n_basis = 3

# Create synthetic basis functions (e.g., different lifetime components)
time_axis = np.linspace(0, 10, n_time)
M = np.zeros((n_time, n_basis))
M[:, 0] = np.exp(-time_axis / 1.0)    # Fast component (1 ns)
M[:, 1] = np.exp(-time_axis / 3.0)    # Medium component (3 ns)
M[:, 2] = np.exp(-time_axis / 8.0)    # Slow component (8 ns)

# Normalize basis functions
M = M / np.sum(M, axis=0)

# Generate synthetic FLIM image data as linear combination of basis functions
np.random.seed(42)
true_coeffs = np.random.rand(nx, ny, n_basis)
y_image = np.zeros((nx, ny, n_time))

for i in range(nx):
    for j in range(ny):
        y_image[i, j, :] = M @ true_coeffs[i, j, :] + 0.1 * np.random.randn(n_time)

# Perform pattern matching analysis
print("=== Pattern Matching Analysis ===")

# Method 1: Default least squares
C_default, Z_default = PatternMatchIm(y_image, M, mode='Default')

# Method 2: Non-negative least squares
C_nonneg, Z_nonneg = PatternMatchIm(y_image, M, mode='Nonneg')

# Method 3: PIRLS (Poisson Iterative Reweighted Least Squares)
C_pirls, Z_pirls = PatternMatchIm(y_image, M, mode='PIRLS')

print(f"Original coefficients shape: {true_coeffs.shape}")
print(f"Recovered coefficients shape: {C_default.shape}")
print(f"Reconstruction error (Default): {np.mean((y_image - Z_default)**2):.6f}")
print(f"Reconstruction error (Non-neg): {np.mean((y_image - Z_nonneg)**2):.6f}")
print(f"Reconstruction error (PIRLS): {np.mean((y_image - Z_pirls)**2):.6f}")

# Single pixel pattern matching example
pixel_data = y_image[32, 32, :]  # Center pixel
c_pixel, z_pixel = PatternMatch(pixel_data, M, mode='Nonneg')

print(f"\nSingle pixel analysis:")
print(f"True coefficients: {true_coeffs[32, 32, :]}")
print(f"Recovered coefficients: {c_pixel}")
```

### 7. Data Visualization and Analysis

```python
# Create comprehensive FLIM visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Original intensity image
axes[0, 0].imshow(img_mem, cmap='gray')
axes[0, 0].set_title('Membrane Intensity')
axes[0, 0].axis('off')

# U-Net prediction
axes[0, 1].imshow(pred[0, :, :], cmap='viridis')
axes[0, 1].set_title('U-Net Prediction')
axes[0, 1].axis('off')

# Final segmentation
axes[0, 2].imshow(best_mask, cmap='tab20')
axes[0, 2].set_title(f'Cell Segmentation ({ncells} cells)')
axes[0, 2].axis('off')

# Lifetime maps (create pixel-wise lifetime images)
LIm = np.zeros((*img_mem.shape, 3))  # 3 lifetime components
AIm = np.zeros_like(LIm)             # 3 amplitude components

for c in range(ncells):
    cell_pixels = (best_mask == c + 1)
    LIm[cell_pixels, :] = tauCell[c, :]
    AIm[cell_pixels, :] = ACell[c, :]

# Plot lifetime components
for i in range(3):
    axes[1, i].imshow(LIm[:, :, i], cmap='hot', vmin=0, vmax=5)
    axes[1, i].set_title(f'τ{i+1} Lifetime Map')
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig('FLIM_analysis_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Summary statistics
print("\n=== FLIM Analysis Summary ===")
print(f"Total cells analyzed: {ncells}")
print(f"Average lifetimes per component:")
for i in range(3):
    valid_lifetimes = tauCell[:, i][tauCell[:, i] > 0]
    if len(valid_lifetimes) > 0:
        print(f"  τ{i+1}: {np.mean(valid_lifetimes):.2f} ± {np.std(valid_lifetimes):.2f} ns")
```

## 🗂️ Module Overview

### Core Modules
- **[`readPTU_FLIM.py`](readPTU_FLIM.py)**: Basic PTU file reading and FLIM data stack generation
- **[`PTU_ScanRead.py`](PTU_ScanRead.py)**: Advanced PTU scanning with frame-by-frame processing
- **[`FLIM_fitter.py`](FLIM_fitter.py)**: Comprehensive fluorescence lifetime fitting algorithms

### Analysis Pipelines
- **[`CellsegPTU_FLIM.py`](CellsegPTU_FLIM.py)**: Complete cell segmentation and FLIM analysis pipeline
- **[`FlavMetaFLIM.py`](FlavMetaFLIM.py)**: Specialized FLIM analysis for metabolic imaging

### Supported Hardware
The library supports PTU files from various PicoQuant TCSPC devices:
- **MultiHarp 150N/150P** (T2/T3 modes)
- **HydraHarp 400** (T2/T3 modes)
- **TimeHarp 260N/260P** (T2/T3 modes)
- **PicoHarp 300** (T2/T3 modes)

### Scanner Types
- Laser scanning microscopy (LSM)
- Piezo scanner systems
- Bidirectional and unidirectional scanning

## 📊 Test Data

**Test_FLIM_image_daisyPollen_PicoHarp_2.ptu**
- [Download test file](https://drive.google.com/file/d/1XtGL2yh_hJhaXIJhEDD5BpHNQXYQZX_p/view?usp=sharing)
- Acquired with: PicoHarp (T3 mode)
- Excitation: 485 nm laser
- Detection channels: 2
- Image: Daisy pollen FLIM data

![Interactive Demo Snapshot](Test_FLIM_image_daisyPollen_PicoHarp_2.png)

## 🔄 Recent Updates

- **2024**: Added comprehensive cell segmentation using PlantSeg U-Net models
- **2024**: Implemented distributed fluorescence lifetime fitting
- **2024**: Added PIE (Pulsed Interleaved Excitation) support
- **2024**: Multi-frame and multi-channel analysis capabilities
- **2024**: Parameter optimization for segmentation quality
- **2024**: Advanced FLIM fitting with maximum likelihood estimation
- **27 Aug, 2019**: Piezo Scanner data readability added to the library

## 💡 Tips for Best Results

1. **Memory Management**: For large PTU files, consider processing frame-by-frame
2. **Parameter Optimization**: Use parameter grid search for optimal segmentation
3. **Quality Control**: Filter cells by minimum photon count thresholds
4. **Visualization**: Always inspect segmentation results before analysis
5. **Performance**: Use `numba` compiled functions for speed-critical operations
6. **GPU Acceleration**: Install CuPy for significant speedup on CUDA-compatible GPUs
7. **CUDA Setup**: Ensure NVIDIA drivers and CUDA toolkit are properly installed for GPU support

## 🐛 Troubleshooting

### Common Issues
- **PlantSeg Import Error**: Ensure PyTorch is properly installed
- **CuPy Installation Issues**: Update setuptools and pip before installing cupy-core
- **CUDA Compatibility**: Check GPU compatibility and CUDA version with `nvidia-smi`
- **Memory Issues**: Reduce image size or process in chunks
- **Fitting Convergence**: Adjust initial parameter guesses
- **Segmentation Quality**: Optimize beta and post_minsize parameters
- **GPU Memory**: Monitor GPU memory usage with large datasets

## 🤝 Contributing

Feel free to contribute improvements, bug fixes, or new features through pull requests!

## 📞 Support

Need help with MATLAB implementation? Email support is available!
