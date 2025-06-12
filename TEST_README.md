# FLIM Pattern Matching Test Suite

This test suite validates the GPU-based pattern matching functionality in `FLIM_fitter.py` using synthetic TCSPC data with bi-exponential and tri-exponential decay models.

## Files Overview

### Test Scripts
- **`test_gpu_pattern_matching.py`** - Comprehensive test suite including GPU performance testing
- **`test_convol_basic.py`** - Basic CPU-only tests for core functionality
- **`run_tests.py`** - Simple test runner script

### Generated Files
- **`TEST_README.md`** - This documentation file

## Quick Start

### Option 1: Basic Tests (Recommended for first run)
```bash
python run_tests.py --basic
```

### Option 2: Full GPU Tests
```bash
python run_tests.py --full
```

### Option 3: Run Tests Directly
```bash
# Basic tests only
python test_convol_basic.py

# Full GPU tests
python test_gpu_pattern_matching.py
```

## Test Details

### Basic Tests (`test_convol_basic.py`)
- Tests the `Convol()` function with known inputs
- Validates bi-exponential and tri-exponential decay models
- Tests basic pattern matching without GPU requirements
- Generates diagnostic plots showing:
  - IRF and decay convolution results
  - Multi-exponential component decomposition
  - Pattern matching fit quality
  - Lifetime distribution analysis

### GPU Tests (`test_gpu_pattern_matching.py`)
- Creates synthetic FLIM data with spatial lifetime variation
- Tests all pattern matching modes: `Default`, `Nonneg`, `PIRLS`
- Benchmarks GPU performance across different data sizes
- Validates reconstruction accuracy with known ground truth
- Comprehensive visualization of results

## Test Features

### Synthetic Data Generation
- **IRF**: Generated using `IRF_Fun()` with realistic parameters
- **Decay Models**: Bi-exponential and tri-exponential with varying lifetimes
- **Noise**: Poisson noise added to simulate realistic TCSPC data
- **Spatial Variation**: FLIM data includes smooth spatial lifetime gradients

### Pattern Matching Modes
1. **Default**: Standard least squares fitting
2. **Nonneg**: Non-negative least squares (ensures physical constraints)
3. **PIRLS**: GPU-accelerated Poisson Iterative Reweighted Least Squares

### Performance Metrics
- Execution time per pixel
- Mean squared error (MSE) between data and reconstruction
- Coefficient extraction accuracy
- GPU memory efficiency

## Expected Outputs

### Generated Plots
- `convol_test_basic.png` - Basic convolution function validation
- `bi-exponential_test.png` - Bi-exponential decay model test
- `tri-exponential_test.png` - Tri-exponential decay model test
- `pattern_matching_simple.png` - Basic pattern matching results
- `lifetime_distribution.png` - Fitted lifetime distributions
- `gpu_pattern_matching_test_results.png` - Comprehensive GPU test results

### Console Output
- Test progress and timing information
- Performance benchmarks (pixels/second)
- Error metrics and validation results
- Summary of successful/failed tests

## Test Parameters

### Configurable Parameters
```python
# Data dimensions
nx, ny = 32, 32          # Image size
n_channels = 256         # Time channels
dt = 0.032              # Time resolution (ns)

# Decay models
bi_exp_lifetimes = [0.8, 3.5]      # ns
tri_exp_lifetimes = [0.3, 1.8, 5.2] # ns

# Noise and signal levels
max_counts = 1000       # Peak photon counts
noise_level = 1.0       # Poisson noise scaling
```

## Requirements

### Basic Tests
- numpy
- matplotlib
- scipy
- FLIM_fitter.py

### GPU Tests (Additional)
- cupy
- numba (with CUDA support)
- GPU with CUDA capability

## Troubleshooting

### Common Issues

1. **CUDA/GPU not available**
   ```
   Solution: Run basic tests only
   python run_tests.py --basic
   ```

2. **Memory errors with large datasets**
   ```
   Solution: Reduce data size in test parameters
   nx, ny = 16, 16  # Smaller image size
   ```

3. **Import errors**
   ```
   Solution: Ensure FLIM_fitter.py is in the same directory
   Check that all dependencies are installed
   ```

### Performance Notes
- GPU tests require significant memory for large datasets
- PIRLS mode is fastest for large image arrays
- Default mode suitable for small datasets and debugging

## Validation Criteria

### Successful Tests Should Show
1. **Convolution**: Proper IRF-decay convolution with expected peak shift
2. **Multi-exponential**: Clear separation of lifetime components
3. **Pattern Matching**: Low reconstruction error (MSE < 1e-3 typically)
4. **GPU Performance**: >100 pixels/second processing speed
5. **Coefficient Accuracy**: Recovered amplitudes within 10% of true values

## Customization

### Adding New Test Cases
```python
# In test_gpu_pattern_matching.py, modify:
test_cases = [
    {
        'name': 'Custom Model',
        'lifetimes': [your_lifetimes],
        'amplitudes': [your_amplitudes]
    }
]
```

### Adjusting Test Data Size
```python
# Modify in create_synthetic_flim_data():
nx, ny, n_channels = 64, 64, 512  # Larger dataset
max_counts = 2000                 # Higher signal
```

## Output Interpretation

### Good Results Indicators
- Smooth fitted curves overlaying noisy data
- Clear lifetime distribution peaks at expected values
- Low residuals (< ±3 standard deviations)
- Consistent performance across different modes

### Warning Signs
- High MSE values (> 1e-2)
- Poor fit quality in log-scale plots
- Large residuals or systematic patterns
- Significant performance degradation

This test suite provides comprehensive validation of the FLIM pattern matching pipeline and serves as a benchmark for GPU acceleration performance.