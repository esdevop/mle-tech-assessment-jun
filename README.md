# Junior Machine Learning Engineer Technical Assessment

## Overview

Welcome to the Junior MLE Technical Assessment! This assignment evaluates your ability to implement core machine learning engineering concepts, specifically **adstock transformation** used in media mix modeling. You will work with real time-series data and implement an exponential decay function that models the carryover effects of advertising.

## Background: Adstock in Media Mix Modeling

**Adstock** (or advertising carryover) is a critical concept in marketing analytics that captures how advertising effects persist over time. When a TV advertisement airs, its impact doesn't disappear immediately—it decays exponentially over subsequent periods.

The mathematical formula for adstock transformation is:
```math
x'_t = x_t + x_{t-1}\cdot \lambda
```
Where:
- $x_t$ is the media exposure (non-transformed) at time t
- The decay factor $\lambda = \exp(\ln(0.5) / \tau)$ ensures that after $\tau$ periods, the carryover effect is reduced by 50%
- $\tau$ is a half-life parameter

## Assignment Objectives

By completing this assignment, you will demonstrate:
- **Algorithm Implementation**: Translate mathematical concepts into efficient code
- **NumPy Proficiency**: Work with typed arrays and numerical computations
- **Testing & Validation**: Use pytest to verify implementation correctness
- **ML Engineering Workflow**: Follow professional development practices

## Setup Instructions

### Option 1: Local Development (Recommended)

#### Prerequisites
- Git with Git LFS support
- Python 3.11+
- uv (recommended) or pip

> **💡 Windows Users**: For the best experience, we recommend using **WSL2 with Ubuntu** instead of native Windows. WSL2 provides a Linux environment that ensures compatibility with all tools and commands. [Install WSL2](https://docs.microsoft.com/en-us/windows/wsl/install) and use Ubuntu 20.04 or later.

#### Setup Steps
```bash
# Install Git LFS if not already installed
# On Ubuntu/Debian:
sudo apt install git-lfs
# On macOS:
brew install git-lfs
# On Windows: Download from https://git-lfs.github.io/

# Initialize Git LFS (one-time setup per machine)
git lfs install

# Clone the repository with LFS files
git clone https://github.com/esdevop/mle-tech-assessment-jun.git
cd mle-tech-assessment-jun

# Ensure LFS files are pulled (data files)
git lfs pull

# Set up Python environment using uv (preferred)
uv sync --group dev

# Alternative: using pip
pip install -e ".[dev]"

# Verify data files are present
ls -la data/raw/ data/processed/
```

### Option 2: Google Colab

```python
# In a Colab cell - Install and setup Git LFS
!curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
!sudo apt-get install git-lfs

# Initialize Git LFS
!git lfs install

# Clone the repository with LFS support
!git clone https://github.com/esdevop/mle-tech-assessment-jun.git
%cd mle-tech-assessment-jun

# Pull LFS files (data files)
!git lfs pull

# Install Python dependencies
!pip install -e ".[dev]"

# Verify data files are downloaded
!ls -la data/raw/ data/processed/
!head -5 data/raw/raw_data.csv
```

#### Troubleshooting Git LFS in Colab
If you encounter LFS issues in Colab, you can manually download the data files:
```python
# Alternative: Direct download if LFS fails
import requests
import os

# Create directories
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Download data files directly (replace with actual raw file URLs if needed)
# Note: This is a fallback - LFS method above is preferred
print("If LFS setup fails, contact the assessment administrator for direct data file access")
```

### Verification

After setup, verify everything is working correctly:

```bash
# Check that data files exist and have content
head -5 data/raw/raw_data.csv
head -5 data/processed/processed_data.csv

# Run a quick test to ensure environment is ready
python -c "import numpy as np; import pandas as pd; import pytest; print('✅ All dependencies available')"

# Test that the function can be imported
python -c "from app.utils.transformations import _apply_halflife; print('✅ Function import successful')"
```

**Expected output for data files:**
- `data/raw/raw_data.csv` should show: `date_week,tv_ad_executions`
- `data/processed/processed_data.csv` should show: `date_week,tv_ad_executions_adstock`

If data files show Git LFS pointer content instead of actual data, run:
```bash
git lfs pull
```

## Task Description

### Your Mission
Implement the **exponential decay logic** in the `_apply_halflife` function located in `app/utils/transformations.py`.

### What's Provided
- ✅ **Function signature** with proper type hints
- ✅ **Input data**: `data/raw/raw_data.csv` containing TV ad execution data
- ✅ **Expected output**: `data/processed/processed_data.csv` with correct adstock values
- ✅ **Comprehensive test suite**: 19 tests covering edge cases and validation
- ✅ **Documentation**: Clear docstring explaining the function

### What You Need to Implement

**File**: `app/utils/transformations.py`  
**Function**: `_apply_halflife`  
**Target**: Replace the comment on lines 20-23 with your implementation

```python
def _apply_halflife(series: NDArray[np.float64], halflife: np.float32, rounding: int=4) -> NDArray[np.float32]:
    """
    Apply exponential decay to a pandas Series based on the given half-life.

    Parameters:
    - series: NDArray[np.float64] - The input time series data
    - halflife: np.float64 - The half-life period for exponential decay
    - rounding: int - Number of decimal places for rounding (default: 4)

    Returns:
    - NDArray[np.float32] - The transformed series with exponential decay applied
    """
    adstocked_series = series.copy()
    
    # TODO: Enter your calculation here
    # Implement the exponential decay logic
    # Hint: Use the formula mentioned in the background section
    
    return np.round(adstocked_series, rounding).astype(np.float32)
```

### Implementation Requirements
1. **Calculate the decay factor** using the halflife parameter
2. **Iterate through the time series** starting from index 1
3. **Apply the adstock formula** to accumulate carryover effects
4. **Return properly typed results** (np.float32 with specified rounding)

### Key Considerations
- The first element of the series should remain unchanged
- Each subsequent element should include both its original value and the decayed carryover from previous periods
- Use efficient NumPy operations for performance
- Maintain numerical precision according to the rounding parameter

## Validation Process

### Step 1: Run the Test Suite
Execute the following command to validate your implementation:

```bash
pytest tests/test_transformations.py -v
```

### Expected Output
If your implementation is correct, you should see:
```
==================== test session starts =====================
collected 19 items

tests/test_transformations.py::TestApplyHalflife::test_basic_functionality PASSED
tests/test_transformations.py::TestApplyHalflife::test_zero_series PASSED
tests/test_transformations.py::TestApplyHalflife::test_single_element PASSED
# ... (all 19 tests should pass)
===================== 19 passed in 0.26s ======================
```

### Step 2: Check Code Coverage (Bonus)
For extra credit, run with coverage reporting:
```bash
pytest tests/test_transformations.py --cov=app.utils.transformations --cov-report=term-missing -v
```

### Step 3: Verify Real Data Transformation
The test suite includes a critical test (`test_actual_data_transformation`) that:
- Loads your raw input data from `data/raw/raw_data.csv`
- Applies your transformation with `halflife=2.5`
- Compares results against expected output in `data/processed/processed_data.csv`

This test ensures your implementation works correctly with real-world data!

## Success Criteria

### ✅ Minimum Requirements (Must Have)
- [ ] All 19 tests pass successfully
- [ ] Implementation handles the mathematical formula correctly
- [ ] Code runs without errors in the chosen environment
- [ ] Function returns properly typed NumPy arrays

### 🌟 Excellence Indicators (Nice to Have)
- [ ] Clean, readable code with logical variable names
- [ ] Efficient NumPy operations (avoiding unnecessary loops where possible)
- [ ] Understanding of the business context (adstock in media mix modeling)
- [ ] Successful execution of coverage reporting
- [ ] Proper handling of edge cases (demonstrated by passing all tests)

## Evaluation Criteria

| Criteria | Weight | Description |
|----------|---------|-------------|
| **Technical Implementation** | 60% | Correctness of algorithm, NumPy proficiency, type handling |
| **Testing & Validation** | 25% | Successful test execution, understanding of results |
| **Setup & Workflow** | 15% | Environment setup, tool usage, debugging ability |

## Common Pitfalls to Avoid

⚠️ **Don't modify the function signature** - Type hints and parameters are part of the assessment  
⚠️ **Don't change the test files** - Tests validate your implementation correctness  
⚠️ **Don't forget the copy()** - Avoid modifying the input series in-place  
⚠️ **Watch the data types** - Input is float64, output should be float32  

## Getting Help

If you encounter issues:
1. **Read the error messages carefully** - They often point to the exact problem
2. **Check the test failures** - Failed tests show what's expected vs. what you produced  
3. **Review the docstring** - All requirements are documented in the function
4. **Examine the example usage** - The `__main__` block shows how the function should work

## Time Expectation

This assessment should take **30-60 minutes** for a junior MLE, including:
- Environment setup: 10-15 minutes
- Implementation: 20-30 minutes  
- Testing and validation: 10-15 minutes

## Submission

Once you have successfully implemented the function and all tests pass:
1. **Document your approach** (optional but recommended)
2. **Share your implementation** or discuss your solution
3. **Be prepared to explain** your understanding of adstock and the algorithm

---

**Good luck!** This assessment reflects real-world MLE tasks where you'll implement mathematical transformations, work with time-series data, and validate your code through comprehensive testing. 🚀
