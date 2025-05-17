# Project Installation Guide

This README explains how to set up a Python virtual environment and install all required dependencies on **Windows** or **Linux**.

---

## 1. Prerequisites

* **Python** 3.7 or higher installed on your system.
* **pip** (comes with recent Python installs).
* (Optional) **git** if you need to clone the repository.

---

## 2. Create a Virtual Environment

### Linux / macOS

```bash
# Navigate to project root
git clone <repo-url>  # if needed
cd <project-folder>

# Create venv folder named 'venv'
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

### Windows (PowerShell)

```powershell
# Navigate to project root
git clone <repo-url>  # if needed
cd <project-folder>

# Create venv folder named 'venv'
python -m venv venv

# Activate the virtual environment
override
.
venv\Scripts\Activate.ps1
# Or for cmd.exe:
venv\Scripts\activate.bat
```

After activation, your prompt should prefixed by `(venv)`.

---

## 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:

* **PyTorch & Torchvision** for deep learning models
* **OpenCV (cv2)** and **Pillow** for image I/O
* **NumPy**, **Pandas**, **Matplotlib** for data handling & plotting
* **Scikit-learn**, **SciPy** for metrics and numerical routines
* **gdown** for downloading data from Google Drive

---

## 4. Verify Installation

In a Python shell or script, try:

```python
import torch
import torchvision
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gdown
from sklearn.metrics import precision_score
from scipy.spatial.distance import directed_hausdorff
from PIL import Image
```

If no errors occur, you are ready to go!

---

