## Project Setup with Poetry

This guide explains how to create and install the Python environment for this project using Poetry on **Linux** and **Windows**.

### Core Dependencies

* **Deep Learning Framework**: `torch`, `torchvision`
* **Image Processing**: `opencv-python`, `Pillow`
* **Data Science**: `numpy`, `pandas`, `matplotlib`
* **Machine Learning**: `scikit-learn`, `scipy`
* **Utilities**: `gdown`

---

## Prerequisites

* **Python** >= 3.8 installed and on your PATH.
* **Git** (optional, for cloning repository).
* **Poetry** installed:

  * Linux: `curl -sSL https://install.python-poetry.org | python3 -`
* **Add Poetry to your PATH** as instructed by the installer.

---

## Installation

### Linux

1. **Clone the repository** (if applicable):

   ```bash
   git clone https://github.com/EdyStan/Unsupervised-Domain-Adaptation-in-Medical-Imaging.git
   cd your_repo
   ```

2. **Install dependencies**:

   ```bash
   poetry install
   ```

<!-- ### Windows (PowerShell)

1. **Clone the repository** (if applicable):

   ```powershell
   git clone https://github.com/EdyStan/Unsupervised-Domain-Adaptation-in-Medical-Imaging.git
   cd your_repo
   ```

2. **Install dependencies**:

   ```powershell
   poetry install
   ```

3. **Activate the virtual environment**:

   ```powershell
   poetry shell
   ``` -->

---

## Usage

Once the environment is active, you can run your scripts as usual:

```bash
poetry run python train_UDA.py 
```

---
