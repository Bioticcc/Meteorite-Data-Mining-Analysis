# NASA Meteorite Landings Data Mining Project

We have two sections to our project, the datamining and the UI section.
- `DM/`: data mining work 
- `UI/`: future interface/website work 
We will focus on the data mining work first, due to professor feedback, then move on to UI if we have time/desire at the end.

## GITHUB
CLONE THE REPO:
```bash
git clone <REPO_URL>
cd Meteorite-Data-Mining-Analysis
```
PULL REPO:
git checkout main
git pull origin main

PUSH REPO:
git status
git add .
git commit -m "bleh"
git push origin main

## Project Overview

The general idea we want is our pipeline that takes in the meteorite dataset from NASA, then using our 4 steps:
1. Data cleaning
2. Exploratory Data Analysis
3. Clustering 
4. Evaluating
We will prroduce accurate clustering data, groupings, figures, and a cleaned normalized dataset that we can then use for our website in UI to display a 2d map of earth with meteorite landing locations displayed as an overlay, with varying differences and filters based on the features of the dataset.

## Repository Structure (Summary)

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── report_notes.md
├── UI/
│   └── README.md
└── DM/
    ├── README.md
    ├── data/
    │   ├── raw/
    │   ├── processed/
    │   └── external/
    ├── notebooks/
    │   ├── 01_data_cleaning.ipynb
    │   ├── 02_eda.ipynb
    │   ├── 03_clustering.ipynb
    │   └── 04_evaluation.ipynb
    ├── src/
    │   ├── __init__.py
    │   ├── data_cleaning.py
    │   ├── eda.py
    │   ├── clustering.py
    │   └── evaluation.py
    ├── outputs/
    │   ├── figures/
    │   ├── tables/
    │   └── models/
    └── tests/
        └── test_preprocess.py
```

## Python Version

- Recommended: **Python 3.11**

If Python is not installed yet, install it first from [python.org](https://www.python.org/downloads/) and then verify installation.

## Setup (Step-by-Step)

Run all commands from the repository root.
(if python3 gives you an error, try using python instead.)

### 1. Check Python version

```bash
python3 --version
```

### 2. Create a local virtual environment (`.venv`)

```bash
python3 -m venv .venv
```

### 3. Activate the virtual environment

macOS/Linux:

```bash
source .venv/bin/activate
```

### 4. Upgrade `pip`

```bash
python3 -m pip install --upgrade pip
```

### 5. Install project dependencies

```bash
python3 -m pip install -r requirements.txt
```

### 6. Register the project Jupyter kernel (no, dont put in a user name I thought you had to do that as well but its just how the command looks >.>)

```bash
python3 -m ipykernel install --user --name meteorite-dm --display-name "Python (.venv) - Meteorite DM"
```

### 7. Launch JupyterLab

```bash
jupyter lab
```

## VS Code Setup (Short)

1. Install the VS Code extensions: **Python** and **Jupyter**.
2. Open this repository folder in VS Code.
3. Select interpreter: `.venv`.
4. Open notebooks in `DM/notebooks/` and choose the kernel **Python (.venv) - Meteorite DM**.

## Data and Workflow Notes

- Place raw source files in `DM/data/raw/`.
- Place cleaned/intermediate outputs in `DM/data/processed/`.
- Use `pathlib` (not hardcoded slashes) for file paths to keep code cross-platform.
- UI implementation is intentionally deferred until later milestones.

## Current Status

Initial scaffold is in place for a two-person team to start data mining work immediately.
