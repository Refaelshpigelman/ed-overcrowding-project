# ED Overcrowding Prediction - Prototype

## Overview
This repository contains a working prototype for predicting Emergency Department (ED) overcrowding patterns using historical ED operational data.

The prototype demonstrates:
- data loading from the provided CSV file
- preprocessing and cleaning
- translation of department names into English labels
- conversion of status values into numeric labels
- feature engineering from timestamps
- baseline machine learning modeling and evaluation

## Repository Structure
```text
ed_overcrowding_prototype/
├── data/
│   └── ED_full_data_2.csv
├── notebooks/
│   └── ed_prototype.ipynb
├── src/
│   ├── preprocessing.py
│   └── model_pipeline.py
├── requirements.txt
└── README.md
```

## How to Run
1. Clone the repository:
   ```bash
   git clone <YOUR_GITHUB_LINK>
   cd ed_overcrowding_prototype
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   ```bash
   jupyter notebook notebooks/ed_prototype.ipynb
   ```

## Data Transformations
- Department names are mapped from Hebrew labels to English labels using a deterministic mapping dictionary.
- Status values are encoded as numeric labels.
- Timestamps are converted to datetime and expanded into time-based features.

## Reproducibility
All preprocessing steps and mappings are deterministic and documented to ensure consistent results across future datasets.

## GitHub Link
https://github.com/Refaelshpigelman/ed-overcrowding-project``
