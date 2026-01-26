# RE-sLDA: Resampling-Enhanced Sparse LDA for Ordinal Outcomes
This repository contains the implementation of RE-sLDA, a framework designed to enhance feature selection stability and accuracy when dealing with high-dimensional data and ordinal outcomes. By integrating resampling techniques with Sparse Linear Discriminant Analysis (sLDA), this method identifies robust biomarker signatures that standard sparse models often miss due to selection instability.

## **Key Features**
- **Ordinal Outcome Optimization:** Specifically tuned for categorical outcomes with a natural ordering (e.g., disease severity, treatment response grades).

- **Resampling-Based Stability:** Utilizes bootstrap-based resampling to calculate Variable Inclusion Probabilities (VIP), ensuring the selected features are not artifacts of a single data split.

- **Parallel Computing Support:** Fully integrated with doParallel and foreach for high-performance execution on multi-core clusters.

---

## **Basic Usage**

### Install Dependences
Clone the repository to your local machine, then install the required Python packages:
```bash
pip install -r requirements.txt
```
**Note:** A virtual environment is recommended to avoid dependency conflicts.

---

### Running the Pipeline
Run the **bootstrapping** script with this command:
```bash
python pipeline.py bootstrapping
```

And **subsampling** with this: 
```bash
python pipeline.py subsampling
```
Each command will execute the corresponding pipeline using the configuration parameters defined in `pipeline.py`.

---

### Parameter Configuration
You can customize the behavior of each pipeline for your own purposes. 

1. Open the `pipeline.py` file
2. Navigate to the `main()` function 
3. Locate the configuration blocks under the `Bootstrapping` and `Subsampling` comment headers
4. Modify parameters as needed

This allows you to adapt the framework to different datasets and experimental settings.

---

### Setup Dataset
There is a sample dataset already provided within this repository, To run the framework on your own data, upload both the **feature matrix (X)** and **response vector (Y)** to the `datasets/` directory

#### **Dataset Requirements:**
- Files must be in `.csv` format
- **Y (response) dataset**
    - Singlular column
    - Contains **ordinal labels** only
- **X (feature) dataset**
    - Rows represent samples
    - Columns represent features
    - First row must contain feature (variable) names
> **Important:** The number of rows in X must match the number of entries in Y.

---

## **Output**

After running either the **bootstrapping** or **subsampling** pipeline, the framework generates one CSV output files in the designated output directory. 

- All output files are saved in the `output/` directory
- The **filename prefix** can be modified in the pipeline parameters

---

### Output File Contents

Each output CSV contains a single row per run with the following fields:

Column Name	| Description |
|-----------|-------------|
Selected_Variables	| Comma-separated list of predictor variables selected by the model
optimal_lambda	| Regularization parameter selected via cross-validation
MAE	Mean Absolute | Error computed on held-out data
Accuracy | Classification accuracy of the model

These values summarize both **feature selection behavior** and **predictive performance** for a given pipeline execution.

---

### Output Naming

Output files are timestamped to ensure reproducibility and prevent overwriting previous results. The filename format is:
```
<prefix>_<pipeline>_<mmddHHMM>.csv
```
Example:
```
BS_Glios_group_bootstrapping_01251445.csv
```

---

### Notes on Interpretation

**Note:** The `Selected_Variables` column reflects the final set of features chosen for that run and may vary across executions due to resampling and randomness.

**Important:** Performance metrics are computed on held-out data and may vary depending on the random seed and parameter configuration.

---

## Reference

> Clemmensen, L., Hastie, T., Witten, D., & Ersbøll, B. (2011).  
> Sparse Discriminant Analysis. Technometrics.

---
