# PIPO-LS-SVR
Code for paper: "Least Squares Support Vector Regression with Probabilistic Linguistic Information"

# PIPO-LS-SVR for Probabilistic Linguistic Term Sets (PLTSs)

This repository contains the MATLAB implementation of the **PIPO-LS-SVR (PLTS Input-PLTS Output Least Squares Support Vector Regression)** model.

The code supports regression tasks where both inputs and outputs are characterized by **Probabilistic Linguistic Term Sets (PLTSs)**, preserving the complete probabilistic distribution structure of linguistic information.

## 📂 Repository Contents

The repository includes two main implementation folders/scripts:

1.  **`Linear_PIPO_LS_SVR_for_PLTSs`**:
    *   Implementation of the **Linear PIPO-LS-SVR** model.
    *   Features: Solved via **Quadratic Programming (QP)** to ensure non-negative constraints on regression coefficients.
    *   Includes data preprocessing (indicator normalization) and model training.

2.  **`Nonlinear_PIPO_LS_SVR_for_PLTSs`**:
    *   Implementation of the **Two-Stage Nonlinear PIPO-LS-SVR** model.
    *   **Stage 1**: Kernel-based regression (Quadratic / RBF kernels) to capture nonlinear mappings.
    *   **Stage 2**: Probability calibration using **Temperature Scaling** to adaptively regulate semantic hesitation.

## 🚀 Prerequisites

*   **MATLAB** (Recommended version: R2020b or later)
*   **Optimization Toolbox** (Required for `quadprog` in the linear model)
*   **Statistics and Machine Learning Toolbox**

## 🔧 How to Run

Navigate to the `Linear_PIPO_LS_SVR_for_PLTSs` or 'Nonlinear_PIPO_LS_SVR_for_PLTSs' folder and run the main script (e.g., `main.m` or `demo.m`).

This code is designed to process PLTS data. Please ensure that the input data is properly formatted as standardized probabilistic linguistic term sets, with the sum of probabilities equaling 1.  
The package includes sample datasets in the form of PLTS, derived from Hotel Reviews and Air Quality Evaluation of cities.
