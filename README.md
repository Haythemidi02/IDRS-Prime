# AI-Based Intrusion Detection and Automated Response System

This repository contains a complete research-oriented pipeline for building an Intrusion Detection and Response System (IDRS) for educational web platforms. The project combines network-flow analysis, classical machine learning, deep learning, anomaly detection, LLM-assisted web-threat analysis, and automated response generation into one staged workflow.

The notebooks are designed to be executed sequentially. Each part produces intermediate datasets, trained models, metrics, visualizations, or registries that are reused by later stages.

## Project Scope

The system focuses on detecting and responding to common threats against web and learning platforms, including denial-of-service behavior, brute-force attempts, reconnaissance, SQL injection, cross-site scripting, command injection, suspicious traffic patterns, and zero-day-like anomalies.

It uses a layered detection strategy:

- Exploratory data analysis and feature engineering for network traffic.
- Classical ML baselines with tuning and explainability.
- Deep learning classifiers for traffic classification.
- Anomaly detectors for unseen or drifting attack patterns.
- Transformer-based payload classification for web threats.
- Automated response playbooks, firewall rule generation, and incident reporting.
- A final ensemble pipeline that combines model outputs and evaluates resilience.

## Repository Structure

```text
.
|-- dataset.py
|-- IDRS_Part1_Setup_EDA.ipynb
|-- IDRS_Part2_Preprocessing_ClassicalML.ipynb
|-- IDRS_Part3_DeepLearning.ipynb
|-- IDRS_Part4_AnomalyDetection.ipynb
|-- IDRS_Part5_LLM_WebThreats.ipynb
|-- IDRS_Part6_ResponseEngine.ipynb
`-- IDRS_Part7_FinalPipeline.ipynb
```

## Notebook Workflow

### Part 1: Setup, Dataset Loading, and EDA

Initializes the environment, loads real or synthetic intrusion-detection data, audits data quality, explores class distributions, analyzes feature relationships, and creates engineered security features.

### Part 2: Preprocessing and Classical ML

Builds the feature matrix, performs stratified train/validation/test splits, applies scaling and imbalance handling, trains classical models such as Random Forest, XGBoost, and LightGBM, tunes hyperparameters with Optuna, and adds SHAP-based explainability.

### Part 3: Deep Learning Detection

Implements PyTorch-based deep learning models, including 1D-CNN, BiLSTM, and hybrid CNN-LSTM architectures. This stage compares deep models against the classical baselines and exports trained artifacts.

### Part 4: Anomaly Detection

Adds zero-day and unknown-threat detection through a deep autoencoder, Isolation Forest, One-Class SVM, ensemble anomaly scoring, concept-drift simulation, and online learning experiments.

### Part 5: LLM-Powered Web Threat Analysis

Constructs a curated web payload dataset and fine-tunes a DistilBERT classifier for SQL injection, XSS, CSRF, command injection, and benign payloads. It also includes adversarial payload testing, alert generation, and ONNX export.

### Part 6: Automated Response Engine

Transforms detection signals into practical response actions. This includes threat scoring, IP reputation tracking, response playbooks, iptables/nftables rule generation, incident report creation, latency benchmarking, and response analytics.

### Part 7: Final Ensemble Pipeline

Loads all trained components, combines their outputs in an ensemble inference engine, benchmarks final performance, tests adversarial robustness, simulates concept drift, analyzes false positives, and serializes the full pipeline.

## Dataset Utilities

`dataset.py` provides helper utilities for working with intrusion-detection datasets hosted on Hugging Face. It supports:

- Dataset metadata inspection.
- Downloading selected subsets and files.
- Merging network-flow, packet-field, packet-byte, and payload-byte features.
- Converting packet or payload hex data into numeric byte columns.
- Loading parquet-backed datasets through the Hugging Face `datasets` API.

The helper currently targets datasets such as `UNSW-NB15` and `CIC-IDS2017`.

## Main Technologies

- Python
- Jupyter Notebook
- pandas, NumPy, SciPy
- scikit-learn, imbalanced-learn
- XGBoost, LightGBM, Optuna
- SHAP
- PyTorch
- Hugging Face Transformers and Datasets
- River for online learning and drift handling
- Matplotlib, Seaborn, Plotly
- ONNX Runtime

## Getting Started

Create and activate a Python environment, then install the main dependencies used by the notebooks:

```bash
pip install pandas numpy scipy scikit-learn imbalanced-learn xgboost lightgbm optuna shap torch transformers datasets evaluate river matplotlib seaborn plotly joblib tqdm huggingface-hub onnxruntime
```

Then open the notebooks in order:

```bash
jupyter notebook
```

Recommended execution order:

1. `IDRS_Part1_Setup_EDA.ipynb`
2. `IDRS_Part2_Preprocessing_ClassicalML.ipynb`
3. `IDRS_Part3_DeepLearning.ipynb`
4. `IDRS_Part4_AnomalyDetection.ipynb`
5. `IDRS_Part5_LLM_WebThreats.ipynb`
6. `IDRS_Part6_ResponseEngine.ipynb`
7. `IDRS_Part7_FinalPipeline.ipynb`

## Outputs

During execution, the notebooks may generate processed datasets, model files, plots, reports, registries, serialized pipelines, and firewall-response artifacts. These outputs are generally produced by earlier stages and consumed by later stages, so the pipeline should be run in order for the final notebook to work correctly.

## Notes

This project is intended for research, experimentation, and educational cybersecurity workflows. Response actions such as firewall rules should be reviewed and tested in a controlled environment before being applied to production systems.
