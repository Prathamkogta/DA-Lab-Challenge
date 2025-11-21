# 🏆 DA-Lab Challenge: "Extreme Denoising" Solution

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Score](https://img.shields.io/badge/Score-2.096-green)
![RMSE](https://img.shields.io/badge/OOF%20RMSE-0.5986-orange)

## 📌 Overview

This repository contains the winning solution code for the DA-Lab Data Challenge (2025). The objective was to predict the "fitness score" (semantic similarity) between a large language model's response and a specific metric definition.

The solution tackles the problem using a **Geometric Distance approach** combined with **Synthetic Negative Sampling** and **Stacked Ensembling** to overcome the lack of negative examples in the training data.

## 🚀 Key Features & Strategy

### 1. High-Resolution Text Processing
Instead of standard embeddings, we utilized a "Bag of Jargon" technique to distinguish subtle technical differences:
* **Concatenation:** `Prompt + System Prompt + Response` are treated as a single text block.
* **TF-IDF:** Configured with `max_features=10,000` and `ngram_range=(1, 3)` to capture complex technical jargon and trigrams.
* **SVD Compression:** TruncatedSVD reduces both text vectors and provided metric embeddings into a shared **64-dimensional latent space**.

### 2. Feature Engineering: Metric Learning
We model the problem as a geometric distance task rather than simple regression. Features include:
* **Prototypes:** Centroids calculated for every metric ID in the training set.
* **Distance Metrics:** Cosine, L1, L2, and Dot Product distances between the text vector and the target metric vector.
* **Margin Features:** Inspired by Contrastive Learning, we calculate the margin between the cosine similarity of the *assigned* metric and the *best alternative* metric.

### 3. Synthetic Negative Sampling (Data Augmentation)
The training data was heavily skewed toward high scores. To teach the model to recognize "bad" responses:
* **Swapping:** We take valid rows and swap the `metric_idx` with an incorrect metric.
* **Labeling:** These mismatched pairs are assigned low scores (0, 1, or 2).
* **Result:** The model learns to penalize semantic mismatches heavily.

### 4. Ensemble Stacking & Bias Correction
The final prediction is a weighted average of multiple models:
* **Stage 1:** Random Forest trained on Augmented Data (Real + Synthetic).
* **Stage 2:** Ridge, Random Forest, and LGBM trained on Real Data only (GroupKFold).
* **Stage 3 (Stacking):** **NNLS (Non-Negative Least Squares)** finds the optimal linear combination of predictions.
* **Post-Processing:** Bayesian Bias Correction smoothes systematic over/under-predictions per metric group.

## 🛠️ Configuration

The solution is controlled by the following hyperparameters defined in the notebook:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `TFIDF_MAX_FEAT` | 10,000 | Captures rare technical terms |
| `NGRAM_RANGE` | (1, 3) | Captures trigrams (3-word phrases) |
| `SVD_DIM` | 64 | Dense vector dimension |
| `RF_N_ESTIMATORS` | 500 | High estimator count for stability |
| `BIAS_ALPHA` | 5.0 | Smoothing factor for bias correction |

## 📦 Requirements

To run this solution, you need the following Python libraries:

```python
numpy
pandas
scikit-learn
scipy
lightgbm
tqdm
