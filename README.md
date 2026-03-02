# Predictive Analytics for Bank Loan Risk Assessment

[cite_start]This repository contains the research and implementation of a machine learning pipeline designed to assess credit risk in highly imbalanced loan datasets[cite: 14]. [cite_start]The project specifically addresses the **"Accuracy Paradox,"** where models achieve high nominal accuracy by failing to identify the small percentage of actual defaulters[cite: 16, 17].

---

## 📊 Project Overview
[cite_start]Loan datasets are inherently skewed, with defaulters forming a very small portion of total borrowers[cite: 15, 31]. [cite_start]This project systematically evaluates strategies to move beyond simple accuracy, focusing on the trade-off between detecting financial risk (minority recall) and maintaining overall predictive performance[cite: 19, 20].

### Core Objectives
* [cite_start]**Mitigate Class Imbalance**: Evaluating cost-sensitive learning and resampling techniques[cite: 18, 50].
* [cite_start]**Architectural Benchmarking**: Comparing performance across **LightGBM**, **CatBoost**, and **XGBoost**[cite: 18, 47].
* [cite_start]**Interpretability**: Utilizing **SHAP** (SHapley Additive exPlanations) to ensure transparency in high-stakes financial decisions[cite: 22, 53].

---

## 🧹 Dataset & Data Cleaning
[cite_start]The project utilizes a financial dataset characterized by extreme class imbalance and complex, non-linear feature relationships[cite: 29, 31].



### Cleaning and Preprocessing Pipeline
* [cite_start]**Feature Engineering**: Extracted a **30-feature subset** using SHAP to prioritize variables with the highest predictive impact on loan default[cite: 96].
* [cite_start]**Categorical Encoding**: Utilized oblivious (symmetric) decision trees in CatBoost to handle sparse categorical data and reduce overfitting[cite: 111, 113].
* **Resampling Strategies**:
    * [cite_start]**SMOTE-Tomek**: A hybrid approach to generate minority instances while eliminating proximal majority-class noise[cite: 69, 70].
    * [cite_start]**ADASYN**: Density-based oversampling that focuses on high-variance regions where classes overlap[cite: 75].
    * [cite_start]**TabularGAN (TVAE)**: Generative synthesis of additional "defaulter" profiles in feature space to improve representation[cite: 79].

---

## 📈 Model Performance & Results
[cite_start]The experimental trajectory moved from high-recall, risk-averse models to precision-optimized architectures[cite: 148].



| Model | Accuracy | Strategy | Key Result |
| :--- | :--- | :--- | :--- |
| **LightGBM** | **17.22%** | Threshold Tuning ($P=0.04$) | [cite_start]High Minority Recall (0.93) [cite: 97, 106] |
| **CatBoost** | **28.21%** | Cost-Sensitive ($W=20$) | [cite_start]0.76 Recall for Defaulters [cite: 109, 116] |
| **XGBoost (Base)**| **46.97%** | ROC-AUC Optimized | [cite_start]Identified 671 actual defaults [cite: 120, 131] |
| **XGBoost (Opt)** | **90.37%** | Precision-Optimized ($W=1$) | [cite_start]Mathematical peak of accuracy [cite: 134, 145] |

---

## 🚀 Future Work
* [cite_start]**Actionable Interpretability**: Integrating **DiCE** (Diverse Counterfactual Explanations) to provide rejected applicants with a path to approval[cite: 172, 176].
* [cite_start]**Generative Imbalance Learning**: Implementing **CTGAN** to better capture interactions between categorical and continuous credit features[cite: 181, 183].

---

## 🛠️ Project Structure & Usage

### Directory Layout
* [cite_start]**/sampling**: Contains implementations for **SMOTE-Tomek**, **ADASYN**, and **TVAE**[cite: 18, 182].
* [cite_start]**/models**: Includes the iterative tuning scripts for the **LightGBM**, **CatBoost**, and **XGBoost** architectures[cite: 18, 93].

### Installation
```bash
pip install xgboost lightgbm catboost shap imbalanced-learn