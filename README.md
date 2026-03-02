## **Predictive Analytics for Bank Loan Risk Assessment**

This repository contains the research and implementation of a machine learning pipeline designed to assess credit risk in highly imbalanced loan datasets. The project specifically addresses the **"Accuracy Paradox,"** where models achieve high nominal accuracy by failing to identify the small percentage of actual defaulters.

---

### **Project Overview**

Loan datasets are inherently skewed, with defaulters forming a very small portion of total borrowers. This project systematically evaluates strategies to move beyond simple accuracy, focusing on the trade-off between detecting financial risk (minority recall) and maintaining overall predictive performance.

#### **Core Objectives**

* 
**Mitigate Class Imbalance**: Evaluating cost-sensitive learning and resampling techniques.


* 
**Architectural Benchmarking**: Comparing performance across **LightGBM**, **CatBoost**, and **XGBoost**.


* 
**Interpretability**: Utilizing **SHAP** to ensure transparency in high-stakes financial decisions.



---

### **Dataset & Data Cleaning**

The project utilizes a financial dataset characterized by extreme class imbalance (defaulters < 10%) and complex, non-linear feature relationships.

#### **Cleaning and Preprocessing Pipeline**

* 
**Feature Engineering**: Extracted a **30-feature subset** using SHAP to prioritize variables with the highest predictive impact on loan default.


* 
**Categorical Encoding**: Utilized oblivious (symmetric) decision trees in CatBoost to handle sparse categorical data and reduce overfitting.


* **Resampling Strategies**:
* 
**SMOTE-Tomek**: A hybrid approach to generate minority instances while eliminating majority-class noise.


* 
**ADASYN**: Density-based oversampling that focuses on high-variance regions where classes overlap.


* 
**TabularGAN (TVAE)**: Generative synthesis of "defaulter" profiles to enrich the training set.





---

### **Model Performance & Results**

The experimental trajectory moved from high-recall, risk-averse models to precision-optimized architectures.

| Model | Accuracy | Strategy | Key Result |
| --- | --- | --- | --- |
| **LightGBM** | <br>**17.22%** 

 | Threshold Tuning ($P=0.04$) 

 | High Minority Recall (0.93) 

 |
| **CatBoost** | <br>**28.21%** 

 | Cost-Sensitive ($W=20$) 

 | Robust Default Detection (0.76 Recall) 

 |
| **XGBoost (Base)** | <br>**46.97%** 

 | ROC-AUC Optimized 

 | Identified 671 actual defaults 

 |
| **XGBoost (Opt)** | <br>**90.37%** 

 | Precision-Optimized ($W=1$) 

 | Mathematical peak of accuracy 

 |

---

### **Future Work**

* 
**Actionable Interpretability**: Integrating **DiCE** (Diverse Counterfactual Explanations) to provide rejected applicants with a path to approval.


* 
**Generative Imbalance Learning**: Implementing **CTGAN** to better capture interactions between categorical and continuous credit features.



---

### **How to Use This Repository**

#### **1. Installation**

```bash
pip install xgboost lightgbm catboost shap imbalanced-learn

```

#### **2. Running the Analysis**

* The `/sampling` directory contains implementations for SMOTE-Tomek, ADASYN, and TVAE.


* The `/models` directory includes the iterative tuning scripts for the boosting architectures.



**Course Coordinator:** Dr. Ravi Nahta **Group Members:** Darsh Patel, Ameya Naik, Yash Bhattad, Yatharth Shivhare **Institution:** Indian Institute of Information Technology, Vadodara 

