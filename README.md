# Heavy Equipment Selling Price Prediction
**IIT Madras MLP Course Project**

This repository contains the end-to-end machine learning pipeline developed for the Machine Learning Practice (MLP) course by Jha Sumitkumar Gangadhar within the BS in Data Science and Applications program at IIT Madras. 

## Project Overview
The primary objective of this project is to build a robust predictive regression model for the private [Heavy Equipment Selling Price Prediction Challenge](https://www.kaggle.com/competitions/heavy-equipment-selling-price-prediction-challenge) on Kaggle. The pipeline handles complex data preprocessing, specialized feature engineering, and hyperparameter tuning to accurately forecast auction prices for heavy machinery based on historical sales data and equipment specifications.

## Key Achievements
- **Algorithm:** Engineered a highly optimized LightGBM Regressor tailored for tabular regression.
- **Validation Strategy:** Implemented a rigorous K-Fold Cross-Validation strategy to prevent overfitting and ensure model generalizability.
- **Performance:** Achieved a highly competitive Out-Of-Fold (OOF) RMSE score of **0.19281**.

## Tech Stack
- **Language:** Python
- **Core Libraries:** Pandas, NumPy, Scikit-Learn, LightGBM
- **Visualization:** Matplotlib, Seaborn

## Repository Structure
- `train.csv` — The primary training dataset (private competition data).
- `23f2004586-notebook-2026t2.ipynb` — The core Jupyter Notebook containing the complete Exploratory Data Analysis (EDA), model training, evaluation metrics, and inference steps.

## How to Run Locally

**1. Clone the repository and Install dependencies
```bash 
git clone https://github.com/23f2004586/iitm-mlp-project.git
pip install pandas numpy scikit-learn lightgbm matplotlib seaborn
