# 🏠 California Housing Price Prediction

This project predicts **median house values** in California using the **1990 California Census dataset**.  
It implements a full **Machine Learning pipeline** from Exploratory Data Analysis (EDA), outlier handling, data preprocessing, model training, to model evaluation.  
The final model is a **Stacking Regressor** with **Gradient Boosting** as the meta-model, which achieved the best performance across RMSE, MAE, and MAPE metrics.



📄 View the Jupyter Notebook online (no download needed):
[Open in nbviewer](https://nbviewer.org/github/rahmaskp/California-Housing-Price-StackingRegression-Model/blob/main/California%20Housing%20Price_Regression%20Model.ipynb)

---

## 📊 Dataset

- **Source:** California Housing dataset (1990 US Census)
- **Rows:** ~14,400
- **Features:**  
  - **Numeric features:** `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`  
  - **Categorical feature:** `ocean_proximity`  
  - **Target:** `median_house_value`  

---

## ⚙️ Workflow

1. **Exploratory Data Analysis (EDA)**
   - Visualize feature distributions
   - Relationship between features and target variable
   - Remove outliers using IQR method

2. **Data Preprocessing**
   - Handle missing values (KNNImputer for numerical features)
   - Encode categorical features using OneHotEncoder
   - Scale numeric features with RobustScaler

3. **Model Training**
   - Baseline models:
     - Linear Regression
     - Ridge Regression
     - Random Forest Regressor
   - Model evaluation using RMSE, MAE, MAPE

4. **Final Model: Stacking Regressor**
   - Base learners: Linear Regression, Ridge Regression, Random Forest Regressor
   - Meta-model: Gradient Boosting Regressor

5. **Model Evaluation**
   - RMSE: *61,700*
   - MAE: *43,447*
   - MAPE: *26.22%*

6. **Model Saving**
   - Full preprocessing + model pipeline saved using `joblib`

---

## 📈 Results Summary

- **Best model:** Stacking Regressor (Gradient Boosting meta-model)
- Achieved the lowest RMSE, MAE, and MAPE compared to all baseline models
- Balanced performance across both train and test sets, showing good generalization

---

## 🚀 Deployment

The model is deployed on **Hugging Face Spaces** with a Streamlit interface:  
🔗 [California Housing Price Prediction App](https://huggingface.co/spaces/rahmaskp/california-housing-price)

You can use the web app to input property and location details in California and get the predicted median house value instantly.

---

## 📌 Recommendations for Future Work

- Experiment with more diverse base learners (Extra Trees, XGBoost, LightGBM).
- Integrate external socio-economic or geographic datasets to enrich features and possibly boost predictive power. 
- Monitor deployed model for performance drift over time.

---

### Author

rahmaskp
