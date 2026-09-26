# Customer Churn Analytics & Prediction

A machine learning project that analyzes customer churn patterns and predicts whether a customer is likely to churn based on their demographic, service, contract, tenure, and billing information.

The project combines **Python, Pandas, SQL, Scikit-learn, and Streamlit** to build an end-to-end data analytics and machine learning application.

## Project Overview

Customer churn is a major business problem where customers stop using a company's services.

This project analyzes historical customer data to:

- Identify patterns associated with customer churn
- Perform exploratory data analysis (EDA)
- Analyze customer data using SQL
- Train machine learning models to predict churn
- Compare Logistic Regression and Random Forest
- Evaluate model performance using multiple metrics
- Provide an interactive Streamlit dashboard for predictions

## Tech Stack

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **SQLite**
- **Scikit-learn**
- **Streamlit**
- **Joblib**
- **Git & GitHub**

## Dataset

The project uses the IBM Telco Customer Churn dataset.

The dataset contains customer information such as:

- Customer demographics
- Tenure
- Contract type
- Internet service
- Technical support
- Payment method
- Monthly charges
- Total charges
- Churn status

The dataset contains 7,043 original records. After converting `TotalCharges` to numeric values and removing rows with missing values, 7,032 records were used for analysis and modeling.

The dataset is not included in this repository because it is excluded through `.gitignore`.

## Project Workflow

```text
Customer Dataset
       ↓
Data Cleaning
       ↓
Exploratory Data Analysis
       ↓
SQL Analysis
       ↓
Feature Engineering & Encoding
       ↓
Train/Test Split
       ↓
Machine Learning
       ↓
Model Evaluation
       ↓
Streamlit Dashboard
       ↓
Customer Churn Prediction

Exploratory Data Analysis

Several churn patterns were analyzed.

Churn Rate by Contract

Observed churn rates:

Contract Type	Churn Rate
Month-to-month	42.71%
One year	    11.28%
Two year	    2.85%

Month-to-month customers had the highest observed churn rate in the dataset.

Churn Rate by Tenure

The analysis showed that newer customers generally had higher observed churn rates than customers with longer tenure.

Monthly Charges

Customers who churned had an average monthly charge of approximately $74.44, compared with approximately $61.31 for customers who stayed.

These relationships represent associations observed in the dataset and do not establish causation.

SQL Analysis

The cleaned dataset was loaded into a SQLite database.

Example SQL analysis included:

Customer count by contract type
Churned customer count by contract
Churn rate by contract
Average monthly charges by churn status

Example query:

SELECT
    Contract,
    COUNT(*) AS Customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS ChurnedCustomers,
    ROUND(
        100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS ChurnRate
FROM customers
GROUP BY Contract;
Machine Learning

The target variable is:

Churn

where:

Yes = customer churned
No = customer stayed

The dataset was split into:

80% training data
20% testing data

Categorical features were converted using one-hot encoding.

Models Compared
Logistic Regression

Used as the primary classification model.

Random Forest

Used as a second model for comparison.

Model Results
Logistic Regression
Metric	          Result
Accuracy	      80.24%
Churn Precision	  64%
Churn Recall	  57%
Churn F1-score	  61%
ROC-AUC	0.84
Random Forest
Metric	          Result
Accuracy	     ~79%
Churn Precision	  62%
Churn Recall	  50%
Churn F1-score	  55%

Based on this test split and these metrics, Logistic Regression was selected as the model used by the Streamlit application.

Streamlit Dashboard

The project includes an interactive Streamlit dashboard that provides:

Customer dataset overview
Total customer count
Overall churn rate
Average monthly charges
Churn analysis by contract
Churn analysis by tenure
Customer-level churn prediction
Estimated churn probability

The application loads the trained Logistic Regression model from:

churn_model.pkl
Project Structure
customer-churn-prediction/
│
├── analysis.py
├── model.py
├── app.py
├── churn_model.pkl
├── .gitignore
└── README.md

The dataset file is intentionally excluded from the repository.

Installation

Clone the repository:

git clone https://github.com/Harshh-4204/customer-churn-prediction.git

Navigate into the project:

cd customer-churn-prediction

Create a virtual environment:

python -m venv venv

Activate the environment on Windows:

venv\Scripts\activate

Install the required packages:

pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib
Dataset Setup

Download the IBM Telco Customer Churn dataset and place it in the project directory as:

customer_churn.csv

The expected filename is:

customer_churn.csv
Run the Analysis
python analysis.py

This performs data cleaning, exploratory analysis, visualization, and SQL-based analysis.

Run the Machine Learning Model
python model.py

This trains the machine learning models, evaluates them, and saves the selected Logistic Regression model as:

churn_model.pkl
Run the Streamlit Dashboard
streamlit run app.py

The application will open in your browser.

Future Improvements

Potential improvements include:

Add more customer input fields to the prediction interface
Improve feature engineering
Tune model hyperparameters
Experiment with XGBoost and other classification algorithms
Use cross-validation for more robust evaluation
Add SHAP-based model explainability
Add customer segmentation
Add automated retention recommendations
Deploy the Streamlit application
Add automated data pipelines
Key Learning Outcomes

This project demonstrates practical experience with:

Data cleaning and preprocessing
Exploratory data analysis
Data visualization
SQL analytics
Feature encoding
Train/test splitting
Classification models
Model comparison
Classification metrics
ROC-AUC evaluation
Model persistence
Streamlit application development
Git and GitHub

## Author

**Harsh Chaudhary**

GitHub: https://github.com/Harshh-4204