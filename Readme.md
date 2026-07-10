# 💳 Credit Card Approval Prediction System

A Machine Learning-based web application that predicts whether a customer's credit card application will be **Approved** or **Rejected** using applicant information. The application is built with **Python**, **Flask**, and **Scikit-learn**, and provides predictions through an interactive web interface.

---

## 📌 Features

- Predicts credit card approval status
- User-friendly Flask web application
- Data preprocessing and feature engineering
- Handles missing values
- Balances data using SMOTE
- Compares multiple Machine Learning models
- Uses Random Forest as the final prediction model
- Displays prediction confidence
- Responsive web interface

---

## 🛠 Technologies Used

### Programming Language

- Python 3

### Machine Learning

- Scikit-learn
- XGBoost
- Imbalanced-Learn (SMOTE)

### Web Development

- Flask
- HTML5
- CSS3

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Model Serialization

- Joblib

---

## 📂 Project Structure

```
Credit_card_approval_system/
│
├── app.py
├── prepare_data.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── model_comparison.csv
│
├── templates/
│   ├── index.html
│   └── result.html
│
└── static/
    └── style.css
```

> **Note:** The trained model (`model.pkl`), encoders, and datasets are not included in this repository because of GitHub file size limits.

---

## 📊 Dataset

This project uses the following datasets:

- `application_record.csv`
- `credit_record.csv`

## Dataset

The datasets used in this project are available on Kaggle.

Download them and place them inside the `dataset/` folder.

```
dataset/
├── application_record.csv
└── credit_record.csv
```

Dataset: https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction

The datasets are merged using the **Applicant ID** to create:

```
final_credit_dataset.csv
```

Place the datasets inside a folder named:

```
dataset/
```

---

## ⚙️ Machine Learning Pipeline

1. Load datasets
2. Merge datasets
3. Handle missing values
4. Feature engineering
5. Encode categorical features
6. Split training and testing data
7. Apply SMOTE for class balancing
8. Train multiple ML models
9. Compare model performance
10. Save the best model
11. Deploy using Flask

---

## 🤖 Machine Learning Models

The following algorithms were trained and compared:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

### Best Performing Model

- **Random Forest**
- **Accuracy:** **85.79%**
- **ROC-AUC Score:** **0.7851**

---

## 📈 Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC Score

---

## 📝 Input Features

The web application accepts:

- Gender
- Annual Income
- Number of Children
- Family Members
- Income Type
- Education Level
- Family Status
- Housing Type
- Occupation
- Own Car
- Own House
- Age
- Years Employed
- Work Phone
- Phone
- Email

---

## 🎯 Prediction Output

The application displays:

- Credit Card Approval Status (Approved / Rejected)
- Prediction Confidence (%)

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/nageswari196/credit_card_approval_prediction.git
```

Move into the project folder:

```bash
cd credit_card_approval_prediction
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## ▶️ Training the Model

Place the datasets inside the `dataset/` folder.

Run:

```bash
python prepare_data.py
```

Then train the model:

```bash
python train_model.py
```

This generates:

```
model.pkl
encoders.pkl
feature_names.pkl
model_comparison.csv
```

---

## 🌐 Running the Application

Start the Flask server:

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

## 🔄 Project Workflow

```
Raw Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Encoding
      │
      ▼
SMOTE
      │
      ▼
Model Training
      │
      ▼
Model Evaluation
      │
      ▼
Best Model Selection
      │
      ▼
Flask Deployment
      │
      ▼
Prediction
```

---

## 🔮 Future Enhancements

- Explainable AI (SHAP/LIME)
- User Authentication
- Database Integration
- Cloud Deployment
- REST API
- Model Retraining Pipeline
- Docker Support

---

## 👩‍💻 Author

**Nageswari**

B.Tech – Computer Science and Engineering

Vignan's Lara Institute of Technology & Science

---

## 📄 License

This project is developed for educational and academic purposes.
