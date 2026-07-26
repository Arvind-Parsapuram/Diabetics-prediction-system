# 🩺 Diabetes Prediction System

An interactive **Machine Learning-based Diabetes Prediction System** built using **Streamlit**, **Random Forest**, and **XGBoost-based missing value imputation**. The application analyzes patient clinical parameters and predicts the likelihood of diabetes through an intuitive and user-friendly web interface.

> **Disclaimer:** This project is developed for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis or treatment.

---

## 📌 Project Overview

Diabetes is one of the most common chronic diseases worldwide, and early diagnosis plays a crucial role in preventing severe health complications.

This project provides an interactive web application that predicts whether a patient is diabetic based on various medical parameters. The system performs intelligent data preprocessing, handles missing values using machine learning, trains a classification model, evaluates its performance, and provides real-time predictions through a modern dashboard.

The project is built using the **Pima Indian Diabetes Dataset** and demonstrates the complete machine learning pipeline from data preprocessing to deployment.

---

## ✨ Features

- 🩺 Interactive Streamlit Web Application
- 🤖 Machine Learning-based Diabetes Prediction
- 📊 Automated Data Preprocessing
- ⚡ XGBoost-based Missing Value Imputation
- 🌲 Random Forest Classification Model
- 📈 Model Performance Evaluation
- 📉 Feature Importance Visualization
- 📋 Dataset Statistics and Analysis
- 📌 Risk Percentage Estimation
- 🎨 Clean and Responsive User Interface

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| Programming Language | Python |
| Frontend | Streamlit |
| Machine Learning | Scikit-learn |
| Missing Value Imputation | XGBoost |
| Data Processing | Pandas, NumPy |
| Dataset | Pima Indian Diabetes Dataset |

---

## 📊 Dataset

The project uses the **Pima Indian Diabetes Dataset**, which contains diagnostic information collected from female patients of Pima Indian heritage.

### Input Features

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

### Target Variable

| Value | Meaning |
|-------|---------|
| 0 | Non-Diabetic |
| 1 | Diabetic |

---

## ⚙️ Machine Learning Workflow

### 1️⃣ Data Loading

The application loads the dataset from **diabetes.csv**.

### 2️⃣ Data Cleaning

Biologically impossible zero values are replaced with missing values (`NaN`) for selected medical features.

### 3️⃣ Missing Value Imputation

An **XGBoost Regressor** predicts missing **Insulin** values, improving data quality before classification.

### 4️⃣ Data Preprocessing

Remaining missing values are replaced using median imputation.

### 5️⃣ Model Training

A **Random Forest Classifier** is trained using an **80:20 stratified train-test split**.

### 6️⃣ Model Evaluation

The model is evaluated using:

- Training Accuracy
- Testing Accuracy
- Precision
- Recall
- F1-Score
- Classification Report
- Feature Importance

### 7️⃣ Prediction

The trained model predicts diabetes risk based on user-provided clinical information.

---

## 📱 Application Modules

### 📊 Data Overview

- Dataset statistics
- Data summary
- Missing value analysis
- Feature distribution visualization

---

### 📈 Model Performance

- Model architecture
- Training accuracy
- Testing accuracy
- Classification report
- Feature importance graph

---

### 🔍 Patient Prediction

- Patient parameter input
- Diabetes prediction
- Risk probability
- Confidence score
- Clinical risk assessment

---

## 🚀 Getting Started

### Clone the Repository

```bash
git clone https://github.com/your-username/Diabetes-Prediction-System.git
```

### Navigate to the Project Folder

```bash
cd Diabetes-Prediction-System
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

---

## 📦 Requirements

- Python 3.10+
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- XGBoost

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## 📈 Model Details

### Missing Value Imputation

- **Algorithm:** XGBoost Regressor

### Classification Model

- **Algorithm:** Random Forest Classifier

### Data Split

- Training Data: **80%**
- Testing Data: **20%**

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Feature Importance

---

## 🔮 Future Enhancements

- Save and load trained models using Pickle/Joblib
- Add Confusion Matrix visualization
- Add ROC Curve and Precision-Recall Curve
- Support multiple Machine Learning algorithms
- Store prediction history
- Export prediction reports
- Deploy on Streamlit Community Cloud
- Improve UI with interactive visualizations

---

## 🎯 Learning Outcomes

This project demonstrates practical implementation of:

- Machine Learning Workflow
- Data Cleaning and Preprocessing
- Missing Value Imputation
- Feature Engineering Concepts
- Classification using Random Forest
- Model Evaluation
- Interactive Dashboard Development
- Streamlit Deployment

---

## 👨‍💻 Author

**P. Parsapuram**

**B.Tech - Computer Science and Engineering (Data Science)**

ACE Engineering College, Hyderabad

---

⭐ If you found this project helpful, consider giving it a **Star** on GitHub.
