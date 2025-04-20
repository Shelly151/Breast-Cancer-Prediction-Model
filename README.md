# 🧠 Breast Cancer Prediction Using Machine Learning

## 📌 Overview

This project focuses on predicting whether a tumor is **benign** or **malignant** using a machine learning model trained on the **Breast Cancer Wisconsin Diagnostic Dataset**. It utilizes **Logistic Regression**, a simple yet powerful classification algorithm, to assist in the early detection of breast cancer.

---

## 🎯 Objective

- Develop a supervised learning model to classify breast cancer tumors.
- Apply preprocessing techniques and feature scaling.
- Evaluate the model’s performance using metrics like accuracy, precision, recall, and F1-score.
- Visualize the insights and correlations within the dataset.

---

## 🗂️ Dataset

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- **Records**: 569
- **Features**: 30 numerical attributes (mean, standard error, worst)
- **Target**: `Diagnosis` - `M` (Malignant), `B` (Benign)

---

## 🧰 Technologies Used

- **Language**: Python 🐍
- **Environment**: Google Colab ☁️
- **Libraries**:
  - `pandas`, `numpy` - Data manipulation
  - `matplotlib`, `seaborn` - Data visualization
  - `sklearn` - Machine learning (Logistic Regression, metrics, preprocessing)

---

## 🔄 Workflow

1. **Data Loading**: Read the dataset using Pandas.
2. **Exploratory Data Analysis (EDA)**:
   - Countplots
   - Feature distributions
   - Correlation matrix
3. **Preprocessing**:
   - Dropping unnecessary columns
   - Label encoding (`M` → 1, `B` → 0)
   - Feature scaling using `StandardScaler`
4. **Model Training**:
   - Train-test split (80:20)
   - Logistic Regression model training
5. **Evaluation**:
   - Accuracy
   - Classification Report
   - Confusion Matrix
6. **Visualization**: Diagnostic insights using Seaborn and Matplotlib

---

## 📈 Results

- **Model**: Logistic Regression
- **Accuracy**: ~94.7%
- **Precision/Recall (Malignant)**: High recall ensures fewer false negatives

> ✅ The model reliably classifies tumors and can serve as a strong baseline for healthcare-based ML applications.

---

## 🔮 Future Enhancements

- Test other algorithms: Random Forest, SVM, XGBoost
- Feature engineering and selection
- Deploy the model using Streamlit or Flask
- Create a web-based diagnostic tool with real-time inputs

---

## 🚀 How to Run

1. Clone this repository or open the Colab notebook.
2. Ensure all necessary libraries are installed.
3. Run the cells step by step from top to bottom.
4. Check evaluation metrics and charts for results.

---

## 📚 References

- [UCI Breast Cancer Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- Kaggle notebooks and community forums

---

## 🙋‍♀️ Author

**Shelly Gupta**  
6th Semester, BTech CSE (CCVT)  
SAP ID: 500101943  

---

