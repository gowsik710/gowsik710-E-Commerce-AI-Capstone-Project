# 🛒 E-Commerce AI Capstone Project

## 📌 Sentiment Analysis on Amazon E-commerce Reviews using ML Classifiers

This project performs **sentiment analysis** on Amazon product reviews using various machine learning classifiers. The goal is to classify customer reviews as either positive or negative, despite a significant class imbalance in the dataset. To handle this, the **SMOTE (Synthetic Minority Over-sampling Technique)** method is applied to balance the data.

---

## 🧠 Project Objectives

- Preprocess and clean raw textual review data
- Extract meaningful features from text using TF-IDF
- Handle class imbalance using SMOTE
- Train and evaluate multiple machine learning classifiers
- Identify the best-performing model based on accuracy and F1-score

---

## 📁 Dataset Summary

- **Source**: Amazon e-commerce review dataset  
- **Target variable**: Sentiment (positive/negative)  
- **Problem type**: Binary classification  
- **Challenge**: Class imbalance (majority positive reviews)

---

## 🔍 Detailed Workflow

### 1. 📦 Required Libraries
Imported necessary libraries for:
- **Data Handling**: `pandas`, `numpy`
- **Text Preprocessing**: `nltk`, `re`, `string`
- **Vectorization**: `TfidfVectorizer`
- **Modeling**: `scikit-learn`, `xgboost`, `imblearn`
- **Balancing**: `SMOTE` from `imblearn.over_sampling`

---

### 2. 📑 Data Exploration
- Loaded dataset using `pandas.read_csv()`
- Examined class distribution to reveal a strong **class imbalance**
  - Majority of reviews are positive
  - Minority (negative) class was underrepresented

---

### 3. 🧹 Text Cleaning and Preprocessing
Text reviews were cleaned and transformed using the following steps:
- Converted all text to **lowercase**
- Removed **punctuation**, **digits**, and **special characters**
- **Tokenized** the text
- Removed **stopwords** using NLTK
- **Lemmatized** tokens using `WordNetLemmatizer` to their base form

---

### 4. ✨ Feature Extraction
- Implemented **TF-IDF vectorization** to convert textual data into numerical format
- Applied `fit_transform()` on training data and `transform()` on test data
- Resulting vectors used as inputs for machine learning models

---

### 5. ⚖️ Handling Imbalanced Data
- Used **SMOTE (Synthetic Minority Oversampling Technique)** to generate synthetic examples for the minority class
- Ensured a **balanced dataset** prior to model training

---

### 6. 🧠 Models Implemented

| Classifier                | Description                                           |
|---------------------------|-------------------------------------------------------|
| Random Forest             | Ensemble of decision trees for robust classification |
| XGBoost                   | Gradient-boosted decision trees                       |
| SVM                       | Support Vector Machine with linear kernel            |
| Voting Classifier         | Ensemble method combining multiple model predictions |
| MLPClassifier (Neural Net)| Feedforward neural network                           |

---

### 7. 📊 Model Evaluation Metrics
Models were evaluated using:
- **Accuracy Score**
- **F1 Score** (especially important for imbalanced data)
- **Confusion Matrix**

#### ✅ Best Performer:
- **Random Forest Classifier**
  - Achieved highest **accuracy** and **F1-score**
  - Most reliable model for deployment

---

## ✅ Key Takeaways

- **SMOTE** is effective for handling imbalanced classification tasks
- **TF-IDF** is a powerful and scalable method for text representation
- Comparing **multiple classifiers** is crucial to find the best fit
- **Random Forest** stood out for its stability and accuracy

---

## 🛠️ Technologies Used

| Tool/Library   | Purpose                                         |
|----------------|-------------------------------------------------|
| Python         | Core programming language                       |
| Pandas, NumPy  | Data loading and manipulation                   |
| NLTK, Regex    | Text preprocessing and lemmatization            |
| Scikit-learn   | Model building, evaluation, and vectorization   |
| XGBoost        | Gradient boosting model                         |
| Imbalanced-learn | SMOTE for class balancing                     |

---

## 🚀 Conclusion

This project showcases how machine learning can be effectively applied to **real-world e-commerce review data** for sentiment analysis. By combining preprocessing, feature extraction, data balancing, and model evaluation, we successfully built a pipeline to classify customer feedback.  
Among the tested models, **Random Forest** proved to be the most effective in terms of both **accuracy and F1-score**, making it suitable for deployment in sentiment-based recommendation systems.

