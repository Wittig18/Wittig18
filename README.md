**Project Overview**`

This project demonstrates a machine learning pipeline for predicting whether a breast tumor is malignant or benign based on tumor features. It uses the Breast Cancer dataset from scikit-learn and a Logistic Regression model to classify tumors.

The goal is to support early detection of cancer, providing insights that could assist medical professionals in diagnosis.

**Problem Statement**
Given features extracted from breast tumors (like radius, texture, smoothness), can we accurately classify a tumor as malignant (cancerous) or benign (non-cancerous)?

Early detection is critical in increasing survival rates and reducing unnecessary medical interventions.

**Dataset**
Source: sklearn.datasets.load_breast_cancer()

Samples: 569

Features: 30 numeric features (mean, standard error, and “worst” values for various tumor characteristics)

Target classes:

0 → Benign

1 → Malignant

**Features**
Some important features include:

mean radius

mean perimeter

mean concavity

worst radius

worst concavity

These features influence model predictions and can help understand tumor aggressiveness.

**Tools & Libraries**

Python 3.1

Pandas

NumPy

Scikit-learn (LogisticRegression, train_test_split, metrics)

Matplotlib / Seaborn (for visualization, optional)

**Methodology**

Load Dataset

Load the breast cancer dataset from scikit-learn.

Data Exploration & Preprocessing

Check feature distributions

Train-Test Split

Split dataset into training (80%) and testing (20%)

Model Training

Train Logistic Regression model

Handle convergence warnings by increasing max_iter

Evaluation

**Evaluate model using:**

Accuracy

Confusion Matrix

Classification Report (Precision, Recall, F1-Score)

Focus on recall for malignant class to minimize false negatives

Feature Importance

Inspect coefficients to identify features contributing most to predictions

**Results**

Accuracy: 0.93

Confusion Matrix:
array([[41,  4],
       [ 4, 65]])

Classification Report:
               precision    recall  f1-score   support

           0       0.91      0.91      0.91        45
           1       0.94      0.94      0.94        69

    accuracy                           0.93       114
   macro avg       0.93      0.93      0.93       114
weighted avg       0.93      0.93      0.93       114


Interpretation:

The model detects 91% of actual malignant cases (high recall).

Only 9% of benign cases are misclassified as malignant (low false positives).

Overall, the model is reliable for early detection purposes.

**Key Learnings**
Logistic Regression is effective for binary classification problems.

Feature scaling improves convergence and model performance.

Recall is crucial in medical datasets where false negatives can be life-threatening.

Feature importance analysis helps understand which tumor characteristics are most predictive.

**Future Improvements**
Experiment with other classifiers (Random Forest, Gradient Boosting).

Perform feature selection or PCA to reduce dimensionality.

Deploy the model as a web app for real-time predictions.

Integrate patient metadata for more personalized predictions.

**Project Structure**
Breast-Cancer-Prediction/
│
├── breast_cancer_prediction.ipynb   # Main Jupyter/Colab notebook
├── README.md                        # Project overview & documentation
├── requirements.txt                 # Python dependencies


**References**
Scikit-learn Breast Cancer Dataset

Logistic Regression documentation: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
