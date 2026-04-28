# breast-cancer-classification
# Overview
This project applies five supervised machine learning classification algorithms to the breast cancer dataset available in the sklearn library. The objective is to evaluate and compare model performance in classifying tumors as malignant or benign.

# Dataset
* Source: sklearn.datasets.load_breast_cancer()
* Samples: 569
* Features: 30 numeric features (mean radius, texture, perimeter, area, smoothness, etc.)
* Target Classes: Malignant (0) — 212 samples | Benign (1) — 357 samples
* Missing Values: None

# Project Structure
````````````````````````````
breast-cancer-classification/
│
├── breast_cancer_classification.ipynb
└── README.md
`````````````````````````````

# Preprocessing Steps
| Step | Method | Reason |
|---|---|---|
| Missing value check | `df.isnull().sum()` | Confirm data completeness |
| Train-test split | `train_test_split` (80/20, stratified) | Preserve class proportions in both train and test sets |
| Feature scaling | `StandardScaler` | Normalize feature ranges for distance-based algorithms like SVM and k-NN |

# Algorithms Implemented
| sl.no | Algorithm | Key Parameter |
|---|---|---|
| 1 | Logistic Regression | max_iter=10000 |
| 2 | Decision Tree Classifier | random_state=42 |
| 3 | Random Forest Classifier | n_estimators=100 | 
| 4 | Support Vector Machine| kernel='rbf' |
| 5 | K-Nearest Neighbors |n_neighbors=5 |

# Results
| Model | Accuracy |
|---|---|
| Logistic Regression | 98.25% |
| SVM | 98.25% |
| Random Forest | 95.61% |
| K-Nearest Neighbors | 95.61% |
| Decision Tree | 91.23% |

**Best Model**: Logistic Regression & SVM — tied at 98.25%

**Worst Model**: Decision Tree — 91.23%

# Key Findings
Logistic Regression and SVM performed equally best, benefiting from well-scaled, linearly separable features after StandardScaling. Decision Tree performed worst due to overfitting — without pruning it fails to generalise to unseen data, reflected in the lowest malignant precision (85%) among all models. For a medical classification task like tumour diagnosis, Logistic Regression is the preferred model — it matches SVM in accuracy while being simpler and more interpretable, which is critical in clinical decision-making.

# Evaluation Metrics Used
Accuracy, Precision, Recall, F1-Score, and Classification Report (full per-class breakdown)

# How to Run
1. Clone or download this repository
2. Open terminal and navigate to the project folder
3. Launch Jupyter Notebook
4. Run all cells using Kernel → Restart & Run All

