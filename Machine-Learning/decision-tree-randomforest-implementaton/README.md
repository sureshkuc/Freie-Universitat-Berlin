```markdown
# Project6: Decision Trees and Random Forests on SPAM Dataset

This project implements **Decision Trees** and **Random Forests** from scratch using Python and applies them to the [UCI Spambase Dataset](https://archive.ics.uci.edu/ml/datasets/spambase). The goal is to classify emails as spam or not spam and analyze feature importance, model performance, and misclassification costs.

## 📁 Project Structure

```

project6/
├── src/
│   ├── config.py             # Configuration and constants
│   ├── model.py              # Decision tree and random forest implementations
│   ├── train.py              # Training logic
│   ├── evaluation.py         # Evaluation metrics, confusion matrix, AUC, etc.
│   ├── plot.py               # Plotting functions (feature importance, results)
│   ├── utils.py              # Helper functions
│   ├── main.py               # Main driver script
│   └── test\_model.py         # Unit tests for the model
│
├── outputs/
│   ├── feature\_imp.png       # Top 5 features by importance
│   ├── auc.png               # ROC curve and AUC for random forest
│   └── result.png            # Model accuracy and performance summary
│
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation

````

---

## 🧠 Exercises Overview

### 🔹 Exercise 1: Decision Trees
- Implemented a classification decision tree using Python and NumPy.
- Custom loss function (Gini Index by default; can be changed).
- **Asymmetric Misclassification Cost**:
  - Misclassifying a **genuine email as spam** is **10× worse** than the reverse.
  - This was handled by weighting the loss function during the split criterion to penalize such misclassifications.
- **Feature Importance**:
  - Top 5 most informative features were identified using node impurity reduction.
  - 📊 Visualization saved as `outputs/feature_imp.png`.

### 🔹 Exercise 2: Random Forest
- Implemented a random forest classifier using bagging and feature subsetting.
- Evaluation using:
  - ✅ Accuracy
  - 🧾 Confusion Matrix
  - 📈 ROC Curve & AUC (`outputs/auc.png`)
- Explored the effect of varying the number of trees on performance.
- Identified an optimal range for number of trees based on validation results.

---

## 📊 Visual Outputs

- `outputs/feature_imp.png` — Top 5 feature importances (based on information gain)
- `outputs/auc.png` — ROC curve for Random Forest
- `outputs/result.png` — Overall performance summary

---

## ⚙️ Installation & Usage

### Step 1: Clone the Repository
```bash
git clone https://github.com/sureshkuc
````

### Step 2: Set Up Environment

Install dependencies (preferably in a virtual environment):

```bash
pip install -r requirements.txt
```

### Step 3: Run the Project

Train the models and generate outputs:

```bash
python src/main.py
```

---

## 🧪 Testing

Unit tests for the model logic can be run with:

```bash
python -m unittest src/test_model.py
```

---

## 📈 Metrics & Results

* Confusion Matrix (Random Forest):

  * True Positives, True Negatives, False Positives, False Negatives
* AUC score
* Feature importance rankings
* Performance with asymmetric costs

---

## 🔍 Dataset

* **Source**: [UCI Spambase Dataset](https://archive.ics.uci.edu/ml/datasets/spambase)
* **Features**: 57 attributes (word frequencies, character frequencies, capital letter run lengths)
* **Label**: Spam (1) or Not Spam (0)

---

## 📌 Future Work

* Add support for pruning in decision trees.
* Perform cross-validation to improve model selection.
* Extend to other text classification datasets.

---

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

```
