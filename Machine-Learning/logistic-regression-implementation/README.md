```markdown
# 🚀 Logistic Regression from Scratch

This repository contains a clean and modular implementation of **Logistic Regression** from scratch using **Python** and **NumPy**. The model is trained on the **ZIP Code Dataset** for digit classification and includes full support for evaluation and visualization.

---

## 📌 Objective

The goal of this exercise is to understand and implement Logistic Regression like a single-layer neural network. The core components are:

- Implementing the **Sigmoid** and **Cross-Entropy Loss** functions from scratch.
- Using **Gradient Descent** for training the weights and bias.
- Exploring alternative loss functions like **Mean Squared Error (MSE)**.
- (Optional) Extending Logistic Regression to handle **multi-class classification**.

---

## 🧠 Concepts Covered

- Logistic Regression fundamentals
- Binary classification using sigmoid activation
- Cross Entropy vs Mean Squared Error (MSE) as loss functions
- Gradient Descent optimization
- ROC curve and probability visualization
- Modular design with reusable Python components

---

## 📁 Project Structure


logistic-regression/
├── requirements.txt
├── README.md
├── outputs/
│   ├── test\_data.png
│   ├── roc.png
│   ├── prob.png
│   └── ... # Additional outputs
├── src/
│   ├── config.py          # Hyperparameters and settings
│   ├── model.py           # Sigmoid, loss functions, logistic regression model
│   ├── train.py           # Training logic with gradient descent
│   ├── plot.py            # Visualization tools (ROC, probability histograms, etc.)
│   ├── main.py            # Pipeline orchestration
│   └── evaluation.py      # Accuracy, precision, recall, confusion matrix, etc.


---

## 🧪 Dataset

We use the **ZIP Code Dataset**, a standard dataset for handwritten digit classification. Each image represents a digit (0–9) as a vector of grayscale pixel intensities.

---

## 🛠️ Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/logistic-regression.git
   cd logistic-regression
````

2. **Create Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Pipeline**

   ```bash
   python src/main.py
   ```

---

## 📊 Outputs

The following output visualizations are stored in the `outputs/` folder:

* `test_data.png` – Visual check of test dataset
* `roc.png` – ROC Curve for performance evaluation
* `prob.png` – Histogram of predicted probabilities

---

## ❓ Experimentation

### (a) Mean Squared Error vs Cross Entropy

We compare the performance of MSE with Cross Entropy:

* ✅ Cross Entropy is optimal for classification with sigmoid.
* ⚠️ MSE can still converge but often slower and less stable.

To run with MSE instead of Cross Entropy, modify the config:

```python
# In config.py
LOSS_FUNCTION = "mse"
```

### (b) Multi-class Classification (Optional)

You can extend the model to multi-class using the **One-vs-Rest** strategy:

* Each class gets a separate logistic regression model.
* Prediction = class with the highest probability.

Changes needed:

* `w` becomes a matrix of shape `[n_classes, n_features]`
* `b` becomes a vector of shape `[n_classes]`

---

## 🔍 Evaluation Metrics

Implemented in `evaluation.py`:

* Accuracy
* Confusion Matrix
* Precision, Recall, F1-Score
* ROC-AUC

---

## 📈 Visualizations

Created using `plot.py`, examples include:

* Loss & accuracy curves over epochs
* ROC curve
* Probability distributions of predictions

---

## 📦 Dependencies

Main dependencies listed in `requirements.txt`:

* numpy
* matplotlib
* scikit-learn
* seaborn

---

---

## 📜 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## 💡 Acknowledgements

* ZIP Code Dataset: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Zip+Code+Data)

---

```

