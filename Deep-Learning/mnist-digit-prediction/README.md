```markdown
# Handwritten Digit Recognition

Welcome to the **Handwritten Digit Recognition** project! 🎉  
This project focuses on training neural network models to classify handwritten digits using an MNIST-like dataset.

---

## 🚀 Project Overview

The goal of this project is to predict handwritten digits (0-9) from images using various neural network models, including:
- A basic Fully Connected Neural Network (MLP)
- A more advanced Convolutional Neural Network (CNN)

You will work with a dataset consisting of:
- **Training Images**: Labeled with digits {0, 1, 2, ..., 9}
- **Test Images**: Unlabeled, to be predicted by the model.

---

## 📂 Project Structure

The project is organized into the following folders and files:

- **`data/`**: Contains training and testing images.
- **`src/`**: Source code for neural network models:
    - **`mnist_neural_network.py`**: Simple MLP for digit classification.
    - **`mnist_cnn.py`**: CNN for better performance.
- **`outputs/`**: Contains model evaluation results (accuracy, loss, etc.):
    - **`image-predictions.png`**: Visual representation of the predicted images.
    - **`imclassified-images.png`**: Examples of the classified images.
    - **`loss-curve.png`**: The loss and accuracy curve for model evaluation.
    - **`weight-layers.png`**: Visuals showing the weight layers of the trained models.

---

## 📋 Files Overview

### `mnist_neural_network.py`
- Implements a **fully connected neural network** (MLP).
- Suitable for baseline classification of digits.

### `mnist_cnn.py`
- Implements a **Convolutional Neural Network** (CNN).
- Captures spatial features, leading to better performance on image classification.

---

## 🛠 Setup & Installation

### 1. Clone the repository:

```bash
git clone https://github.com/sureshkuc/mnist-digit-prediction.git
```

### 2. Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset:

Place the MNIST-like dataset (training and test images) in the `data/` folder.

---

## 🚀 How to Train the Model

### 1. Train with the Fully Connected Neural Network (MLP):

```bash
python src/mnist_neural_network.py
```

### 2. Train with the Convolutional Neural Network (CNN):

```bash
python src/mnist_cnn.py
```

> Both scripts will store the results (e.g., training accuracy, loss) in the `outputs/` folder. 📊

---

## 📈 Model Evaluation

Once training is completed, evaluate your model's performance on the test set. The evaluation results will be saved in the `outputs/` folder. You can visualize metrics such as accuracy and loss curves for further analysis.

- **Loss and Accuracy Curves**: ![Loss Curve](outputs/loss-curve.png)  
  This image shows the training loss and accuracy over time for model evaluation.



- **MissClassified Images**: ![MissClassified Images](outputs/imclassified-images.png)  
  Examples of images and their predicted labels after training.

- **Weight Layers**: ![Weight Layers](outputs/weight-layers.png)  
  Visual representation of the weights in different layers of the neural network model.

---

## 📝 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🏆 Contributing

Feel free to contribute to this project! Open an issue or submit a pull request if you have suggestions, improvements, or bug fixes. 🚀

---

## 💬 Contact

If you have any questions or suggestions, feel free to reach out:  
- Email: skcberlin dot gmail.com  
- LinkedIn: 
```
