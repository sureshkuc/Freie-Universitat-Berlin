```markdown
# Breast Cancer Classification using Deep Learning

## Goal of the Project
The goal of this project is to build a Deep Learning algorithm that classifies breast tumour tissue as either Benign (Non-Cancer) or Malignant (Cancer) using breast cancer histopathology images.

## Main Result
A Convolutional Neural Network (CNN) classifier was successfully built to classify breast tumour tissues as either cancerous (Malignant) or non-cancerous (Benign). The model leverages advanced architectures such as DenseNet201 for improved performance.

## Personal Key Learnings
- We built a deep learning classifier for the first time, which gave us practical insights into the key aspects of deep learning algorithms.
- We learned how to handle and preprocess histopathology image data, fine-tune model architectures, and evaluate performance using metrics like accuracy, loss, and confusion matrices.

## Keywords
- Convolutional Neural Network (CNN)
- DenseNet201
- Deep Learning
- Breast Cancer Histopathology Images

## Project Structure

The project is organized into the following directories and files:

### `src/` - Source code for model, training, and evaluation
- `config.py`: Configuration settings for model training and evaluation.
- `model.py`: The model architecture (CNN based).
- `train.py`: Script to train the deep learning model.
- `model_predictions_and_plot.py`: Script for making predictions and generating plots for visualization.
- `evaluation.py`: Script to evaluate model performance (accuracy, confusion matrix, etc.).
- `main.py`: Entry point for the project, including data preprocessing, model training, and evaluation.

### `outputs/` - Folder containing output files
- `breast-cancer-images.png`: Visual representation of breast cancer images used in training.
- `roc.png`: Receiver Operating Characteristic (ROC) curve for model evaluation.
- `accuracy.png`: Plot showing the accuracy of the model over time.
- `cm.png`: Confusion matrix plot.
- `loss.png`: Plot showing the loss during model training.

### `data/` - Folder containing dataset (not specified here but assumed to include breast cancer histopathology images).

### `docs/` - Documentation for the project.

### `requirements.txt` - Required Python packages for the project.

## Installation

To set up the project environment, you can install the required dependencies by running:

```bash
pip install -r requirements.txt
```

## Usage

1. **Data Preprocessing**: Preprocess the breast cancer histopathology images using the configuration defined in `config.py`.
2. **Model Training**: Run `train.py` to train the CNN model.
3. **Evaluation**: Evaluate the model's performance by running `evaluation.py`. This will generate plots like accuracy, loss, confusion matrix, and ROC curve.
4. **Predictions**: Use `model_predictions_and_plot.py` to make predictions on new data and visualize the results.

## Model Architecture

The model is built using a Convolutional Neural Network (CNN) with DenseNet201 architecture for better feature extraction and classification accuracy.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- [DenseNet](https://arxiv.org/abs/1608.06993): DenseNet paper for the architecture used.
- [Keras](https://keras.io/): Deep learning library used to build and train the model.
- [TensorFlow](https://www.tensorflow.org/): Backend library for neural network computations.
```
