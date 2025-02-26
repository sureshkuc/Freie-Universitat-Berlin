---
# Sparse Hierarchical Graph Classification

In this study, we implement a sparse hierarchical graph classification architecture designed to predict class labels for entire graphs. The main aim of this approach is to reduce memory usage during model training without sacrificing accuracy, especially when compared to prior prominent methods that are not scalable to large graphs.

---

## 🚀 Project Overview

This project focuses on:
- **Reducing Memory Usage**: The proposed method requires only O(V + E) storage complexity, as compared to the quadratic O(kV²) storage complexity required by the state-of-the-art approach (DiffPool1).
- **Maintaining Accuracy**: The method achieves competitive performance, with accuracy within 1% of the state-of-the-art algorithm.
- **Scalability**: The method scales efficiently on large graph datasets, ensuring faster processing and lower memory consumption.

We compare the performance of this architecture against the state-of-the-art DiffPool1 on several graph classification benchmarks.

---

## 📊 Dataset Overview

We evaluated the proposed method on the following established graph classification datasets:

| Dataset         | No. of Classes | Avg. No. of Nodes | Avg. No. of Edges | Node Attributes (Dim.) |
|-----------------|----------------|-------------------|-------------------|------------------------|
| **Enzymes**     | 6              | 32.63             | 62.14             | 18                     |
| **Proteins**    | 2              | 39.06             | 72.82             | 29                     |
| **D&D**         | 2              | 284.32            | 715.66            | -                      |
| **Collab**      | 3              | 74.49             | 2457.78           | -                      |

---

## 📂 Project Structure

The project is organized as follows:

- **`src/`**: Source code for implementing the graph classification models:
  - **`collab_main.py`**: Main script for the Collab dataset.
  - **`dd_main.py`**: Main script for the D&D dataset.
  - **`enzymes_main.py`**: Main script for the Enzymes dataset.
  - **`proteins_main.py`**: Main script for the Proteins dataset.

- **`outputs/`**: Contains the output of the model, including visualizations and performance results.
  - **`proposed-pipeline.png`**: Image showing the proposed pipeline.
  
- **`data/`**: Folder containing the dataset files (Enzymes, Proteins, D&D, Collab).

- **`docs/`**: Documentation files, including detailed explanations of the methods and algorithms used.

---

## 🧑‍💻 Model Parameters

The model parameters used in the study, based on Cangea et al., are as follows:

| Dataset         | No. of Feature Layers | Learning Rate | Epochs |
|-----------------|-----------------------|---------------|--------|
| **Enzymes**     | 128                   | 0.0005        | 100    |
| **Proteins**    | 64                    | 0.005         | 40     |
| **D&D**         | 64                    | 0.0005        | 20     |
| **Collab**      | 128                   | 0.0005        | 30     |

---

## 📈 Performance

The proposed method achieved the following accuracies on different datasets:

- **Enzymes**: 64.17%
- **D&D**: 78.59%
- **Collab**: 74.54%
- **Proteins**: 75.46%

---

## 🛠 Setup & Installation

### 1. Clone the repository:

```bash
git clone https://github.com/sureshkuc/sparse-hierarchical-graph-classification.git
```

### 2. Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Dataset Preparation:

Place the datasets (Enzymes, Proteins, D&D, Collab) in the `data/` folder.

---

## 🚀 How to Run

To run the model for a specific dataset, execute one of the following scripts:

### 1. For the **Collab** dataset:

```bash
python src/collab_main.py
```

### 2. For the **D&D** dataset:

```bash
python src/dd_main.py
```

### 3. For the **Enzymes** dataset:

```bash
python src/enzymes_main.py
```

### 4. For the **Proteins** dataset:

```bash
python src/proteins_main.py
```

---

## 📊 Results and Visualization

- The results of the model, including performance metrics (accuracy, loss), are stored in the `outputs/` folder.
- The **`proposed-pipeline.png`** provides a visualization of the proposed pipeline for graph classification.

---

## 📝 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🏆 Contributing

Feel free to contribute to this project! If you have suggestions, improvements, or bug fixes, open an issue or submit a pull request.

---

## 💬 Contact

If you have any questions or suggestions, feel free to reach out:

- Email: skcberlin dot gmail.com  
- LinkedIn: 

---
