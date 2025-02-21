# Logistics Regression to Classify Iris Species and evaluate model with Learning Curves and Confusion Matrix

This repository contains the code and report of classifying 3 different species of Iris plant using the Iris dataset and Logistic Regression model.
After implementing the classification model, learning curves and confusion matrices are used to evaluate its efficiency.

## Table of Contents

- [Introduction](#introduction)
- [Methodology](#methodology)
- [Results](#results)
- [Files](#files)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)

## Introduction

This assignment explores the relationship between training set size and test accuracy by visualizing learning curves.  A confusion matrix is also generated to evaluate the performance of a Logistic Regression model on the Iris dataset.

## Methodology

1. **Dataset:** The Iris dataset was used, containing measurements of sepal and petal lengths and widths for three Iris species (setosa, versicolor, and virginica).
2. **Model:** A Logistic Regression model was trained and evaluated.
3. **Training and Evaluation:** The dataset was split into 80% training and 20% testing sets using stratified sampling. The model was trained on varying sizes of the training set and evaluated on the fixed test set.
4. **Confusion Matrix:** A confusion matrix was generated to visualize the model's classification performance.

## Results

* **Learning Curve:** Plots of different training data sizes vs accuracy.
* **Confusion Matrix:** Visualization as a heat map 
* **Key Metrics:** Table with values of accuracy, precision, recall, F1 score, specificity

## Files

* `main.py`: The Python script containing the code for training, evaluating, and plotting.
* `report.pdf`: The report summarizing the project, including analysis of the results.  (Or `report.md` if you have a markdown report).
* `README.md`: This file.

## How to Run

1. Clone the repository: `git clone [repository URL]`
2. Navigate to the directory: `cd [repository directory]`
3. Install the required dependencies (see below).
4. Run the script: `python main.py`

## Dependencies

* Python 3
* scikit-learn
* matplotlib
* numpy (if used for confusion matrix display)

You can install the dependencies using pip:

```bash
pip install scikit-learn matplotlib numpy
