import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import random

# Trains and evaluates logistic regression models with 6 different training sizes.
def train_and_evaluate(X, Y, train_sizes):
    X_train_full, X_test, Y_train_full, Y_test = train_test_split(X, Y, train_size=0.8, stratify=Y, random_state=42)
    results = {} # initialize a dictionary for containing accuracy scores for each training size.

    for size in train_sizes:
        indices = random.sample(range(len(X_train_full)), size)
        X_train = X_train_full[indices]
        Y_train = Y_train_full[indices]

        logit_model = LogisticRegression(max_iter=1000)
        logit_model.fit(X_train, Y_train)


        Y_predictions = logit_model.predict(X_test)
        accuracy = accuracy_score(Y_test, Y_predictions)
        results[size] = accuracy
    return results


# plot
def plot_learning_curve(train_sizes, accuracies):
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, accuracies, marker=".", linestyle='-')
    plt.xlabel("Training Size")  # Corrected label
    plt.ylabel("Accuracy")  # Corrected label
    plt.title("Test Accuracy vs. Training Set Size")
    plt.grid(True)
    plt.xticks(train_sizes)
    plt.show()

def main():
    #1 Load the Iris dataset
    iris = datasets.load_iris()
    X = iris.data  # Features
    Y = iris.target  # Target labels

    #2 Define training sizes
    train_sizes = [20, 40, 60, 80, 100, 120]

    #3 Train and evaluate
    results = train_and_evaluate(X, Y, train_sizes)

    #4 Print results
    print("\nTraining Size vs. Accuracy:")
    for size, accuracy in results.items():
        print(f"Training Size: {size} -> Accuracy: {accuracy:.4f}")

    #5 Plot
    plot_learning_curve(train_sizes, list(results.values()))

    # 6 Confusion Matrix for Full Training Set
    X_train_full, X_test, Y_train_full, Y_test = train_test_split(X, Y, train_size=0.8, stratify=Y, random_state=42)
    logit_model_full = LogisticRegression(max_iter=1000)
    logit_model_full.fit(X_train_full, Y_train_full)
    Y_pred_full = logit_model_full.predict(X_test)


    cm = confusion_matrix(Y_test, Y_pred_full)

    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.xticks(np.arange(cm.shape[1]), labels=iris.target_names)  # Add x-axis labels
    plt.yticks(np.arange(cm.shape[0]), labels=iris.target_names)  # Add y-axis labels
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Add text annotations inside the heatmap cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='w')  # 'w' for white text

    plt.show()
if __name__ == "__main__":
    main()