import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load and split the dataset
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 2. Set up MLflow for experiment tracking
mlflow.set_experiment("Iris_Classification_Experiment")  # You can create a custom experiment name

# 3. Logistic Regression Model (underfitting by reducing iterations and increasing regularization)
with mlflow.start_run():

    # Train Logistic Regression model (underfitting by reducing iterations and increasing regularization)
    lr_model = LogisticRegression(max_iter=50, C=1000, solver='liblinear')  # Reduced max_iter and high regularization
    lr_model.fit(X_train, y_train)

    # Make predictions
    lr_predictions = lr_model.predict(X_test)

    # Calculate accuracy
    lr_accuracy = accuracy_score(y_test, lr_predictions)

    # Confusion matrix
    lr_conf_matrix = confusion_matrix(y_test, lr_predictions)

    # Log parameters and metrics to MLflow
    mlflow.log_param("model", "Logistic Regression")
    mlflow.log_metric("accuracy", lr_accuracy)

    # Log the confusion matrix as an image
    plt.figure(figsize=(6,6))
    sns.heatmap(lr_conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.title("Confusion Matrix - Logistic Regression")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("lr_conf_matrix.png")  # Save confusion matrix image
    mlflow.log_artifact("lr_conf_matrix.png")  # Log the image to MLflow

    # Log the model
    mlflow.sklearn.log_model(lr_model, "logistic_regression_model")

    # Register the Logistic Regression model to the Model Registry
    log_model_uri = "runs:/{}/logistic_regression_model".format(mlflow.active_run().info.run_id)
    registered_model = mlflow.register_model(log_model_uri, "Logistic_Regression_Model")

    # End the MLflow run
    mlflow.end_run()

    print(f"Logistic Regression Accuracy: {lr_accuracy}")
    print("Confusion Matrix:")
    print(lr_conf_matrix)

# 4. Random Forest Model (underfitting by reducing trees, depth, and min_samples_split)
with mlflow.start_run():
    
    # Train Random Forest model (underfitting by reducing trees and limiting depth)
    rf_model = RandomForestClassifier(n_estimators=10, max_depth=3, min_samples_split=10, random_state=0, criterion='entropy')
    rf_model.fit(X_train, y_train)

    # Make predictions
    rf_predictions = rf_model.predict(X_test)

    # Calculate accuracy
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    print(f"Random Forest Accuracy: {rf_accuracy}")

    # Confusion matrix
    rf_conf_matrix = confusion_matrix(y_test, rf_predictions)
    print(f"Random Forest Confusion Matrix:")
    print(rf_conf_matrix)

    # Log parameters and metrics to MLflow
    mlflow.log_param("model", "Random Forest")
    mlflow.log_metric("accuracy", rf_accuracy)

    # Log the confusion matrix as an image
    plt.figure(figsize=(6,6))
    sns.heatmap(rf_conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.title("Confusion Matrix - Random Forest")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("rf_conf_matrix.png")  # Save confusion matrix image
    mlflow.log_artifact("rf_conf_matrix.png")  # Log the image to MLflow

    # Log the model
    mlflow.sklearn.log_model(rf_model, "random_forest_model")

    # Register the Random Forest model to the Model Registry
    rf_model_uri = "runs:/{}/random_forest_model".format(mlflow.active_run().info.run_id)
    registered_rf_model = mlflow.register_model(rf_model_uri, "Random_Forest_Model")

    # End the MLflow run
    mlflow.end_run()

    print(f"Random Forest Accuracy: {rf_accuracy}")
    print("Confusion Matrix:")
    print(rf_conf_matrix)

# Now to load the model by version or by 'latest':
# Example of loading the Logistic Regression model by version or latest
logistic_model_uri = "models:/Logistic_Regression_Model/latest"  # Use 'latest' to fetch the most recent version
loaded_lr_model = mlflow.sklearn.load_model(logistic_model_uri)

# Example of loading the Random Forest model by version or latest
rf_model_uri = "models:/Random_Forest_Model/latest"  # Use 'latest' to fetch the most recent version
loaded_rf_model = mlflow.sklearn.load_model(rf_model_uri)

