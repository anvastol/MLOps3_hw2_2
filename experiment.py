import os
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import mlflow
from clearml import Task

def main():
    parser = argparse.ArgumentParser(description="Train a logistic regression model")
    parser.add_argument("--model-type", type=str, default="model_1", help="Type of model to train")
    args = parser.parse_args()

    mlflow.set_tracking_uri('file:///C:/Users/anvas/mlops_2_2/mlruns')

        # Создание эксперимента, если не существует
    if not mlflow.get_experiment_by_name("default"):
        mlflow.create_experiment("default")
    mlflow.set_experiment("default")

    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if args.model_type == "model_1":
        C = 0.1
        solver = "liblinear"
    elif args.model_type == "model_2":
        C = 1.0
        solver = "lbfgs"

    task = Task.current_task()
    task_name = task.name if task else "local"
    
    with mlflow.start_run(run_name=f"{task_name}-{args.model_type}"):
         mlflow.log_param("C", C)
         mlflow.log_param("solver", solver)
         
         model = LogisticRegression(C=C, solver=solver)
         model.fit(X_train, y_train)

         y_pred = model.predict(X_test)

         accuracy = accuracy_score(y_test, y_pred)
         mlflow.log_metric("accuracy", accuracy)
         if task:
             task.logger.report_scalar(title="Accuracy", series="Accuracy", value=accuracy)
    
         mlflow.sklearn.log_model(model, artifact_path="logistic_regression")

    print(f"Model training completed successfully. Accuracy: {accuracy}")

if __name__ == "__main__":
    main()