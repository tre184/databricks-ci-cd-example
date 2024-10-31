import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    # Configurer MLflow pour utiliser un chemin relatif dans le dossier du projet
    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
    # Load data
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log metrics and model with MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "random_forest_model")

    # Register the model
    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/random_forest_model"
    mlflow.register_model(model_uri, "IrisRFModel")

if __name__ == "__main__":
    main()