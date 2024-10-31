import mlflow
from notebooks.train_model import main
import tempfile

# Ajouter le chemin du projet au PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_model_training():
    # Définir le tracking URI pour le test
    if "GITHUB_ACTIONS" in os.environ:
        # En CI/CD, utiliser un répertoire temporaire pour `mlruns`
        tracking_uri = "file://" + tempfile.mkdtemp()
    else:
        # En local, utiliser le répertoire `mlruns` dans le projet
        tracking_uri = "file://" + os.path.abspath("mlruns")

    mlflow.set_tracking_uri(tracking_uri)
    print(f"Tracking URI configuré pour le test : {mlflow.get_tracking_uri()}")

    try:
        # Exécuter la fonction main
        main()
    except Exception as e:
        # Afficher un message d'erreur pour aider au diagnostic
        print(f"Erreur lors de l'exécution de main(): {e}")
        raise

    # Vérifier si le run MLflow a été enregistré
    client = mlflow.tracking.MlflowClient()
    experiments = client.list_experiments()
    assert len(experiments) > 0, "No experiments found in mlruns"

    # Vérifier qu'il y a au moins un run dans le dernier experiment
    last_experiment_id = experiments[-1].experiment_id
    runs = client.list_run_infos(last_experiment_id)
    assert len(runs) > 0, "No runs found in the last experiment"