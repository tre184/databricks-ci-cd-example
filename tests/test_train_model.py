import sys
import os
from notebooks.train_model import main

# Ajouter le chemin du projet au PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_model_training():
    try:
        # Exécuter la fonction main
        main()
    except Exception as e:
        # Afficher un message d'erreur pour aider au diagnostic
        print(f"Erreur lors de l'exécution de main(): {e}")
        raise

    # Chemin complet pour mlruns
    mlruns_path = os.path.join(os.getcwd(), "mlruns")

    # Vérifier si le répertoire mlruns a été créé
    assert os.path.exists(mlruns_path), "MLflow run not found in the expected path"

    # Vérifier qu'il y a au moins un run dans mlruns
    experiments = os.listdir(mlruns_path)
    assert len(experiments) > 0, "No experiments found in mlruns"