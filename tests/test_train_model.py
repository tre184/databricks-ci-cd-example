from notebooks.train_model import main
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
def test_model_training():
    # Run the main function
    main()
    # Check if MLflow run exists
    assert os.path.exists("mlruns"), "MLflow run not found"

    # Additional tests can be added here

