import pytest
import pandas as pd
from pathlib import Path


from src.models import train_and_evaluate

@pytest.fixture
def mock_data():
    """Generates a tiny dummy dataset to keep the test lightning fast."""
    X = pd.DataFrame({
        'feature_1': [1, 2, 3, 4, 5, 6, 7, 8],
        'feature_2': [8, 7, 6, 5, 4, 3, 2, 1]
    })
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    return X, X, y, y

def test_main_generates_all_artifacts(monkeypatch, tmp_path, mock_data):
    """
    Tests if main() successfully runs and writes all PNGs and CSVs 
    to the results directory without crashing.
    """
    X_train, X_test, y_train, y_test = mock_data
   
    monkeypatch.setattr(train_and_evaluate, 'X_train', X_train)
    monkeypatch.setattr(train_and_evaluate, 'X_test', X_test)
    monkeypatch.setattr(train_and_evaluate, 'y_train', y_train)
    monkeypatch.setattr(train_and_evaluate, 'y_test', y_test)
    
    monkeypatch.chdir(tmp_path)
    
  
    train_and_evaluate.main()
    
    results_dir = tmp_path / "results"
    assert results_dir.exists(), "The 'results' directory was not created."
    assert results_dir.is_dir(), "The 'results' path exists but is not a directory."
 
    expected_files = [
        "metrics.csv",
        "confusion_matrix_knn.png",
        "confusion_matrix_dt.png",
        "confusion_matrix_rf.png",
        "model_comparison.png",
        "rf_feature_importance.png",
        "sample_match_probability.png"
    ]
    
    for file_name in expected_files:
        file_path = results_dir / file_name
        assert file_path.exists(), f"Missing expected output file: {file_name}"
        assert file_path.stat().st_size > 0, f"File {file_name} was created but is completely empty."