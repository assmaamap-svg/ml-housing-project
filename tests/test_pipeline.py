from src.training.pipeline import run_pipeline


def test_pipeline_returns_metrics():
    metrics = run_pipeline()
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics