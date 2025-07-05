from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_endpoint():
    response = client.post(
        "/predict",
        data={
            "run_id": "test_run",
            "model_name": "test_model",
            "text": "Sample text"
        }
    )
    assert response.status_code == 200
    assert "prediction" in response.json()