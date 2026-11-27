from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_iris_classification():
    response = client.post(
        "/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        },
    )
    assert response.status_code == 200
    result = response.json()
    assert result["species"] == "setosa"
    assert result["class_id"] == 0
    assert 0 <= result["confidence"] <= 1


def test_predict_rejects_missing_or_out_of_range_measurements():
    missing = client.post("/predict", json={"sepal_length": 5.1})
    invalid = client.post(
        "/predict",
        json={
            "sepal_length": -1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        },
    )
    assert missing.status_code == 422
    assert invalid.status_code == 422
