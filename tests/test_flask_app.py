from app import app


VALID_FORM_DATA = {
    "fixed_acidity": "7.4",
    "volatile_acidity": "0.70",
    "citric_acid": "0.00",
    "residual_sugar": "1.9",
    "chlorides": "0.076",
    "free_sulfur_dioxide": "11",
    "total_sulfur_dioxide": "34",
    "density": "0.9978",
    "ph": "3.51",
    "sulphates": "0.56",
    "alcohol": "9.4",
}


def test_home_page_returns_200():
    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert b"WineMetric" in response.data


def test_prediction_endpoint_returns_200():
    client = app.test_client()

    response = client.post(
        "/predict",
        data=VALID_FORM_DATA,
    )

    assert response.status_code == 200
    assert b"Predicted Wine Quality" in response.data


def test_prediction_endpoint_handles_invalid_input():
    client = app.test_client()

    invalid_data = VALID_FORM_DATA.copy()
    invalid_data["alcohol"] = "invalid"

    response = client.post(
        "/predict",
        data=invalid_data,
    )

    assert response.status_code == 200
    assert b"Please enter valid numeric values" in response.data