from flask import Flask, render_template, request

from mlProject.pipeline.prediction import PredictionPipeline


app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = {
            "fixed acidity": float(request.form["fixed_acidity"]),
            "volatile acidity": float(request.form["volatile_acidity"]),
            "citric acid": float(request.form["citric_acid"]),
            "residual sugar": float(request.form["residual_sugar"]),
            "chlorides": float(request.form["chlorides"]),
            "free sulfur dioxide": float(request.form["free_sulfur_dioxide"]),
            "total sulfur dioxide": float(request.form["total_sulfur_dioxide"]),
            "density": float(request.form["density"]),
            "pH": float(request.form["ph"]),
            "sulphates": float(request.form["sulphates"]),
            "alcohol": float(request.form["alcohol"]),
        }

        prediction_pipeline = PredictionPipeline()
        prediction = prediction_pipeline.predict(features)

        return render_template(
            "index.html",
            prediction=round(prediction, 2),
        )

    except (ValueError, KeyError) as exc:
        return render_template(
            "index.html",
            error="Please enter valid numeric values for all fields.",
        )

    except FileNotFoundError:
        return render_template(
            "index.html",
            error="Trained model was not found. Run the training pipeline first.",
        )

    except Exception:
        return render_template(
            "index.html",
            error="An unexpected error occurred while generating the prediction.",
        )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=8080,
        debug=True,
    )