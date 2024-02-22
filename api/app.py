from utils.custom_model import CustomModelling, load_checkpoint
from pathlib import Path
from flask import Flask, jsonify, request, render_template
import pandas as pd
from utils.dataset import feature_engineering


def load_model():
    model_path = Path("./models/model_best.pickle")
    print(model_path)
    model = load_checkpoint(model_path)
    return model


custom_model = CustomModelling(
    model=load_model(), x_train=None, y_train=None, x_test=None
)

class_names = {
    0: "Insufficient Weight",
    1: "Normal Weight",
    2: "Obesity Type I",
    3: "Obesity Type II",
    4: "Obesity Type III",
    5: "Overweight Level I",
    6: "Overweight Level II",
}


app = Flask(__name__)


def predict_sample(sample: dict) -> dict:
    sample = [sample]
    sample_df = pd.DataFrame(sample)

    feature = feature_engineering(data=sample_df)

    predictions = custom_model.inference(feature)
    return {"prediction": predictions.item(), "class": class_names[predictions.item()]}


@app.route("/")
def home():
    return render_template("home.html")


@app.post("/predict")
def predict():
    data = request.json
    try:
        sample = data["data"]
    except KeyError:
        return jsonify({"error": "Not text was fetched!"})
    result_data = predict_sample(sample)
    try:
        result = jsonify(
            result_data
        )
    except TypeError as e:
        result = jsonify({"error": str(e)})

    return result


if __name__ == "__main__":
    app.run(debug=True)
