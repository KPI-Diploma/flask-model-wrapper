from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from keras.models import load_model
import pickle

app = Flask(__name__)


def run_model(colors):
    assert len(colors) == 3, "Input should be an array of length 3"
    assert all(isinstance(color, str) for color in colors), "Input should be an array of 3 colors"

    model = load_model('model.h5')

    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)

    colors_rgb = [hex_to_rgb(color) for color in colors]
    colors_flat = [component for color in colors_rgb for component in color]

    input_df = pd.DataFrame([colors_flat], columns=[
        'color1_r', 'color1_g', 'color1_b',
        'color2_r', 'color2_g', 'color2_b',
        'color3_r', 'color3_g', 'color3_b'
    ])

    predictions = model.predict(input_df.values)[0]
    top_10_indices = np.argsort(predictions)[::-1][:10]

    return le.inverse_transform(top_10_indices).tolist()


@app.route('/')
def greeting():
    return 'Food is Good!'


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if 'colors' not in data:
        return jsonify({"error": "No colors provided"}), 400

    try:
        result = run_model(data['colors'])
    except AssertionError:
        return jsonify({"error": "Something went wrong with input"}), 400

    return jsonify({"predictions": result}), 200


def hex_to_rgb(hex_color):
    return [int(hex_color[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]


if __name__ == '__main__':
    app.run()
