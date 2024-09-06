from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    prediction = model.predict([data["input"]])
    return jsonify({"prediction": prediction.tolist()})

if __name__ == "__main__":
    app.run(debug=True)
