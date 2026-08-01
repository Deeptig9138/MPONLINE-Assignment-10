from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load serialized model
model = joblib.load('model.pkl')

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "API active",
        "message": "Heart Disease Prediction REST API is running successfully."
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Expected clinical features order:
        # [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        features = [
            data['age'], data['sex'], data['cp'], data['trestbps'], 
            data['chol'], data['fbs'], data['restecg'], data['thalach'], 
            data['exang'], data['oldpeak'], data['slope'], data['ca'], data['thal']
        ]
        
        input_array = np.array([features])
        prediction = model.predict(input_array)[0]
        
        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected"
        
        return jsonify({
            "prediction": result,
            "prediction_code": int(prediction)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
