from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# load model and scaler
model  = joblib.load('models/fraud_model.pkl')
scaler = joblib.load('models/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data     = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        pred     = model.predict(features)[0]
        prob     = model.predict_proba(features)[0][1]
        result   = {
            'prediction'  : int(pred),
            'probability' : round(float(prob) * 100, 2),
            'status'      : 'FRAUD DETECTED' if pred == 1 else 'LEGITIMATE'
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/stats')
def stats():
    stats_data = {
        'total_transactions' : 284807,
        'fraud_cases'        : 492,
        'legit_cases'        : 284315,
        'fraud_percentage'   : 0.173,
        'model_roc_auc'      : 96.88,
        'model_recall'       : 81.0
    }
    return jsonify(stats_data)

if __name__ == '__main__':
    app.run(debug=True)