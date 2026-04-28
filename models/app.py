import pandas as pd
import xgboost as xgb
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the NEW model
model = xgb.Booster()
model.load_model("xgb_source_sector.json")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    try:
        # Create the DataFrame using the keys we just set in index.html
        input_df = pd.DataFrame([{
            'State': data['state'],
            'Year': int(data['year']),
            'Total population': float(data['prod']) * 1000000 # Convert millions to actual number
        }])

        input_df['State'] = input_df['State'].astype('category')
        
        dmatrix = xgb.DMatrix(input_df, enable_categorical=True)
        prediction = model.predict(dmatrix)
        
        return jsonify({'emissions': float(prediction[0]), 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400

if __name__ == '__main__':
    app.run(port=5000, debug=True)