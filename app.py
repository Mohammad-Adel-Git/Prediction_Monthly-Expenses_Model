import numpy as np
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)
model = pickle.load(open("model_file.pkl", "rb"))

min_vals = [517, 0, 0, 1, 0, 0]
max_vals = [541566, 423247, 2, 26, 8, 10]

@app.route('/predict/<int:input1>/<int:input2>/<int:input3>/<int:input4>/<int:input5>/<int:input6>', methods=['GET'])
def predict(input1, input2, input3, input4, input5, input6):
    # Normalize the input values
    normalized_input = []
    for i in range(6):
        normalized_value = (locals()[f"input{i+1}"] - min_vals[i]) / (max_vals[i] - min_vals[i])
        normalized_input.append(normalized_value)

    # Convert the input to a NumPy array
    input_array = np.array([normalized_input])

    # Make the prediction
    prediction = model.predict(input_array)

    # Print and Return the response
    print("the predicted Expenses is:", prediction)
    response = {
        'prediction': prediction.tolist()
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
