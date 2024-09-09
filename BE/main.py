from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the crop recommendation model
loaded_model = pickle.load(open("crop_model.pkl", 'rb'))

# Initialize the Flask app
app = Flask(__name__)

# Define the main route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Crop Recommendation System
@app.route('/crop-recommendation', methods=['GET', 'POST'])
def crop_recommendation():
    if request.method == 'POST':
        try:
            # Get inputs from form
            N = float(request.form['N'])
            P = float(request.form['P'])
            K = float(request.form['K'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            pH = float(request.form['pH'])
            rainfall = float(request.form['rainfall'])

            # Predict using the model
            input_value = np.array([[N, P, K, temperature, humidity, pH, rainfall]])
            prediction = loaded_model.predict(input_value)
            pred = prediction[0]

            # Mapping prediction to crop names
            crops = {
                1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
            }

            result = crops.get(pred, "Sorry, we could not determine the best crop to be cultivated with the provided data.")
            return render_template('crop_recommendation.html', result=result)

        except Exception as e:
            return f"Error: {str(e)}"
    return render_template('crop_recommendation.html')


if __name__ == '__main__':
    app.run(debug=True)
