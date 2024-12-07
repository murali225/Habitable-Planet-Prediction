from flask import Flask, request, render_template
import numpy as np
import pandas as pd  # Added to handle DataFrame conversion
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model pipeline 
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    """Root URL that displays the form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict habitability based on user inputs.
    Expects form data with keys:
    - 'Planet_Name'
    - 'Discovery_Method'
    - 'Distance'
    - 'Period'
    - 'Mass'
    - 'Temperature'
    """
    try:
        # Retrieve form data
        data = request.form

        # Extract categorical and numerical inputs
        name = data.get("Planet_Name", "")
        discovery_method = data.get("Discovery_Method", "")
        mass = float(data.get("Mass", 0))  # Use 0 if missing
        period = float(data.get("Period", 0))  # Use 0 if missing
        distance = float(data.get("Distance", 0))  # Use 0 if missing
        host_star_temp = float(data.get("Temperature", 0))  # Use 0 if missing

        # Combine all features into the correct order for the model
        # Create a DataFrame to maintain consistency with model expectations
        input_data = pd.DataFrame([[name, discovery_method, mass, period, distance, host_star_temp]],
                                  columns=["Name", "Discovery method", "Mass", "Period", "Distance", "Host star temp"])

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Render the result using the template
        return render_template("index.html", prediction_text="The planet's habitability is predicted as: {}".format(prediction))

    except Exception as e:
        # Handle errors and return an appropriate message
        return render_template("index.html", prediction_text="Error: {}".format(str(e)))

if __name__ == "__main__":
    # Run the Flask app in debug mode
    app.run(host='0.0.0.0',port=8080)
    
