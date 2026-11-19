
from flask import Flask, render_template, request
import joblib
import numpy as np
import os

# ==========================================
# 1. INITIALIZE FLASK APPLICATION
# ==========================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ==========================================
# 2. LOAD MACHINE LEARNING MODEL AND SCALER
# ==========================================

MODEL_PATH = os.path.join(BASE_DIR, "revenue_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("revenue_model.pkl not found!")

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("scaler.pkl not found!")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# ==========================================
# 3. DEFINE INPUT FEATURES
# ==========================================

FEATURES = [
    "net_income",
    "market_cap",
    "employees",
    "legal_units_count",
    "direct_subsidiaries_count",
    "max_hierarchy_depth",
    "company_type_encoded"
]


# ==========================================
# 4. HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template("home.html")


# ==========================================
# 5. PREDICTION PAGE
# ==========================================

@app.route("/predict", methods=["GET", "POST"])
def predict():

    # Display input form
    if request.method == "GET":

        return render_template("predict.html")


    # Process submitted enterprise details
    try:

        # Collect input values
        input_values = []

        for feature in FEATURES:

            value = request.form.get(feature, "").strip()

            if value == "":
                raise ValueError("Please fill in all fields.")

            input_values.append(float(value))


        # Convert inputs into NumPy array
        input_data = np.array(
            input_values,
            dtype=float
        ).reshape(1, -1)


        # Validate input dimensions
        if input_data.shape[1] != scaler.n_features_in_:
            raise ValueError("Incorrect number of input features.")


        # Apply StandardScaler
        scaled_data = scaler.transform(input_data)


        # Generate revenue prediction
        prediction = model.predict(scaled_data)[0]


        # Convert prediction to Python float
        prediction = float(prediction)


        # Prepare enterprise details for result page
        enterprise_details = dict(
            zip(FEATURES, input_values)
        )


        # Display result page
        return render_template(
            "result.html",
            prediction=prediction,
            values=input_values,
            enterprise_details=enterprise_details
        )


    except ValueError as e:

        return render_template(
            "predict.html",
            error=str(e)
        )


    except Exception:

        app.logger.exception("Revenue prediction failed")

        return render_template(
            "predict.html",
            error="Prediction failed. Please check your inputs."
        )


# ==========================================
# 6. ERROR HANDLING
# ==========================================

@app.errorhandler(404)
def page_not_found(error):

    return """
    <h1>404 - Page Not Found</h1>
    <a href="/">Return to Home Page</a>
    """, 404


# ==========================================
# 7. RUN FLASK APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
