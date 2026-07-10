from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# ==========================================================
# LOAD MODEL
# ==========================================================

model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")
feature_names = joblib.load("feature_names.pkl")

# ==========================================================
# HOME PAGE
# ==========================================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================================
# PREDICTION
# ==========================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = {}

        # ------------------------
        # Categorical Inputs
        # ------------------------

        data["CODE_GENDER"] = request.form["CODE_GENDER"]
        data["FLAG_OWN_CAR"] = request.form["FLAG_OWN_CAR"]
        data["FLAG_OWN_REALTY"] = request.form["FLAG_OWN_REALTY"]

        data["NAME_INCOME_TYPE"] = request.form["NAME_INCOME_TYPE"]
        data["NAME_EDUCATION_TYPE"] = request.form["NAME_EDUCATION_TYPE"]
        data["NAME_FAMILY_STATUS"] = request.form["NAME_FAMILY_STATUS"]
        data["NAME_HOUSING_TYPE"] = request.form["NAME_HOUSING_TYPE"]

        data["OCCUPATION_TYPE"] = request.form["OCCUPATION_TYPE"]

        # ------------------------
        # Numerical Inputs
        # ------------------------

        data["CNT_CHILDREN"] = int(request.form["CNT_CHILDREN"])

        data["AMT_INCOME_TOTAL"] = float(
            request.form["AMT_INCOME_TOTAL"]
        )

        data["CNT_FAM_MEMBERS"] = float(
            request.form["CNT_FAM_MEMBERS"]
        )

        data["AGE"] = int(
            request.form["AGE"]
        )

        data["YEARS_EMPLOYED"] = int(
            request.form["YEARS_EMPLOYED"]
        )

        data["FLAG_WORK_PHONE"] = int(
            request.form["FLAG_WORK_PHONE"]
        )

        data["FLAG_PHONE"] = int(
            request.form["FLAG_PHONE"]
        )

        data["FLAG_EMAIL"] = int(
            request.form["FLAG_EMAIL"]
        )

        # ------------------------
        # Encode Categorical Data
        # ------------------------

        categorical_columns = [

            "CODE_GENDER",
            "FLAG_OWN_CAR",
            "FLAG_OWN_REALTY",
            "NAME_INCOME_TYPE",
            "NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS",
            "NAME_HOUSING_TYPE",
            "OCCUPATION_TYPE"

        ]

        for col in categorical_columns:

            try:

                data[col] = encoders[col].transform(
                    [data[col]]
                )[0]

            except:

                return render_template(
                    "result.html",
                    result="Invalid Input",
                    probability=None
                )

        # ------------------------
        # Arrange Features
        # ------------------------

        final_input_df = pd.DataFrame([data])[feature_names]

        prediction = model.predict(final_input_df)[0]

        probability = None

        if hasattr(model, "predict_proba"):

            probability = round(

                max(
                    model.predict_proba(final_input_df)[0]
                ) * 100,

                2

            )

        if prediction == 1:

            result = "Approved"

        else:

            result = "Rejected"

        return render_template(

            "result.html",

            result=result,

            probability=probability

        )

    except Exception as e:

        return f"Error : {e}"


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    app.run(debug=True)