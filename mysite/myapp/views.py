from django.shortcuts import render
from django.http import HttpResponse
from joblib import load
import numpy as np
import pandas as pd

# Load trained XGBoost model
model = load("loan.joblib")

def home(request):
    return render(request, 'home.html')

def add(request):
    return render(request, 'add.html')

def pred(request):
    if request.method == "POST":
        # Get inputs from form
        gender = request.POST['Gender']
        married = request.POST['Married']
        dependents = request.POST['Dependents']
        self_emp = request.POST['Self_Employed']
        income = float(request.POST['ApplicantIncome'])
        loan_amt = float(request.POST['LoanAmount'])
        loan_term = float(request.POST['Loan_Amount_Term'])
        credit = int(request.POST['Credit_History'])
        prop_area = request.POST['Property_Area']

        # Make input row
        inp = pd.DataFrame([{
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Self_Employed": self_emp,
            "ApplicantIncome": income,
            "LoanAmount": loan_amt,
            "Loan_Amount_Term": loan_term,
            "Credit_History": credit,
            "Property_Area": prop_area
        }])

        # Prediction + Probability
        pred_class = model.predict(inp)[0]
        pred_prob = model.predict_proba(inp)[0][pred_class] * 100

        # Result text
        if pred_class == 1:
            result = "Approved ✅"
        else:
            result = "Rejected ❌"

        context = {
            "result": result,
            "probability": round(pred_prob, 2)
        }
        return render(request, "pred.html", context)
    else:
        return HttpResponse("Invalid Request")
