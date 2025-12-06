from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pickle
import pandas as pd
import os
import csv
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load ML model
model = pickle.load(open('gwp.pkl', 'rb'))

# File to store all predictions
PREDICTIONS_FILE = 'predictions.csv'
if not os.path.exists(PREDICTIONS_FILE):
    # Create CSV with headers if it doesn't exist
    with open(PREDICTIONS_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'prediction_value', 'prediction_label'])


# ------------------- Helper to read historical predictions -------------------
def load_history():
    if os.path.exists(PREDICTIONS_FILE):
        df = pd.read_csv(PREDICTIONS_FILE)
        if 'timestamp' in df.columns and 'prediction_value' in df.columns:
            months = df['timestamp'].tolist()
            values = df['prediction_value'].tolist()
        else:
            months, values = [], []
    else:
        months, values = [], []
    return months, values


# ------------------- STATIC ROUTES -------------------
@app.route("/")
def home():
    return render_template('home.html')


@app.route("/about")
def about():
    return render_template('about.html')


@app.route("/predict")
def predict_page():
    return render_template('predict.html')


@app.route("/submit")
def submit_page():
    months, values = load_history()
    return render_template('submit.html', months=months, values=values)


# ------------------- SINGLE PREDICTION -------------------
@app.route('/pred', methods=['POST'])
def pred():
    # Collect form inputs
    fields = [
        'quarter', 'department', 'day', 'team', 'targeted_productivity', 'smv',
        'over_time', 'incentive', 'idle_time', 'idle_men',
        'no_of_style_change', 'no_of_workers', 'month'
    ]
    data = [float(request.form[f]) for f in fields]
    prediction = model.predict([data])[0]

    # Label
    if prediction <= 0.3:
        pred_text = "The employee is averagely productive."
    elif prediction <= 0.8:
        pred_text = "The employee is medium productive."
    else:
        pred_text = "The employee is highly productive."

    # Store prediction in CSV
    with open(PREDICTIONS_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(), prediction, pred_text])

    # Chart data based on input (deterministic)
    targeted_productivity = float(request.form['targeted_productivity'])
    idle_time = float(request.form['idle_time'])
    team = int(request.form['team'])
    idle_men = int(request.form['idle_men'])
    over_time = int(request.form['over_time'])
    smv = float(request.form['smv'])
    no_of_style_change = int(request.form['no_of_style_change'])
    incentive = float(request.form['incentive'])

    weekly = [targeted_productivity * (0.8 + 0.05 * i) for i in range(5)]
    long_term = np.cumsum(weekly).tolist()
    total_time = targeted_productivity + idle_time
    focus = [
        round(targeted_productivity / total_time * 100),
        round(idle_time / total_time * 100)
    ]
    work_time = targeted_productivity * team
    idle_total = idle_time * idle_men
    resource_allocation = [round(work_time), round(idle_total), 0]
    idle_vs_overtime = [
        idle_total,
        over_time,
        max(0, team * 8 - (idle_total + over_time))
    ]
    quality = [
        round(smv * team),
        round(no_of_style_change * 5),
        round(incentive)
    ]
    charts = {
        "weekly": weekly,
        "long_term": long_term,
        "focus_score": focus,
        "resource_allocation": resource_allocation,
        "idle_vs_overtime": idle_vs_overtime,
        "quality_metrics": quality
    }

    months, values = load_history()

    return render_template(
        'submit.html',
        prediction_text=pred_text,
        charts=charts,
        months=months,
        values=values
    )


# ------------------- BULK CSV PREDICTION -------------------
@app.route('/bulk_predict', methods=['POST'])
def bulk_predict():
    file = request.files.get('file')
    if not file or file.filename == "":
        return redirect(url_for("predict_page"))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    df = pd.read_csv(filepath)
    expected = [
        'quarter', 'department', 'day', 'team', 'targeted_productivity',
        'smv', 'over_time', 'incentive', 'idle_time', 'idle_men',
        'no_of_style_change', 'no_of_workers', 'month'
    ]
    for col in expected:
        if col not in df.columns:
            df[col] = 0

    df = df[expected]
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    predictions = []
    for _, row in df.iterrows():
        p = model.predict([row.tolist()])[0]
        if p <= 0.3:
            label = "Averagely productive"
        elif p <= 0.8:
            label = "Medium productive"
        else:
            label = "Highly productive"
        predictions.append(label)

        # Store bulk prediction in CSV
        with open(PREDICTIONS_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now(), p, label])

    df['prediction'] = predictions
    months, values = load_history()

    return render_template(
        'submit.html',
        bulk=True,
        table=df.to_html(classes="table table-striped", index=False),
        months=months,
        values=values
    )


# ------------------- VISUALIZATION ROUTE REMOVED -------------------
@app.route("/visualization")
def visualization():
    return redirect(url_for("submit_page"))


# ------------------- MAIN -------------------
if __name__ == "__main__":
    app.run(debug=True)
