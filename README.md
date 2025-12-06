# **Employee Performance Prediction** 

## **Project Overview**

This project aims to predict garment factory worker productivity using machine learning models. It leverages historical employee performance data to forecast daily productivity, helping HR and managers optimize workforce allocation, improve efficiency, and make data-driven decisions.

## **📂 Project Structure**

Employee-Performance-Prediction/  
│  
├── Dataset/  
│   └── garments_worker_productivity.csv      # Raw dataset  
│  
├── Flask/                                    # Flask web application  
│   ├── app.py                                # Main application  
│   ├── evaluate_model.py                     # Script to evaluate model performance  
│   ├── gwp.pkl                               # Trained XGBoost model  
│   ├── predictions.csv                       # Sample predictions  
│   ├── requirements.txt                      # Python dependencies  
│   ├── static/                               # CSS, JS, images  
│   ├── templates/                            # HTML templates (home, predict, about, submit)  
│   └── uploads/                              # Uploaded files for bulk prediction  
│  
├── IBM Files/                                # Backup of model files  
│   └── gwp.pkl  
│  
├── Training Files/                           # Jupyter notebooks  
│   ├── Employee_Prediction.ipynb             # EDA + model training notebook  
│   ├── gwp.pkl                               # Serialized model  
│   └── mcle.pkl                              # Optional alternate model  
│  
└── README.md                                 # Project documentation

## **📊 Dataset Description**

The dataset tracks daily productivity metrics of sewing and finishing teams in a garment factory.

* **File:** `garments_worker_productivity.csv`
* **Size:** ~1,200 entries
* **Columns:** 15 features
* **Target Variable:** `actual_productivity`

### **Key Features**

* **Temporal/Structural:** `quarter`, `day`, `department`, `team`
* **Metrics:** `targeted_productivity`, `over_time`, `idle_time`, `no_of_style_change`, `smv`, `incentive`
* **Personnel:** `no_of_workers`, `idle_men`

## **🧠 Machine Learning Approach**

* **Model Used:** XGBoost Regressor (XGBRegressor)
* **Task:** Regression (predicting `actual_productivity`)
* **Features Used:** `quarter`, `day`, `department`, `team`, `targeted_productivity`, `over_time`, `idle_time`, `no_of_style_change`, `smv`, `incentive`, `idle_men`, `no_of_workers`

### **Model Performance (Test Data)**

| Metric | Score |
| :---- | :---- |
| **R² Score** | 0.5453 |
| **RMSE** | 0.1176 |
| **MAE** | 0.0729 |

## **🛠 Installation & Setup**

1. **Clone the repository**
   ```bash
   git clone https://github.com/SabarishR08/Employee-Performance-Prediction.git
   cd Employee-Performance-Prediction/Flask
   ```

2. **Create and activate a virtual environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Mac/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask application**
   ```bash
   python app.py
   ```

5. **Access the Application**
   Open your browser and visit: `http://127.0.0.1:5000`

## **📈 How to Use**

### **1. Predict Single Entry**

* Navigate to the **Predict** page
* Enter the specific employee and daily metrics into the form
* Click **Submit** to view the predicted productivity score

### **2. Bulk Prediction**

* Upload a CSV file containing multiple entries (matching the dataset structure)
* Predictions will be processed and saved as `bulk_predictions.csv`

### **3. Evaluate Model**

* Run the evaluation script to see R², RMSE, MAE, and sample predictions in the console:
  ```bash
  python evaluate_model.py
  ```

## **💡 Key Insights from Analysis**

* **Department Variance:** Finishing department teams consistently show higher productivity than Sewing teams.  
* **Overtime:** Interestingly, overtime does not always correlate with higher productivity.  
* **Team Performance:** There is significant variance between teams; some consistently outperform others.  
* **Feature Impact:** Sparse features such as incentive, idle\_time, and style changes showed limited impact on the final model predictions.

## **📂 Deliverables**

* **Code:** Flask web app, Jupyter notebook (`Employee_Prediction.ipynb`), evaluation scripts
* **Models:** Trained XGBoost models (`gwp.pkl`)
* **Data:** Original dataset (`garments_worker_productivity.csv`)
* **Reports:** Predictions CSV files and analysis insights

## **📌 Notes**

* Ensure `gwp.pkl` is present in the `Flask/` folder before running the app
* The project is built using **Python 3.12**, **XGBoost**, **pandas**, **scikit-learn**, and **Flask**

## **🎓 Author**

**Sabarish R**

* [LinkedIn Profile](https://www.linkedin.com/in/sabarishr08)

*If you find this project useful, please give it a star on GitHub! ⭐*
