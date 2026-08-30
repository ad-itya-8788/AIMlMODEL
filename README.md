# Classroom Attendance Prediction

## How to run this project

1. Install dependencies:
   pip install -r requirements.txt

2. Train the models (run once):
   Open `final_notebook.ipynb` in Jupyter and run all cells top to bottom.
   This reads `attendense.csv`, cleans it, engineers features, and trains +
   evaluates every algorithm required by the project brief:

   Regression:  Linear Regression, Decision Tree, Random Forest,
                Gradient Boosting, SVM, XGBoost
   Classification: Logistic Regression, Decision Tree, Random Forest,
                SVM, k-NN, Naive Bayes, XGBoost

   It saves one .pkl file per algorithm, plus:
     - cleaned_attendance.csv
     - label_encoder.pkl
     - regression_results.csv
     - classification_results.csv

3. Run the Streamlit app:
   streamlit run app.py

   The app loads every saved model, shows the full evaluation/comparison
   tables, and predicts attendance with every regression and classification
   model side by side (the best model per task, based on the notebook's
   evaluation metrics, is marked with a star). It never retrains anything.

## Files in this folder
- final_notebook.ipynb   - full data cleaning, EDA, feature engineering, model training/evaluation
- app.py                 - Streamlit dashboard: EDA, model comparison, and live attendance prediction
- attendense.csv         - the original attendance dataset (unchanged)
- requirements.txt       - Python dependencies

Running the notebook also creates the cleaned CSV, one trained model file per
algorithm, the label encoder, and the two results CSVs listed above - these
are generated output, not source files, so they aren't included in the zip.
