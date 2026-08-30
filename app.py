import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import date, time

# ============================================================
# Step 1: Page configuration
# ============================================================
st.set_page_config(
    page_title="Classroom Attendance Prediction",
    page_icon="\U0001F4CA",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent


def to_dense_array(x):
    """Convert a sparse matrix to a dense array. This must exist here with this
    exact name because a saved model pipeline may reference this function by
    name when it is un-pickled (it is used for the Naive Bayes step in the
    notebook). It is not called directly in this file - joblib/pickle needs to
    find it in this module in order to load that pipeline."""
    return x.toarray() if hasattr(x, "toarray") else x

# ============================================================
# Step 2: Load the cleaned dataset produced by the notebook
# ============================================================
DATA_FILE = APP_DIR / "cleaned_attendance.csv"

if not DATA_FILE.exists():
    st.error(
        "cleaned_attendance.csv not found. Run final_notebook.ipynb first - "
        "its data-cleaning step saves this file into the same folder as this app."
    )
    st.stop()


@st.cache_data
def load_dataset():
    data = pd.read_csv(DATA_FILE)
    data["Date"] = pd.to_datetime(data["Date"], dayfirst=True)
    return data


df = load_dataset()

# ============================================================
# Step 3: Load every trained model (already trained in the notebook).

REGRESSION_FILES = {
    "Linear Regression": "linear_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "Gradient Boosting": "gradient_model.pkl",
    "SVM": "svm_model.pkl",
    "XGBoost": "xgb_reg_model.pkl",
}

CLASSIFICATION_FILES = {
    "Logistic Regression": "logistic_model.pkl",
    "Decision Tree": "decision_tree_classifier_model.pkl",
    "Random Forest": "random_forest_classifier_model.pkl",
    "SVM": "svm_classifier_model.pkl",
    "k-NN": "knn_classifier_model.pkl",
    "Naive Bayes": "naive_bayes_model.pkl",
    "XGBoost": "xgb_classifier_model.pkl",
}

LABEL_ENCODER_FILE = APP_DIR / "label_encoder.pkl"
REG_RESULTS_FILE = APP_DIR / "regression_results.csv"
CLASS_RESULTS_FILE = APP_DIR / "classification_results.csv"


@st.cache_resource
def load_models(file_map):
    models, missing = {}, []
    for name, filename in file_map.items():
        path = APP_DIR / filename
        if path.exists():
            models[name] = joblib.load(path)
        else:
            missing.append(filename)
    return models, missing


regression_models, missing_reg_files = load_models(REGRESSION_FILES)
classification_models, missing_class_files = load_models(CLASSIFICATION_FILES)

label_encoder = None
if LABEL_ENCODER_FILE.exists():
    label_encoder = joblib.load(LABEL_ENCODER_FILE)

if not regression_models and not classification_models:
    st.error(
        "No trained .pkl model files were found. Run final_notebook.ipynb first - "
        "its last step trains and saves every model this app needs."
    )
    st.stop()


best_reg_name = None
if REG_RESULTS_FILE.exists():
    reg_results = pd.read_csv(REG_RESULTS_FILE)
    best_reg_name = reg_results.loc[reg_results["MAE"].idxmin(), "Model"]

best_class_name = None
if CLASS_RESULTS_FILE.exists():
    class_results = pd.read_csv(CLASS_RESULTS_FILE)
    best_class_name = class_results.loc[class_results["F1 Score"].idxmax(), "Model"]


# ============================================================
# Step 4: Helper functions
# ============================================================
def options(column, default="Not Available"):
    values = df[column].dropna().unique().tolist()
    values = sorted(values, key=str)
    return values if values else [default]


def align_input_to_model(model, input_data):
    """Put the input row's columns in exactly the order the trained pipeline
    expects. Every saved pipeline already includes its own preprocessing
    (imputing / scaling / one-hot encoding) - we only ever call .predict()."""
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        missing_cols = [c for c in expected if c not in input_data.columns]
        if missing_cols:
            raise ValueError(
                "The trained model expects these columns but they are missing "
                "from the input: " + ", ".join(missing_cols)
            )
        return input_data[expected]
    return input_data


def build_feature_row(
    day_of_week, lecture_number, subject, subject_code, faculty_id, semester,
    branch, section, classroom, total_students, previous_attendance, gap,
    lecture_type, internal_test, assignment_due, holiday, weather, special_event,
    faculty_experience, selected_date, start_time_value, end_time_value,
    rolling_average, consecutive_count, days_since_holiday,
):
    """Build ONE lecture's feature row, using exactly the same feature
    definitions as Steps 4-6 of the final notebook. This keeps the notebook and
    the app perfectly consistent - no different feature names, no different
    calculations, no column-mismatch errors."""

    selected_date_ts = pd.Timestamp(selected_date)
    start_stamp = pd.Timestamp.combine(selected_date_ts, start_time_value)
    end_stamp = pd.Timestamp.combine(selected_date_ts, end_time_value)
    duration = (end_stamp - start_stamp).total_seconds() / 60

    if duration <= 0:
        raise ValueError("End Time must be after Start Time.")

    semester_start = df.loc[df["Semester"] == semester, "Date"].min()
    if pd.isna(semester_start):
        semester_start = df["Date"].min()
    day_of_semester = (selected_date_ts - semester_start).days + 1

    return pd.DataFrame({
        "Day of Week": [day_of_week],
        "Lecture Number": [lecture_number],
        "Subject": [subject],
        "Subject Code": [subject_code],
        "Faculty ID": [faculty_id],
        "Semester": [semester],
        "Branch": [branch],
        "Section": [section],
        "Classroom": [classroom],
        "Total Enrolled Students": [total_students],
        "Previous Lecture Attendance": [previous_attendance],
        "Gap Since Previous Lecture": [gap],
        "Practical/Theory": [lecture_type],
        "Internal Test Week": [internal_test],
        "Assignment Due": [assignment_due],
        "Holiday Before/After": [holiday],
        "Weather": [weather],
        "Special Event": [special_event],
        "Faculty Experience": [faculty_experience],
        "Lecture Duration (Minutes)": [duration],
        "Day of Semester": [int(day_of_semester)],
        "Week Number": [int(selected_date_ts.isocalendar().week)],
        "Month": [int(selected_date_ts.month)],
        "Previous Attendance Rolling Average": [rolling_average],
        "Consecutive Lecture Count": [consecutive_count],
        "Days Since Last Holiday": [days_since_holiday],
        "Time Category": ["Morning" if start_time_value.hour < 12 else "Afternoon"],
        "Exam Week Flag": [1 if internal_test == "Yes" else 0],
        "Assignment Due Flag": [1 if assignment_due == "Yes" else 0],
        "Holiday Flag": [1 if holiday == "Yes" else 0],
        "Special Event Flag": [1 if special_event == "Yes" else 0],
    })


# ============================================================
# Step 5: Title / intro
# ============================================================
st.title("\U0001F4CA Classroom Attendance Prediction System")
st.write(
    "Predicts attendance for an upcoming lecture using historical attendance "
    "patterns and academic schedule information. Models were trained and "
    "compared in the project notebook; this app only *loads* the winning "
    "regression and classification models, it never retrains them."
)
st.divider()

# ============================================================
# Step 6: Dataset overview
# ============================================================
st.header("\U0001F4C1 Dataset Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Records", f"{len(df):,}")
c2.metric("Columns", str(df.shape[1]))
c3.metric("Average Attendance", f"{df['Attendance Percentage'].mean():.2f}%")
c4.metric("Subjects", str(df["Subject"].nunique()))

with st.expander("View a sample of the cleaned dataset"):
    st.dataframe(df.head(20), use_container_width=True)

st.caption(
    "Note: every row in this dataset has Data Source = 'Synthetic development data'. "
    "This is generated data, not data physically collected from real lectures - see "
    "the project report for what this means for the submission requirements."
)

st.divider()

# ============================================================
# Step 7: Attendance analysis
# ============================================================
st.header("\U0001F4C8 Attendance Analysis")

left, right = st.columns(2)

with left:
    st.subheader("Attendance Distribution")
    fig, ax = plt.subplots(figsize=(5, 3.3))
    ax.hist(df["Attendance Percentage"], bins=20, edgecolor="black")
    ax.set_xlabel("Attendance Percentage")
    ax.set_ylabel("Number of Lectures")
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("Attendance by Day of Week")
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    day_avg = df.groupby("Day of Week")["Attendance Percentage"].mean()
    day_avg = day_avg.reindex([d for d in day_order if d in day_avg.index])
    fig, ax = plt.subplots(figsize=(5, 3.3))
    ax.bar(day_avg.index, day_avg.values)
    ax.set_ylabel("Average Attendance (%)")
    plt.xticks(rotation=30)
    st.pyplot(fig)
    plt.close(fig)

left2, right2 = st.columns(2)

with left2:
    st.subheader("Subjects With Lowest Attendance")
    subject_avg = df.groupby("Subject")["Attendance Percentage"].mean().sort_values().head(10)
    st.dataframe(
        subject_avg.round(2).reset_index().rename(
            columns={"Attendance Percentage": "Average Attendance (%)"}
        ),
        use_container_width=True, hide_index=True,
    )

with right2:
    st.subheader("Time Slots With Consistently Low Attendance")
    slot_avg = (
        df.groupby(["Lecture Number", "Time Category"])["Attendance Percentage"]
          .mean().round(2).reset_index()
          .sort_values("Attendance Percentage")
          .rename(columns={"Attendance Percentage": "Average Attendance (%)"})
    )
    st.dataframe(slot_avg, use_container_width=True, hide_index=True)

st.subheader("Academic Condition Impact")
condition_rows = []
for column in ["Internal Test Week", "Assignment Due", "Holiday Before/After", "Special Event"]:
    grouped = df.groupby(column)["Attendance Percentage"].mean().round(2)
    for condition, value in grouped.items():
        condition_rows.append({"Condition": column, "Value": condition, "Avg Attendance (%)": value})
st.dataframe(pd.DataFrame(condition_rows), use_container_width=True, hide_index=True)
st.caption(
    "This shows the historical impact of tests, assignments, holidays, and special events. "
    "To estimate the impact for a *specific* upcoming lecture, set these fields in the "
    "prediction form below and compare the predicted attendance with them toggled Yes vs No."
)

st.divider()

# ============================================================
# Step 7B: Visual Insights (additional important graphs)
# ============================================================
st.header("\U0001F4CA Visual Insights")
st.caption(
    "Extra charts covering the patterns referenced in Key Findings and the "
    "Phase 3/4 requirements of the project (recent-history signal, time-of-day, "
    "semester trend, and feature correlation)."
)

vis_left, vis_right = st.columns(2)

with vis_left:
    st.subheader("Recent History vs Current Attendance")
    fig, ax = plt.subplots(figsize=(5, 3.3))
    ax.scatter(
        df["Previous Attendance Rolling Average"],
        df["Attendance Percentage"],
        alpha=0.25, s=10, color="#4C72B0",
    )
    ax.set_xlabel("Rolling Average of Last 3 Lectures (%)")
    ax.set_ylabel("Current Attendance (%)")
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        f"Correlation with current attendance: **{df['Previous Attendance Rolling Average'].corr(df['Attendance Percentage']):.2f}** "
        "- this is the strongest engineered predictive signal in the dataset."
    )

with vis_right:
    st.subheader("Morning vs Afternoon")
    fig, ax = plt.subplots(figsize=(5, 3.3))
    time_avg = df.groupby("Time Category")["Attendance Percentage"].mean()
    ax.bar(time_avg.index, time_avg.values, color=["#4C72B0", "#DD8452"])
    ax.set_ylabel("Average Attendance (%)")
    st.pyplot(fig)
    plt.close(fig)

vis_left2, vis_right2 = st.columns(2)

with vis_left2:
    st.subheader("Attendance by Semester")
    fig, ax = plt.subplots(figsize=(5, 3.3))
    sem_avg = df.groupby("Semester")["Attendance Percentage"].mean().sort_index()
    ax.bar(sem_avg.index.astype(str), sem_avg.values, color="#55A868")
    ax.set_xlabel("Semester")
    ax.set_ylabel("Average Attendance (%)")
    st.pyplot(fig)
    plt.close(fig)

with vis_right2:
    st.subheader("Correlation Between Numeric Features")
    numeric_cols = [
        c for c in [
            "Attendance Percentage", "Previous Lecture Attendance",
            "Previous Attendance Rolling Average", "Gap Since Previous Lecture",
            "Days Since Last Holiday", "Faculty Experience", "Total Enrolled Students",
        ] if c in df.columns
    ]
    corr_matrix = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(5, 3.8))
    im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_yticklabels(numeric_cols, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.divider()

# ============================================================
# Step 8: Key findings
# ============================================================
st.header("\U0001F4A1 Key Findings")

corr = df["Previous Lecture Attendance"].corr(df["Attendance Percentage"])
low_share = (df["Attendance Percentage"] < 50).mean() * 100

st.markdown(f"""
- Correlation between a lecture's previous-lecture attendance and its own attendance
  is **{corr:.2f}** - recent history is a genuinely useful (and non-leaking) signal.
- Only **{low_share:.1f}%** of lectures ever had attendance below 50% in this dataset,
  which is why the "Low" category almost never appears in the classification results.
- The lowest-attendance subjects, time slots, and the day-of-week / test-week / holiday
  patterns above are the strongest, data-backed factors behind attendance changes here.
""")

st.divider()

# ============================================================
# Step 9: Model comparison
# ============================================================
st.header("\U0001F916 Model Comparison")

tab1, tab2 = st.tabs(["Regression Models", "Classification Models"])

with tab1:
    if REG_RESULTS_FILE.exists():
        chart_col, table_col = st.columns([1, 1])
        with chart_col:
            st.subheader("MAE by Model")
            fig, ax = plt.subplots(figsize=(5, 3.3))
            colors = ["#DD8452" if m == best_reg_name else "#4C72B0" for m in reg_results["Model"]]
            ax.bar(reg_results["Model"], reg_results["MAE"], color=colors)
            ax.set_ylabel("MAE (lower is better)")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        with table_col:
            st.subheader("Full Results")
            st.dataframe(reg_results, use_container_width=True, hide_index=True)
        st.success(f"Best regression model (lowest MAE): **{best_reg_name}**")
    else:
        st.info("Run the notebook first to generate regression_results.csv")

with tab2:
    if CLASS_RESULTS_FILE.exists():
        chart_col, table_col = st.columns([1, 1])
        with chart_col:
            st.subheader("F1 Score by Model")
            fig, ax = plt.subplots(figsize=(5, 3.3))
            colors = ["#DD8452" if m == best_class_name else "#4C72B0" for m in class_results["Model"]]
            ax.bar(class_results["Model"], class_results["F1 Score"], color=colors)
            ax.set_ylabel("F1 Score (higher is better)")
            ax.set_ylim(0, 1)
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        with table_col:
            st.subheader("Full Results")
            st.dataframe(class_results, use_container_width=True, hide_index=True)
        st.success(f"Best classification model (highest F1 Score): **{best_class_name}**")
    else:
        st.info("Run the notebook first to generate classification_results.csv")

st.divider()

# ============================================================
# Step 10: Predict attendance for an upcoming lecture
# ============================================================
st.header("\U0001F52E Predict Attendance for an Upcoming Lecture")

st.sidebar.header("\U0001F4DA Upcoming Lecture")

selected_date = st.sidebar.date_input("Lecture Date", value=date.today(), format="DD-MM-YYYY")
start_time_value = st.sidebar.time_input("Start Time", value=time(9, 15))
end_time_value = st.sidebar.time_input("End Time", value=time(10, 15))
day_of_week = st.sidebar.selectbox("Day of Week", options("Day of Week"))
lecture_number = st.sidebar.number_input("Lecture Number", min_value=1, max_value=10, value=1, step=1)
subject = st.sidebar.selectbox("Subject", options("Subject"))
subject_code = st.sidebar.selectbox("Subject Code", options("Subject Code"))
faculty_id = st.sidebar.selectbox("Faculty ID", options("Faculty ID"))
semester = st.sidebar.selectbox("Semester", sorted(df["Semester"].unique().tolist()))
branch = st.sidebar.selectbox("Branch", options("Branch"))
section = st.sidebar.selectbox("Section", options("Section"))
classroom = st.sidebar.selectbox("Classroom", options("Classroom"))
total_students = st.sidebar.number_input(
    "Total Enrolled Students", min_value=1, max_value=500,
    value=int(df["Total Enrolled Students"].median()), step=1,
)

st.sidebar.header("\U0001F4C8 Recent Attendance History")
previous_attendance = st.sidebar.number_input(
    "Previous Lecture Attendance (%)", min_value=0.0, max_value=100.0,
    value=float(df["Previous Lecture Attendance"].median()), step=0.1,
)
rolling_average = st.sidebar.number_input(
    "Rolling Average of Last 3 Lectures (%)", min_value=0.0, max_value=100.0,
    value=float(df["Previous Attendance Rolling Average"].dropna().median()), step=0.1,
)
gap = st.sidebar.number_input(
    "Gap Since Previous Lecture (hours)", min_value=0.0, max_value=1000.0,
    value=float(df["Gap Since Previous Lecture"].median()), step=0.5,
)
consecutive_count = st.sidebar.number_input(
    "Consecutive Lecture Count (this streak)", min_value=1, max_value=10, value=1, step=1,
)
days_since_holiday = st.sidebar.number_input(
    "Days Since Last Holiday", min_value=0, max_value=365,
    value=int(df["Days Since Last Holiday"].median()), step=1,
)
faculty_experience = st.sidebar.number_input(
    "Faculty Experience (Years)", min_value=0.0, max_value=50.0,
    value=float(df["Faculty Experience"].median()), step=0.5,
)

st.sidebar.header("\U0001F393 Academic Conditions")
lecture_type = st.sidebar.selectbox("Practical/Theory", options("Practical/Theory"))
internal_test = st.sidebar.selectbox("Internal Test Week", ["No", "Yes"])
assignment_due = st.sidebar.selectbox("Assignment Due", ["No", "Yes"])
holiday = st.sidebar.selectbox("Holiday Before/After", ["No", "Yes"])
weather = st.sidebar.selectbox("Weather", options("Weather"))
special_event = st.sidebar.selectbox("Special Event", ["No", "Yes"])

predict_clicked = st.button("\U0001F52E Predict Attendance", type="primary", use_container_width=True)

if predict_clicked:
    try:
        input_row = build_feature_row(
            day_of_week, lecture_number, subject, subject_code, faculty_id, semester,
            branch, section, classroom, total_students, previous_attendance, gap,
            lecture_type, internal_test, assignment_due, holiday, weather, special_event,
            faculty_experience, selected_date, start_time_value, end_time_value,
            rolling_average, consecutive_count, days_since_holiday,
        )

        # ---- Regression predictions: one per trained algorithm ----
        st.subheader("Predicted Attendance Percentage")
        reg_cols = st.columns(len(regression_models)) if regression_models else []
        for col, (name, model) in zip(reg_cols, regression_models.items()):
            aligned = align_input_to_model(model, input_row)
            pred_value = float(model.predict(aligned)[0])
            label = f"{name} \u2b50" if name == best_reg_name else name
            col.metric(label, f"{pred_value:.1f}%")

        # ---- Classification predictions: one per trained algorithm ----
        if classification_models:
            st.subheader("Predicted Attendance Category")
            class_cols = st.columns(len(classification_models))
            for col, (name, model) in zip(class_cols, classification_models.items()):
                aligned = align_input_to_model(model, input_row)
                pred_encoded = model.predict(aligned)[0]
                pred_label = (
                    label_encoder.inverse_transform([pred_encoded])[0]
                    if label_encoder is not None else str(pred_encoded)
                )
                label = f"{name} \u2b50" if name == best_class_name else name
                col.metric(label, pred_label)

        with st.expander("\U0001F50D View exact model input row"):
            st.dataframe(input_row.astype(str), use_container_width=True, hide_index=True)

    except Exception as error:
        st.error("Prediction failed. See technical details below.")
        with st.expander("Technical Details"):
            st.exception(error)

st.divider()

# ============================================================
# Step 11: Loaded model status + footer
# ============================================================
with st.expander("\U0001F4CC Loaded Model Files"):
    loaded_rows = (
        [{"Type": "Regression", "Model": n, "File": f} for n, f in REGRESSION_FILES.items() if n in regression_models]
        + [{"Type": "Classification", "Model": n, "File": f} for n, f in CLASSIFICATION_FILES.items() if n in classification_models]
    )
    st.dataframe(pd.DataFrame(loaded_rows), use_container_width=True, hide_index=True)
    st.caption("\u2b50 marks the best model for each task, based on the evaluation metrics in the Model Comparison section above.")

    if missing_reg_files or missing_class_files:
        st.warning(
            "Missing model files (run the notebook to generate them): "
            + ", ".join(missing_reg_files + missing_class_files)
        )

st.caption("Classroom Attendance Prediction | Streamlit Deployment | Models loaded from trained .pkl files")