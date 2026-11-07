"""Student Performance Dashboard — run with: streamlit run 'python project.py'"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

BASE = Path(__file__).resolve().parent
DATA, MODEL, METRICS = (BASE / name for name in
                        ("students.csv", "student_model.joblib", "metrics.json"))
GENDERS = ["Female", "Male", "Other"]
DEFAULTS = dict(student_id="STU0000", age=20, gender="Other", attendance=75., study_hours=10.,
                previous_marks=70., assignment_score=75., midterm_score=70., internal_score=70.,
                participation=70., previous_gpa=7., failed_subjects=0)
NUMERIC = ["age", "attendance", "study_hours", "previous_marks", "assignment_score", "midterm_score",
           "internal_score", "participation", "previous_gpa", "failed_subjects", "average_assessment",
           "academic_average", "attendance_risk"]


def features(frame):
    frame = frame.copy()
    for col in ("assignment_score", "midterm_score", "internal_score", "previous_marks", "attendance"):
        if col not in frame: frame[col] = DEFAULTS.get(col, 0.)
    frame["average_assessment"] = frame[["assignment_score", "midterm_score", "internal_score"]].mean(axis=1)
    frame["academic_average"] = (frame["previous_marks"] + frame["average_assessment"]) / 2
    frame["attendance_risk"] = (frame["attendance"] < 75).astype(int)
    return frame


class StudentRegressor:
    """NumPy least-squares regressor with fixed gender columns."""
    def matrix(self, frame):
        frame = frame.copy()
        for col in NUMERIC:
            if col not in frame: frame[col] = DEFAULTS.get(col, 0.)
        if "gender" not in frame: frame["gender"] = "Other"
        gender = pd.get_dummies(frame["gender"].fillna("Other").astype(str), dtype=float).reindex(columns=GENDERS, fill_value=0.)
        return np.column_stack([frame[NUMERIC].astype(float), gender])

    def fit(self, frame, target):
        x = self.matrix(frame)
        self.coefficients = np.linalg.lstsq(np.c_[np.ones(len(x)), x], target, rcond=None)[0]
        self.feature_names = NUMERIC + [f"gender_{g.lower()}" for g in GENDERS]
        weights = np.abs(self.coefficients[1:]); self.feature_importances = weights / weights.sum() if weights.sum() else weights
        return self

    def predict(self, frame):
        x = self.matrix(frame)
        return np.c_[np.ones(len(x)), x] @ self.coefficients

    def to_state(self):
        return {"coefficients": self.coefficients, "feature_names": self.feature_names,
                "feature_importances": self.feature_importances}

    @classmethod
    def from_state(cls, state):
        model = cls()
        for key, value in state.items(): setattr(model, key, value)
        return model


def save_model(model): joblib.dump(model.to_state(), MODEL)
def load_model(): return StudentRegressor.from_state(joblib.load(MODEL))


def synthetic_data(n=500, seed=42):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame({"student_id": [f"STU{1000+i}" for i in range(n)], "age": rng.integers(18, 25, n),
        "gender": rng.choice(GENDERS, n), "attendance": np.clip(rng.normal(80, 15, n), 40, 100),
        "study_hours": np.clip(rng.normal(15, 5, n), 1, 40), "previous_marks": np.clip(rng.normal(70, 12, n), 35, 100),
        "assignment_score": np.clip(rng.normal(75, 10, n), 40, 100), "midterm_score": np.clip(rng.normal(68, 15, n), 30, 100),
        "internal_score": np.clip(rng.normal(72, 12, n), 35, 100), "participation": np.clip(rng.normal(70, 20, n), 20, 100),
        "failed_subjects": rng.choice([0, 1, 2, 3], n, p=[.7, .15, .1, .05])})
    data["previous_gpa"] = np.clip(data.previous_marks / 10 + rng.normal(0, .3, n), 1, 10)
    data["final_marks"] = np.clip(.25*data.previous_marks + .25*data.midterm_score + .2*data.assignment_score
        + .15*data.attendance + .2*data.study_hours - 3*data.failed_subjects + rng.normal(0, 3, n), 20, 100)
    return data


def evaluate(model, data):
    actual = data["final_marks"].to_numpy(); error = actual - model.predict(features(data))
    variance = ((actual - actual.mean()) ** 2).sum()
    return {"MAE": round(float(np.abs(error).mean()), 2), "RMSE": round(float(np.sqrt((error**2).mean())), 2),
            "R2": round(float(1 - (error**2).sum() / variance), 3) if variance else 0.}


def train():
    data = pd.read_csv(DATA) if DATA.exists() else synthetic_data()
    data.to_csv(DATA, index=False); model = StudentRegressor().fit(features(data), data["final_marks"].to_numpy())
    metrics = evaluate(model, data); save_model(model); METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return model, data, metrics


@st.cache_resource
def get_model(): return load_model() if MODEL.exists() else train()[0]


def category(score): return next((label for limit, label in [(50, "Poor"), (65, "Average"), (80, "Good")] if score < limit), "Excellent")


def normalize(payload):
    result = {}
    for key, default in DEFAULTS.items():
        value = payload.get(key, default)
        if key in ("student_id", "gender"):
            value = str(value or default).strip() or default
            if key == "gender": value = next((g for g in GENDERS if g.lower() == value.lower()), "Other")
        else:
            try: value = float(value) if pd.notna(value) and str(value).strip() else default
            except (TypeError, ValueError): value = default
            if key in ("age", "failed_subjects"): value = int(value)
        result[key] = value
    return result


def predict(payload):
    cleaned = normalize(payload)
    score = round(float(np.clip(get_model().predict(features(pd.DataFrame([cleaned])))[0], 0, 100)), 2)
    return score, category(score), "At Risk" if score < 55 else "Not At Risk"


def main():
    st.set_page_config(page_title="Student Performance Dashboard", page_icon="🎓", layout="wide")
    # Keep prediction history in the visitor's session only. Public visitors' data
    # must not be persisted to disk or shared with other visitors.
    st.session_state.setdefault("prediction_history", [])
    page = st.sidebar.radio("Navigation", ["Dashboard", "Individual Prediction", "Batch CSV", "History", "Model Analytics"])
    if page == "Dashboard":
        _, data, _ = train() if not DATA.exists() else (get_model(), pd.read_csv(DATA), None)
        st.title("🎓 Student Performance Dashboard")
        history = st.session_state.prediction_history
        count = len({row["student_id"] for row in history})
        average = np.mean([row["predicted_score"] for row in history]) if history else 0
        a, b, c = st.columns(3); a.metric("Students Predicted This Session", count); b.metric("Session Average", f"{average:.1f}"); c.metric("Training Rows", len(data))
        a, b = st.columns(2)
        a.plotly_chart(px.histogram(data, x="final_marks", title="Final Marks Distribution"), use_container_width=True)
        b.plotly_chart(px.scatter(data, x="attendance", y="final_marks", title="Attendance vs Final Marks"), use_container_width=True)
    elif page == "Individual Prediction":
        st.title("Individual Prediction")
        with st.form("prediction"):
            a, b, c = st.columns(3)
            student_id = a.text_input("Student ID", "STU1001")
            b.caption("Predictions stay in this browser session and are cleared when it ends.")
            gender = c.selectbox("Gender", GENDERS); age = a.number_input("Age", 10, 100, 20)
            attendance = b.slider("Attendance (%)", 0., 100., 85.); study_hours = c.number_input("Weekly Study Hours", 0., 168., 15.)
            previous_marks = a.slider("Previous Marks", 0., 100., 75.); assignment_score = b.slider("Assignment Score", 0., 100., 80.)
            midterm_score = c.slider("Midterm Score", 0., 100., 70.); internal_score = a.slider("Internal Score", 0., 100., 72.)
            participation = b.slider("Participation", 0., 100., 80.); previous_gpa = c.number_input("Previous GPA", 0., 10., 7.5)
            failed_subjects = a.number_input("Failed Subjects", 0, 20, 0); submit = st.form_submit_button("Predict")
        if submit:
            payload = dict(student_id=student_id, age=age, gender=gender, attendance=attendance, study_hours=study_hours,
                previous_marks=previous_marks, assignment_score=assignment_score, midterm_score=midterm_score,
                internal_score=internal_score, participation=participation, previous_gpa=previous_gpa, failed_subjects=failed_subjects)
            score, label, risk = predict(payload)
            st.session_state.prediction_history.insert(0, {
                "student_id": student_id,
                "predicted_score": score,
                "category": label,
                "risk": risk,
            })
            a, b, c = st.columns(3); a.metric("Predicted Score", f"{score}/100"); b.metric("Category", label); c.metric("Risk", risk)
    elif page == "Batch CSV":
        st.title("Batch CSV Prediction"); upload = st.file_uploader("Upload a CSV with student feature columns", type="csv")
        if upload:
            batch = pd.read_csv(upload)
            if "student_id" not in batch: batch.insert(0, "student_id", [f"STU{i+1}" for i in range(len(batch))])
            rows = [predict(row.to_dict()) for _, row in batch.iterrows()]
            output = pd.DataFrame(rows, columns=["Predicted Score", "Category", "Risk"]); output.insert(0, "student_id", batch.student_id.astype(str))
            st.dataframe(output, use_container_width=True); st.download_button("Download results", output.to_csv(index=False), "predictions.csv")
    elif page == "History":
        st.title("Prediction History")
        student_id = st.text_input("Filter by Student ID")
        history = [row for row in st.session_state.prediction_history
                   if not student_id.strip() or row["student_id"] == student_id.strip()]
        st.dataframe(history if history else pd.DataFrame({"Message": ["No predictions in this session."]}), use_container_width=True)
    else:
        st.title("Model Analytics")
        if st.button("Retrain model"):
            _, data, metrics = train(); get_model.clear(); st.success("Model retrained.")
        elif DATA.exists(): data = pd.read_csv(DATA); metrics = evaluate(get_model(), data)
        else: _, data, metrics = train()
        st.json(metrics)
    st.info("Predictions are academic-support indicators, not final administrative decisions.")


if __name__ == "__main__": main()
