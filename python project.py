"""Student Performance Dashboard — run with: streamlit run 'python project.py'"""
import json
from datetime import datetime
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

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


def configured_auth():
    try:
        auth = st.secrets.get("auth", {})
        return bool(auth.get("client_id") and auth.get("client_secret") and auth.get("server_metadata_url"))
    except (StreamlitSecretNotFoundError, KeyError):
        return False


def current_user():
    if not configured_auth() or not st.user.is_logged_in:
        return None
    email = str(getattr(st.user, "email", "")).strip().lower()
    try:
        admins = {item.strip().lower() for item in st.secrets.get("app", {}).get("admin_emails", [])}
    except (StreamlitSecretNotFoundError, KeyError, TypeError):
        admins = set()
    return {"email": email, "name": getattr(st.user, "name", email), "is_admin": email in admins}


def pdf_report(record):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    output = BytesIO()
    doc = SimpleDocTemplate(output, pagesize=letter, rightMargin=.7*inch, leftMargin=.7*inch)
    styles = getSampleStyleSheet()
    rows = [["Student", record.get("student_name", "Student")],
            ["Student ID", record.get("student_id", "—")],
            ["Predicted score", f"{record.get('predicted_score', 0):.2f} / 100"],
            ["Performance band", record.get("category", "—")],
            ["Support indicator", record.get("risk", "—")],
            ["Change from baseline", f"{record.get('growth', 0):+.2f} points"],
            ["Suggested focus", record.get("area_of_improvement", "—")]]
    table = Table(rows, colWidths=[1.8*inch, 4.7*inch])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef2ff")),
                               ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                               ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#d7dce5")),
                               ("PADDING", (0, 0), (-1, -1), 9)]))
    story = [Paragraph("Student Performance Report", styles["Title"]),
             Paragraph(datetime.now(ZoneInfo("Asia/Kolkata")).strftime("Generated %d %b %Y, %I:%M %p IST"), styles["Normal"]),
             Spacer(1, 18), table, Spacer(1, 18),
             Paragraph("Use this estimate to support learning conversations. It is not a final academic or administrative decision.", styles["BodyText"])]
    doc.build(story)
    return output.getvalue()


def assistant_reply(prompt, data):
    try:
        settings = st.secrets.get("app", {})
        api_key = settings.get("openai_api_key", "")
        model = settings.get("openai_model", "gpt-4o-mini")
    except (StreamlitSecretNotFoundError, KeyError):
        api_key, model = "", "gpt-4o-mini"
    if api_key:
        body = json.dumps({"model": model, "messages": [
            {"role": "system", "content": "Give concise, supportive study guidance. Do not diagnose students or make high-stakes decisions. "
             f"Aggregate dataset: {len(data)} rows, average score {data.final_marks.mean():.1f}, average attendance {data.attendance.mean():.1f}%."},
            {"role": "user", "content": prompt}], "temperature": .4}).encode("utf-8")
        req = Request("https://api.openai.com/v1/chat/completions", data=body,
                      headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))["choices"][0]["message"]["content"].strip()
        except (HTTPError, URLError, TimeoutError, KeyError, ValueError) as error:
            return f"The configured AI service could not respond ({error.__class__.__name__}). Please try again."
    text = prompt.lower()
    if "attend" in text:
        return f"Average attendance is {data.attendance.mean():.1f}%. Discuss barriers and agree on a realistic attendance plan."
    if "risk" in text or "support" in text:
        share = (data.final_marks < 55).mean() * 100
        return f"{share:.1f}% of the loaded rows are below 55. Treat this as a prompt for a supportive check-in, not a decision."
    if any(word in text for word in ("score", "marks", "average", "performance")):
        return f"The loaded dataset's mean final score is {data.final_marks.mean():.1f}/100. Compare attendance and assessment scores in the dashboard."
    return "I can explain the dashboard, suggest study routines, or discuss supportive check-ins. Add an OpenAI API key in secrets for open-ended AI replies."


def main():
    st.set_page_config(page_title="Student Performance Dashboard", page_icon=":material/school:", layout="wide")
    auth_enabled = configured_auth()
    if auth_enabled and not st.user.is_logged_in:
        st.title("Sign in to Student Insights")
        st.caption("Use your configured identity provider to continue.")
        if st.button("Sign in", type="primary", icon=":material/login:"):
            st.login()
        st.stop()
    user = current_user()
    if DATA.exists():
        data = pd.read_csv(DATA)
    else:
        _, data, _ = train()

    # Visitor records are held only in the active browser session.
    st.session_state.setdefault("prediction_history", [])
    st.session_state.setdefault("student_records", {})
    st.session_state.setdefault("chat_messages", [])
    st.sidebar.title("Student Insights", icon=":material/school:")
    if auth_enabled:
        st.sidebar.caption(f"Signed in as {user['name']}")
        if st.sidebar.button("Sign out", icon=":material/logout:"):
            st.logout()
    else:
        st.sidebar.info("Demo mode · configure OIDC to enable sign-in")
    pages = ["Dashboard", "Individual Prediction", "Student Records", "PDF Reports",
             "AI Chatbox", "Batch CSV", "History"]
    if not auth_enabled or (user and user["is_admin"]):
        pages += ["Model Analytics", "Admin Console"]
    page = st.sidebar.radio("Go to", pages)
    if page == "Dashboard":
        st.title("Student performance dashboard", icon=":material/school:")
        st.caption("A planner-inspired overview of cohort progress, study habits, and supportive follow-up.")
        with st.sidebar:
            st.markdown("#### Dashboard filters")
            genders = sorted(data["gender"].dropna().astype(str).unique())
            selected_genders = st.multiselect("Gender", genders, default=genders)
            score_range = st.slider("Final score range", 0, 100, (0, 100))
            attendance_range = st.slider("Attendance range", 0, 100, (0, 100))
        filtered = data[data["gender"].astype(str).isin(selected_genders)]
        filtered = filtered[filtered["final_marks"].between(*score_range)]
        filtered = filtered[filtered["attendance"].between(*attendance_range)].copy()
        if filtered.empty:
            st.info("No rows match the selected filters.")
        filtered["performance_band"] = filtered["final_marks"].map(category)
        band_order = ["Poor", "Average", "Good", "Excellent"]
        band_colors = {"Poor": "#E17878", "Average": "#D9A441", "Good": "#3D9D8F", "Excellent": "#6750D8"}
        cohort_size = len(filtered)
        mean_score = float(filtered["final_marks"].mean()) if cohort_size else 0
        median_score = float(filtered["final_marks"].median()) if cohort_size else 0
        pass_rate = float(filtered["final_marks"].ge(50).mean() * 100) if cohort_size else 0
        mean_attendance = float(filtered["attendance"].mean()) if cohort_size else 0
        support_count = int(filtered["final_marks"].lt(55).sum()) if cohort_size else 0
        support_share = support_count / cohort_size * 100 if cohort_size else 0
        st.subheader("Cohort overview")
        kpis = st.columns(3)
        kpis[0].metric("Students in view", f"{cohort_size:,}", border=True)
        kpis[1].metric("Average final score", f"{mean_score:.1f}/100" if cohort_size else "—", border=True)
        kpis[2].metric("Median final score", f"{median_score:.1f}/100" if cohort_size else "—", border=True)
        kpis = st.columns(3)
        kpis[0].metric("Pass rate · 50+", f"{pass_rate:.1f}%" if cohort_size else "—", border=True)
        kpis[1].metric("Average attendance", f"{mean_attendance:.1f}%" if cohort_size else "—", border=True)
        kpis[2].metric("Below 55 · review", f"{support_count} ({support_share:.1f}%)" if cohort_size else "—", border=True)
        history = st.session_state.prediction_history
        predictions = len({row["student_id"] for row in history})
        prediction_avg = np.mean([row["predicted_score"] for row in history]) if history else 0
        st.caption(f"{predictions} students predicted in this session · prediction average {prediction_avg:.1f}/100 · {len(data):,} training rows")
        if cohort_size:
            left, right = st.columns(2)
            with left.container(border=True):
                st.markdown("**Final score distribution**")
                fig = px.histogram(filtered, x="final_marks", color="performance_band", nbins=18,
                                   category_orders={"performance_band": band_order}, color_discrete_map=band_colors,
                                   labels={"final_marks": "Final score", "performance_band": "Band"}, opacity=.88)
                fig.add_vline(x=mean_score, line_dash="dash", line_color="#34495e",
                              annotation_text=f"Mean {mean_score:.1f}", annotation_position="top")
                fig.update_layout(barmode="overlay", yaxis_title="Students", margin=dict(t=20, b=10), legend_title_text="Band")
                st.plotly_chart(fig, width="stretch", key="dashboard_distribution",
                                alt="Final score distribution grouped by performance band")
            with right.container(border=True):
                st.markdown("**Attendance and final score**")
                fig = px.scatter(filtered, x="attendance", y="final_marks", color="performance_band",
                                 size="study_hours", size_max=18,
                                 hover_data={"student_id": True, "gender": True, "study_hours": ":.1f"},
                                 category_orders={"performance_band": band_order}, color_discrete_map=band_colors,
                                 labels={"attendance": "Attendance (%)", "final_marks": "Final score",
                                         "performance_band": "Band", "study_hours": "Weekly study hours"})
                fig.update_layout(xaxis_range=[0, 100], yaxis_range=[0, 100], margin=dict(t=20, b=10), legend_title_text="Band")
                st.plotly_chart(fig, width="stretch", key="dashboard_attendance",
                                alt="Final score versus attendance; bubble size represents weekly study hours")
            left, right = st.columns(2)
            with left.container(border=True):
                st.markdown("**Performance bands**")
                counts = (filtered["performance_band"].value_counts().reindex(band_order, fill_value=0)
                          .rename_axis("Performance band").reset_index(name="Students"))
                fig = px.bar(counts, x="Performance band", y="Students", color="Performance band",
                             category_orders={"Performance band": band_order}, color_discrete_map=band_colors)
                fig.update_layout(showlegend=False, margin=dict(t=15, b=10))
                st.plotly_chart(fig, width="stretch", key="dashboard_bands",
                                alt="Number of students in each performance band")
            with right.container(border=True):
                st.markdown("**Assessment score profile**")
                columns = ["previous_marks", "assignment_score", "midterm_score", "internal_score"]
                labels = {"previous_marks": "Previous marks", "assignment_score": "Assignment",
                          "midterm_score": "Midterm", "internal_score": "Internal"}
                assessment = filtered[columns].rename(columns=labels).melt(var_name="Assessment", value_name="Score")
                fig = px.box(assessment, x="Assessment", y="Score", color="Assessment", points=False,
                             color_discrete_map={"Previous marks": "#6750D8", "Assignment": "#3D9D8F",
                                                 "Midterm": "#D9A441", "Internal": "#E17878"},
                             labels={"Score": "Score (out of 100)"})
                fig.update_layout(showlegend=False, yaxis_range=[0, 100], margin=dict(t=15, b=10))
                st.plotly_chart(fig, width="stretch", key="dashboard_assessment",
                                alt="Score ranges for previous marks, assignment, midterm, and internal assessments")
            st.subheader("Performance analysis")
            factors = ["attendance", "study_hours", "previous_marks", "assignment_score", "midterm_score",
                       "internal_score", "participation", "previous_gpa"]
            corr = filtered[factors + ["final_marks"]].corr(numeric_only=True)["final_marks"].drop("final_marks").dropna()
            left, right = st.columns(2)
            with left.container(border=True):
                if not corr.empty:
                    strongest = corr.abs().idxmax()
                    names = {"attendance": "Attendance", "study_hours": "Weekly study hours", "previous_marks": "Previous marks",
                             "assignment_score": "Assignment", "midterm_score": "Midterm", "internal_score": "Internal",
                             "participation": "Participation", "previous_gpa": "Previous GPA"}
                    st.metric(f"Strongest score relationship · {names[strongest]}", f"r = {corr[strongest]:+.2f}")
                    st.caption("Correlation describes this dataset; it does not establish cause.")
                else:
                    st.caption("Not enough variation in this filtered group to calculate score relationships.")
            with right.container(border=True):
                low = filtered.loc[filtered.attendance < 75, "final_marks"]
                regular = filtered.loc[filtered.attendance >= 75, "final_marks"]
                gap = regular.mean() - low.mean() if len(low) and len(regular) else None
                st.metric("Average score gap · attendance 75%+ vs below 75%",
                          f"{gap:+.1f} points" if gap is not None else "Not enough rows")
                st.caption(f"{len(low):,} rows below 75% attendance · {len(regular):,} at or above 75%.")
            st.subheader("Students for a supportive check-in")
            risk = filtered.loc[filtered.final_marks < 55,
                                ["student_id", "attendance", "study_hours", "final_marks"]].sort_values("final_marks")
            if risk.empty:
                st.success("No students in this filtered view are below the 55 point review threshold.")
            else:
                st.dataframe(risk.head(10), hide_index=True, width="stretch",
                             alt="Students with final marks below the support review threshold")
        st.caption(f"Showing {cohort_size:,} of {len(data):,} training rows. Indicators are prompts for review, not decisions.")
    elif page == "Individual Prediction":
        st.title("Individual Prediction")
        st.caption("Enter a student's details and assessment scores to estimate final performance.")
        with st.form("prediction"):
            a, b, c, d = st.columns(4)
            student_name = a.text_input("Student name", placeholder="Enter full name")
            student_id = b.text_input("Student ID", "STU1001")
            age = c.number_input("Age", 10, 100, 20)
            gender = d.selectbox("Gender", GENDERS)
            st.markdown("#### Academic record")
            a, b, c = st.columns(3)
            previous_marks = a.number_input("Previous marks (%)", 0., 100., 75., step=1.)
            previous_gpa = b.number_input("Previous GPA", 0., 10., 7.5, step=.1)
            failed_subjects = c.number_input("Failed subjects", 0, 20, 0)
            st.markdown("#### Current assessments")
            a, b, c = st.columns(3)
            attendance = a.number_input("Attendance (%)", 0., 100., 85., step=1.)
            study_hours = b.number_input("Weekly study hours", 0., 168., 15., step=1.)
            participation = c.number_input("Participation (%)", 0., 100., 80., step=1.)
            a, b, c = st.columns(3)
            assignment_score = a.number_input("Assignment score (%)", 0., 100., 80., step=1.)
            midterm_score = b.number_input("Midterm score (%)", 0., 100., 70., step=1.)
            internal_score = c.number_input("Internal score (%)", 0., 100., 72., step=1.)
            st.caption("Predictions and records stay in this browser session.")
            submit = st.form_submit_button("✨ Predict performance", width="stretch")
        if submit:
            student_id = str(student_id).strip() or "STU0000"
            payload = dict(student_id=student_id, age=age, gender=gender, attendance=attendance, study_hours=study_hours,
                previous_marks=previous_marks, assignment_score=assignment_score, midterm_score=midterm_score,
                internal_score=internal_score, participation=participation, previous_gpa=previous_gpa, failed_subjects=failed_subjects)
            score, label, risk = predict(payload)
            student_name = str(student_name).strip() or student_id
            old = st.session_state.student_records.get(student_id)
            growth = round(score - (old["predicted_score"] if old else previous_marks), 2)
            focus = min({"Attendance": attendance, "Assignment": assignment_score, "Midterm": midterm_score,
                         "Internal": internal_score, "Participation": participation}, key=lambda k: {"Attendance": attendance,
                         "Assignment": assignment_score, "Midterm": midterm_score, "Internal": internal_score,
                         "Participation": participation}[k])
            record = {"student_id": student_id, "student_name": student_name, "predicted_score": score,
                      "growth": growth, "area_of_improvement": focus, "category": label, "risk": risk}
            st.session_state.student_records[student_id] = record
            st.session_state.prediction_history.insert(0, {
                "student_id": student_id,
                "student_name": student_name,
                "predicted_score": score,
                "category": label,
                "risk": risk,
            })
            st.subheader("Prediction result")
            a, b, c = st.columns(3)
            a.metric("Predicted score", f"{score}/100")
            b.metric("Performance band", label)
            c.metric("Support indicator", risk)
            st.metric("Growth from baseline", f"{growth:+.2f} points")
            st.caption(f"Suggested improvement focus: {focus}")
    elif page == "Student Records":
        st.title("Student Records")
        records = st.session_state.student_records
        if records:
            st.dataframe(pd.DataFrame(records.values()), hide_index=True, width="stretch",
                         alt="Session student records with scores, growth, and support indicators")
            a, b = st.columns(2, vertical_alignment="bottom")
            selected = a.selectbox("Choose a student to delete", list(records),
                                   format_func=lambda sid: f"{records[sid]['student_name']} ({sid})")
            if b.button("Delete student", width="stretch"):
                records.pop(selected, None)
                st.session_state.prediction_history = [r for r in st.session_state.prediction_history if r["student_id"] != selected]
                st.rerun()
            if st.button("Delete all student records"):
                records.clear(); st.session_state.prediction_history.clear(); st.rerun()
        else:
            st.info("Create an individual prediction to add a student record.")
    elif page == "PDF Reports":
        st.title("PDF Reports")
        records = st.session_state.student_records
        if records:
            selected = st.selectbox("Choose a student", list(records),
                                    format_func=lambda sid: f"{records[sid]['student_name']} ({sid})")
            record = records[selected]
            st.metric("Predicted score", f"{record['predicted_score']:.2f}/100")
            st.caption(f"{record['category']} · {record['risk']} · Suggested focus: {record['area_of_improvement']}")
            st.download_button("Download PDF report", pdf_report(record),
                               file_name=f"student_report_{selected}.pdf", mime="application/pdf", width="stretch")
        else:
            st.info("Create an individual prediction to prepare a PDF report.")
    elif page == "AI Chatbox":
        st.title("AI Chatbox")
        try:
            connected = bool(st.secrets.get("app", {}).get("openai_api_key", ""))
        except StreamlitSecretNotFoundError:
            connected = False
        st.caption("OpenAI is connected. Chat prompts are sent to the service; avoid identifying student details." if connected
                   else "Local study helper is active. Add an OpenAI key in secrets for open-ended AI responses.")
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        prompt = st.chat_input("Ask about study strategies or dashboard results")
        if prompt:
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                answer = assistant_reply(prompt, data)
                st.markdown(answer)
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})
        if st.session_state.chat_messages and st.button("Clear chat"):
            st.session_state.chat_messages.clear(); st.rerun()
    elif page == "Batch CSV":
        st.title("Batch CSV Prediction"); upload = st.file_uploader("Upload a CSV with student feature columns", type="csv")
        if upload:
            batch = pd.read_csv(upload)
            if "student_id" not in batch: batch.insert(0, "student_id", [f"STU{i+1}" for i in range(len(batch))])
            rows = [predict(row.to_dict()) for _, row in batch.iterrows()]
            output = pd.DataFrame(rows, columns=["Predicted Score", "Category", "Risk"]); output.insert(0, "student_id", batch.student_id.astype(str))
            st.dataframe(output, width="stretch", alt="Batch student prediction results")
            st.download_button("Download results", output.to_csv(index=False), "predictions.csv", width="stretch")
    elif page == "History":
        st.title("Prediction History")
        student_id = st.text_input("Filter by Student ID")
        history = [row for row in st.session_state.prediction_history
                   if not student_id.strip() or row["student_id"] == student_id.strip()]
        st.dataframe(history if history else pd.DataFrame({"Message": ["No predictions in this session."]}),
                     width="stretch", alt="Prediction history")
    elif page == "Model Analytics":
        if auth_enabled and not (user and user["is_admin"]):
            st.error("Administrator access is required."); st.stop()
        st.title("Model Analytics")
        if st.button("Retrain model"):
            _, data, metrics = train(); get_model.clear(); st.success("Model retrained.")
        elif DATA.exists(): data = pd.read_csv(DATA); metrics = evaluate(get_model(), data)
        else: _, data, metrics = train()
        st.json(metrics)
    else:
        st.title("Admin Console")
        if not auth_enabled:
            st.info("Sign-in is not configured. Add an OIDC provider and administrator email to Streamlit secrets.")
            st.code('[auth]\nredirect_uri = "http://localhost:8501/oauth2callback"\ncookie_secret = "<long-random-secret>"\nclient_id = "<client-id>"\nclient_secret = "<client-secret>"\nserver_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"\n\n[app]\nadmin_emails = ["admin@example.com"]', language="toml")
        elif user and user["is_admin"]:
            st.success(f"Administrator access · {user['email']}")
            st.metric("Training rows", len(data))
            st.dataframe(data.head(20), hide_index=True, width="stretch", alt="Sample of the training data")
            if st.button("Retrain model", type="primary"):
                _, _, metrics = train(); get_model.clear()
                st.success("Model retrained."); st.json(metrics)
        else:
            st.error("Administrator access is required.")
    st.info("Predictions are academic-support indicators, not final administrative decisions.")


if __name__ == "__main__": main()
