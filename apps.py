import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def predict_burnout(data):
    # Replace with your trained classifier
    return np.random.choice(["Low", "Medium", "High"])

def predict_productivity(data):
    # Replace with your trained regression model
    return round(np.random.uniform(2, 7), 2)

def assign_cluster(data):
    # Replace with your clustering model
    return np.random.choice([0, 1, 2])

def generate_schedule(student_type, tasks):
    # Simple mock: assign time slots depending on type
    schedule = {}
    if student_type == 0:
        duration = 1
    elif student_type == 1:
        duration = 2
    else:
        duration = 1.5
    time = 9
    for task in tasks:
        schedule[task] = f"{time}:00 - {time+duration}:00"
        time += duration
    return schedule

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Prediction", "Visualization", "Scheduler"])

# -----------------------------
# HOME PAGE
# -----------------------------
if page == "Home":
    st.title("Student Productivity & Burnout Analysis System")
    st.write("""
    Welcome! This system helps students predict burnout risk, productivity scores, 
    identify student types, visualize patterns, and generate task schedules.
    """)

# -----------------------------
# PREDICTION PAGE
# -----------------------------
elif page == "Prediction":
    st.title("Predictions")
    st.write("Input your behavioural data to get predictions:")

    with st.form("student_form"):

        age = st.number_input("Age", 10, 70, 25)
        gender = st.selectbox("Gender", ["Male", "Female"])

        daily_social_media_time = st.slider("Daily social media time (hours)", 0.0, 12.0, 3.0)
        work_hours = st.slider("Work hours per day", 0.0, 16.0, 6.0)

        actual_productivity_score = st.slider("Self-rated productivity (1–10)", 1.0, 10.0, 5.0)
        stress_level = st.slider("Stress level (1–10)", 1, 10, 5)

        sleep_hours = st.slider("Average sleep hours", 0.0, 12.0, 7.0)
        breaks_during_work = st.slider("Breaks during work (times/day)", 0, 20, 2)

        uses_focus_apps = st.selectbox("Uses focus apps?", [True, False])
        has_digital_wellbeing_enabled = st.selectbox("Digital Wellbeing enabled?", [True, False])

        coffee_consumption_per_day = st.slider("Coffee cups per day", 0, 10, 1)
        days_feeling_burnout_per_month = st.slider("Burnout days per month", 0, 30, 5)

        weekly_offline_hours = st.slider("Weekly offline hours", 0.0, 50.0, 10.0)
        job_satisfaction_score = st.slider("Job satisfaction score (1–10)", 1.0, 10.0, 7.0)

        submitted = st.form_submit_button("Predict")

    if submitted:
        # Create a dataframe with all inputs
        data = pd.DataFrame({
            "age": [age],
            "gender": [gender],
            "daily_social_media_time": [daily_social_media_time],
            "work_hours_per_day": [work_hours],
            "actual_productivity_score": [actual_productivity_score],
            "stress_level": [stress_level],
            "sleep_hours": [sleep_hours],
            "breaks_during_work": [breaks_during_work],
            "uses_focus_apps": [uses_focus_apps],
            "has_digital_wellbeing_enabled": [has_digital_wellbeing_enabled],
            "coffee_consumption_per_day": [coffee_consumption_per_day],
            "days_feeling_burnout_per_month": [days_feeling_burnout_per_month],
            "weekly_offline_hours": [weekly_offline_hours],
            "job_satisfaction_score": [job_satisfaction_score],
        })

        # Predictions
        burnout = predict_burnout(data)
        productivity = predict_productivity(data)
        student_type = assign_cluster(data)

        st.success(f"Predicted Burnout Level: {burnout}")
        st.success(f"Predicted Productivity Score: {productivity}")
        st.success(f"Assigned Student Type (Cluster): {student_type}")

# -----------------------------
# VISUALIZATION PAGE
# -----------------------------
elif page == "Visualization":
    st.title("Cluster Visualizations")
    st.write("Visualize student types and feature importance:")

    
# -----------------------------
# SCHEDULER PAGE
# -----------------------------
elif page == "Scheduler":
    st.title("Task Scheduler")
    st.write("Generate a task schedule based on your student type and tasks:")

    student_type = st.selectbox("Select your student type (cluster)", [0, 1, 2])
    tasks_input = st.text_area("Enter your tasks (one per line)")
    tasks = [t.strip() for t in tasks_input.split("\n") if t.strip()]

    if st.button("Generate Schedule"):
        schedule = generate_schedule(student_type, tasks)
        st.subheader("Generated Schedule")
        for task, time in schedule.items():
            st.write(f"{task}: {time}")
