import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io

# -----------------------------
# 1. LOAD ALL ASSETS
# -----------------------------
@st.cache_resource
def load_models():
    reg_model = joblib.load('best_productivity_model.pkl')
    imputer = joblib.load('imputer.pkl')
    scaler = joblib.load('scaler.pkl') 
    model_cols = joblib.load('model_columns.pkl')
    cluster_model = joblib.load('best_clustering_model.pkl')
    cluster_scaler = joblib.load('cluster_scaler.pkl')
    cluster_features = joblib.load('cluster_features.pkl')
    burnout_model = joblib.load('burnout_stacking_model.pkl')
    burnout_le = joblib.load('burnout_label_encoder.pkl')
    return reg_model, imputer, scaler, model_cols, cluster_model, cluster_scaler, cluster_features, burnout_model, burnout_le

# Initialize state
if 'tasks' not in st.session_state:
    st.session_state.tasks = []

reg_model, imputer, scaler, model_cols, cluster_model, cluster_scaler, cluster_features, burnout_model, burnout_le = load_models()

cluster_names = {0: "Overwhelmed", 1: "Grinders", 2: "Zen Masters", 3: "Burnout Risk", 4: "Coasters", 5: "Restless Sleepers"}

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def get_recommendation(prod, persona, burnout):
    if burnout == "High":
        return "🚨 **Urgent:** High burnout risk. Reduce work hours and prioritize 8h sleep."
    if persona == "Grinders" and prod > 7:
        return "⚠️ **Caution:** You are a high achiever but at risk. Implement Pomodoro (5min breaks)."
    return "✅ **Stable:** Your current patterns appear sustainable. Maintain your offline hours."

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("🎯 Student Wellbeing")
page = st.sidebar.selectbox("Navigate To", ["🏠 Home", "📊 Prediction & Analysis", "📈 Global Insights", "📅 AI Smart Scheduler"])

# -----------------------------
# PAGE: HOME
# -----------------------------
if page == "🏠 Home":
    st.title("Student Productivity & Burnout Analysis System")
    st.markdown("### Integrating Machine Learning for Student Success")
    st.image("https://images.unsplash.com/photo-1434030216411-0b793f4b4173?w=800")
    st.info("Use the sidebar to input your data or view the global dashboard.")

# -----------------------------
# PAGE: PREDICTION & ANALYSIS
# -----------------------------
elif page == "📊 Prediction & Analysis":
    st.header("🧠 Behavioral AI Diagnostics")
    st.markdown("Please answer the following questions honestly to get an accurate productivity and burnout assessment.")
    
    with st.form("main_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Demographic & Baseline")
            age = st.number_input("How old are you?", 15, 60, 22)
            gender = st.selectbox("What is your gender identity?", ["Male", "Female", "Other"])
            
            st.subheader("🕒 Daily Habits")
            work_hours = st.slider("On average, how many hours do you spend studying or working per day?", 1.0, 16.0, 7.0)
            sleep_hours = st.slider("How many hours of sleep did you get last night?", 4.0, 12.0, 6.5)
            stress = st.slider("On a scale of 1-10, how 'stressed out' do you feel right now?", 1, 10, 5, 
                               help="1 is totally relaxed, 10 is extremely overwhelmed.")
        
        with col2:
            st.subheader("📱 Digital Lifestyle")
            social_media = st.slider("How much time do you spend on social media daily (hrs)?", 0.0, 12.0, 3.0)
            offline_hrs = st.slider("In a week, how many hours do you spend completely 'unplugged' from devices?", 0.0, 100.0, 12.0)
            breaks = st.number_input("How many intentional breaks do you take during your work sessions?", 0, 15, 2)
            
            st.subheader("🎯 Self-Assessment")
            workload_score = st.slider("Rate your current workload intensity (0-10):", 0.0, 10.0, 5.0,
                               help="0 is very light, 10 is maximum capacity.")
            sat_score_raw = st.slider("How satisfied are you with your current academic/work progress (0-10)?", 0.0, 10.0, 6.0)
            completion = st.slider("What percentage of your planned tasks did you actually finish today (0-10)?", 0.0, 10.0, 6.0, 
                                   help="Move to 10 if you finished everything you planned.")
            goal_rate = st.slider("How successful are you at meeting your long-term goals lately (0-10)?", 0.0, 10.0, 5.0)
        
        st.markdown("---")
        submitted = st.form_submit_button("🔍 Generate My Analysis")

    if submitted:
        # Create Dataframe
        df_input = pd.DataFrame({
            "age": [age], "gender": [gender], "daily_social_media_time": [social_media],
            "work_hours_per_day": [work_hours], "stress_level": [stress], "sleep_hours": [sleep_hours],
            "breaks_during_work": [breaks], "uses_focus_apps": [True], 
            "has_digital_wellbeing_enabled": [True], "coffee_consumption_per_day": [2.0], 
            "days_feeling_burnout_per_month": [8], "weekly_offline_hours": [offline_hrs], 
            "job_satisfaction_score": [sat_score_raw]
        })

        # Feature Engineering for Regression
        df_input["social_media_intensity"] = df_input["daily_social_media_time"] / (df_input["work_hours_per_day"] + 1)
        df_input["work_stress_ratio"] = df_input["work_hours_per_day"] / (df_input["stress_level"] + 1)
        df_input["rest_work_balance"] = (df_input["sleep_hours"] + df_input["weekly_offline_hours"] / 7) / (df_input["work_hours_per_day"] + 1)

        # 1. Predict Productivity
        df_reg = pd.get_dummies(df_input).reindex(columns=model_cols, fill_value=0)
        prod_score = reg_model.predict(imputer.transform(df_reg))[0]

        # 2. Predict Persona (Clustering)
        df_input['actual_productivity_score'] = prod_score
        clust_input = cluster_scaler.transform(df_input[cluster_features])
        cluster_id = cluster_model.predict(clust_input)[0]
        persona = cluster_names.get(cluster_id, "Unknown")

        # 3. Predict Burnout (Classification)
        # Note: Ensure columns align with your burnout_classifier training (added stress_level)
        clf_input = pd.DataFrame([[sat_score_raw, workload_score, completion, goal_rate]], 
                        columns=['satisfaction_score', 'workload_score', 'project_completion_rate', 'goal_achievement_rate'])

        # Ensure the columns are in the EXACT order your model expects:
        correct_order = ['satisfaction_score', 'workload_score', 'project_completion_rate', 'goal_achievement_rate']
        clf_input = clf_input[correct_order]

        burnout_idx = burnout_model.predict(clf_input)[0]
        burnout_probs = burnout_model.predict_proba(clf_input)[0]
        burnout_label = burnout_le.inverse_transform([burnout_idx])[0]

        # UI Results Display
        st.success("Analysis Complete!")
        st.divider()
        res1, res2, res3 = st.columns(3)
        res1.metric("Calculated Productivity", f"{prod_score:.2f}/10")
        res2.metric("Your Student Persona", persona)
        res3.metric("Burnout Risk Level", burnout_label)

        # Burnout Probability Breakdown
        with st.expander("📊 Technical Breakdown: Burnout Risk Probabilities"):
            st.write("This chart shows the probability assigned by the AI to each risk category based on your responses.")
            prob_df = pd.DataFrame({
                "Risk Level": burnout_le.classes_,
                "Probability (%)": [p * 100 for p in burnout_probs]
            })
            st.bar_chart(prob_df.set_index("Risk Level"))
        
        st.info(get_recommendation(prod_score, persona, burnout_label))
# -----------------------------
# PAGE: GLOBAL INSIGHTS (DASHBOARD)
# -----------------------------
elif page == "📈 Global Insights":
    st.title("Behavioral Dashboard")
    
    # Selection for comparison
    persona_filter = st.multiselect("Select Personas to Compare", 
                                    options=list(cluster_names.values()), 
                                    default=["Grinders", "Zen Masters", "Overwhelmed"])
    
    # Mock means for the radar - Replace these with your df.groupby('cluster').mean() logic
    data_map = {
        "Overwhelmed": [4, 4, 9, 2, 3, 10],
        "Grinders": [9, 5, 8, 4, 7, 7],
        "Zen Masters": [6, 9, 2, 9, 8, 3],
        "Burnout Risk": [8, 3, 10, 1, 4, 8],
        "Coasters": [3, 7, 2, 6, 5, 5],
        "Restless Sleepers": [5, 2, 7, 4, 5, 6]
    }
    
    st.subheader("Inter-Persona Multi-Dimensional Comparison")
    categories = ['Work Intensity', 'Sleep Quality', 'Stress Level', 'Satisfaction', 'Task Completion', 'Social Media']
    
    fig_radar = go.Figure()
    for p in persona_filter:
        fig_radar.add_trace(go.Scatterpolar(
            r=data_map[p],
            theta=categories,
            fill='toself',
            name=p
        ))
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=True,
        height=500
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    

# -----------------------------
# PAGE: SCHEDULER
# -----------------------------
elif page == "📅 AI Smart Scheduler":
    st.title("AI Smart Scheduler")
    
    # State Handling
    current_persona = st.selectbox("Current Persona Profile", list(cluster_names.values()), key="persona_select")
    
    # Clear session state if persona changes to reset suggestions
    if "prev_persona" not in st.session_state:
        st.session_state.prev_persona = current_persona
    
    if st.session_state.prev_persona != current_persona:
        st.session_state.tasks = []
        st.session_state.prev_persona = current_persona
        st.warning(f"Persona changed to {current_persona}. Schedule cleared to apply new logic.")

    with st.expander("➕ Add New Task"):
        t_name = st.text_input("Task Description")
        t_dur = st.number_input("Duration (hrs)", 0.5, 8.0, 1.0, 0.5)
        t_prio = st.selectbox("Priority", ["High", "Medium", "Low"])
        if st.button("Append Task"):
            st.session_state.tasks.append({"Task": t_name, "Duration": t_dur, "Priority": t_prio})
            st.rerun()

    if st.session_state.tasks:
        st.subheader("Current Task List")
        # Editable table with DELETE option
        edited_tasks = st.data_editor(st.session_state.tasks, num_rows="dynamic", key="task_editor")
        st.session_state.tasks = edited_tasks

        if st.button("Generate AI Timeline"):
            start_time = datetime.strptime("09:00", "%H:%M")
            schedule_list = []
            
            # Logic Tuning by Persona
            logic = {
                "Overwhelmed": {"buffer": 45, "max_work": 1.0, "tip": "Short sprints + long breaks."},
                "Grinders": {"buffer": 10, "max_work": 3.0, "tip": "Deep work focus."},
                "Zen Masters": {"buffer": 20, "max_work": 2.0, "tip": "Balanced flow."},
                "Burnout Risk": {"buffer": 60, "max_work": 0.5, "tip": "Recovery focus. Minimum work."},
                "Coasters": {"buffer": 15, "max_work": 2.5, "tip": "Steady pace."},
                "Restless Sleepers": {"buffer": 30, "max_work": 1.5, "tip": "Frequent eye-rest breaks."}
            }
            
            cfg = logic.get(current_persona, logic["Zen Masters"])
            st.caption(f"**Scheduling Strategy for {current_persona}:** {cfg['tip']}")

            for task in st.session_state.tasks:
                if task['Duration'] > cfg['max_work']:
                    st.warning(f"Task '{task['Task']}' exceeds recommended focus for your persona. Split it!")
                
                end_time = start_time + timedelta(hours=task['Duration'])
                schedule_list.append({
                    "Slot": f"{start_time.strftime('%I:%M %p')} - {end_time.strftime('%I:%M %p')}",
                    "Task": task['Task'],
                    "Type": "Work",
                    "Status": task['Priority']
                })
                
                # Buffer/Rest
                break_start = end_time
                break_end = break_start + timedelta(minutes=cfg['buffer'])
                schedule_list.append({
                    "Slot": f"{break_start.strftime('%I:%M %p')} - {break_end.strftime('%I:%M %p')}",
                    "Task": "☕ Rest & Recovery",
                    "Type": "Rest",
                    "Status": "-"
                })
                start_time = break_end

            st.table(schedule_list)
            
            output = io.BytesIO()
            pd.DataFrame(schedule_list).to_excel(output, index=False)
            st.download_button("📥 Export to Excel", data=output.getvalue(), file_name="student_schedule.xlsx")