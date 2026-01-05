import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import time  # For debugging

# -----------------------------
# 1. LOAD ALL ASSETS WITH CACHING
# -----------------------------
@st.cache_resource
def load_models():
    """Load ML models once and cache"""
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

@st.cache_data
def load_global_data():
    """Load and cache the global dataset"""
    clustered_studentType_dataset = "C:/Users/Jia Xuan/Downloads/fyp-linked-github/fyp_Degree/data_with_cluster_type.xlsx"
    return pd.read_excel(clustered_studentType_dataset)

# -----------------------------
# 2. INITIALIZE SESSION STATE ONCE
# -----------------------------
if 'initialized' not in st.session_state:
    # Load heavy resources once
    st.session_state.models = load_models()
    st.session_state.df_global = load_global_data()
    st.session_state.cluster_names = {0: "Overwhelmed", 1: "Grinders", 2: "Zen Masters", 
                                      3: "Burnout Risk", 4: "Coasters", 5: "Restless Sleepers"}
    st.session_state.tasks = []
    st.session_state.prev_persona = None
    st.session_state.initialized = True
    st.session_state.last_submission_time = None  # For form debouncing

# Unpack from session state for cleaner access
models = st.session_state.models
reg_model, imputer, scaler, model_cols, cluster_model, cluster_scaler, cluster_features, burnout_model, burnout_le = models
df_global = st.session_state.df_global
cluster_names = st.session_state.cluster_names

# -----------------------------
# 3. HELPER FUNCTIONS (Lightweight)
# -----------------------------
def get_recommendation(prod, persona, burnout):
    if burnout == "High":
        return "🚨 **Urgent:** High burnout risk. Reduce work hours and prioritize 8h sleep."
    if persona == "Grinders" and prod > 7:
        return "⚠️ **Caution:** You are a high achiever but at risk. Implement Pomodoro (5min breaks)."
    return "✅ **Stable:** Your current patterns appear sustainable. Maintain your offline hours."

# -----------------------------
# 4. SIDEBAR NAVIGATION (Keep outside pages)
# -----------------------------
st.sidebar.title("🎯 Student Wellbeing Helper")
page = st.sidebar.selectbox("Navigate To", ["🏠 Home", "📊 Prediction & Analysis", "📈 Global Insights", "📅 AI Smart Scheduler"])

# -----------------------------
# 5. PAGE: HOME (Lightweight)
# -----------------------------
if page == "🏠 Home":
    st.title("Student Productivity & Burnout Analysis System")
    st.markdown("### Integrating Machine Learning for Student Success")
    st.image("https://images.unsplash.com/photo-1434030216411-0b793f4b4173?w=800")
    st.info("Use the sidebar to input your data or view the global dashboard.")

# -----------------------------
# 6. PAGE: PREDICTION & ANALYSIS (Optimized)
# -----------------------------
elif page == "📊 Prediction & Analysis":
    st.header("🧠 Behavioral AI Diagnostics")
    st.markdown("Please answer the following questions honestly to get an accurate productivity and burnout assessment.")
    
    # Use a form key to prevent re-submission on every interaction
    with st.form(key="prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Demographic & Baseline")
            age = st.number_input("How old are you?", 15, 60, 22, key="age_input")
            gender = st.selectbox("What is your gender identity?", ["Male", "Female", "Other"], key="gender_input")
            
            st.subheader("🕒 Daily Habits")
            work_hours = st.slider("On average, how many hours do you spend studying or working per day?", 
                                  1.0, 16.0, 7.0, key="work_hours_slider")
            sleep_hours = st.slider("How many hours of sleep did you get last night?", 
                                  4.0, 12.0, 6.5, key="sleep_hours_slider")
            stress = st.slider("On a scale of 1-10, how 'stressed out' do you feel right now?", 
                              1, 10, 5, help="1 is totally relaxed, 10 is extremely overwhelmed.", 
                              key="stress_slider")
        
        with col2:
            st.subheader("📱 Digital Lifestyle")
            social_media = st.slider("How much time do you spend on social media daily (hrs)?", 
                                    0.0, 12.0, 3.0, key="social_media_slider")
            offline_hrs = st.slider("In a week, how many hours do you spend completely 'unplugged' from devices?", 
                                   0.0, 100.0, 12.0, key="offline_hrs_slider")
            breaks = st.number_input("How many intentional breaks do you take during your work sessions?", 
                                    0, 15, 2, key="breaks_input")
            
            st.subheader("🎯 Self-Assessment")
            workload_score = st.slider("Rate your current workload intensity (0-10):", 
                                       0.0, 10.0, 5.0, help="0 is very light, 10 is maximum capacity.",
                                       key="workload_slider")
            sat_score_raw = st.slider("How satisfied are you with your current academic/work progress (0-10)?", 
                                      0.0, 10.0, 6.0, key="satisfaction_slider")
            completion = st.slider("What percentage of your planned tasks did you actually finish today (0-10)?", 
                                   0.0, 10.0, 6.0, help="Move to 10 if you finished everything you planned.",
                                   key="completion_slider")
            goal_rate = st.slider("How successful are you at meeting your long-term goals lately (0-10)?", 
                                  0.0, 10.0, 5.0, key="goal_rate_slider")
        
        st.markdown("---")
        submitted = st.form_submit_button("🔍 Generate My Analysis", type="primary")
    
    # Only run heavy computation when form is submitted
    if submitted:
        # Add a spinner for user feedback
        with st.spinner("🤖 AI is analyzing your patterns..."):
            # Create Dataframe
            df_input = pd.DataFrame({
                "age": [age], "gender": [gender], "daily_social_media_time": [social_media],
                "work_hours_per_day": [work_hours], "stress_level": [stress], "sleep_hours": [sleep_hours],
                "breaks_during_work": [breaks], "uses_focus_apps": [True], 
                "has_digital_wellbeing_enabled": [True], "coffee_consumption_per_day": [2.0], 
                "days_feeling_burnout_per_month": [8], "weekly_offline_hours": [offline_hrs], 
                "job_satisfaction_score": [sat_score_raw]
            })

            # Feature Engineering
            df_input["social_media_intensity"] = df_input["daily_social_media_time"] / (df_input["work_hours_per_day"] + 1)
            df_input["work_stress_ratio"] = df_input["work_hours_per_day"] / (df_input["stress_level"] + 1)
            df_input["rest_work_balance"] = (df_input["sleep_hours"] + df_input["weekly_offline_hours"] / 7) / (df_input["work_hours_per_day"] + 1)

            # 1. Predict Productivity
            df_reg = pd.get_dummies(df_input).reindex(columns=model_cols, fill_value=0)
            prod_score = reg_model.predict(imputer.transform(df_reg))[0]

            # 2. Predict Persona
            df_input['actual_productivity_score'] = prod_score
            clust_input = cluster_scaler.transform(df_input[cluster_features])
            cluster_id = cluster_model.predict(clust_input)[0]
            persona = cluster_names.get(cluster_id, "Unknown")

            # 3. Predict Burnout
            clf_input = pd.DataFrame([[sat_score_raw, workload_score, completion, goal_rate]], 
                            columns=['satisfaction_score', 'workload_score', 'project_completion_rate', 'goal_achievement_rate'])
            
            correct_order = ['satisfaction_score', 'workload_score', 'project_completion_rate', 'goal_achievement_rate']
            clf_input = clf_input[correct_order]

            burnout_idx = burnout_model.predict(clf_input)[0]
            burnout_probs = burnout_model.predict_proba(clf_input)[0]
            burnout_label = burnout_le.inverse_transform([burnout_idx])[0]
        
        # Store results in session state to persist between interactions
        st.session_state.prediction_results = {
            "prod_score": prod_score,
            "persona": persona,
            "burnout_label": burnout_label,
            "burnout_probs": burnout_probs
        }
    
    # Display results from session state if they exist
    if "prediction_results" in st.session_state:
        results = st.session_state.prediction_results
        
        st.success("Analysis Complete!")
        st.divider()
        
        res1, res2, res3 = st.columns(3)
        res1.metric("Calculated Productivity", f"{results['prod_score']:.2f}/10")
        res2.metric("Your Student Persona", results['persona'])
        res3.metric("Burnout Risk Level", results['burnout_label'])

        # Burnout Probability Breakdown
        with st.expander("📊 Technical Breakdown: Burnout Risk Probabilities"):
            st.write("This chart shows the probability assigned by the AI to each risk category based on your responses.")
            prob_df = pd.DataFrame({
                "Risk Level": burnout_le.classes_,
                "Probability (%)": [p * 100 for p in results['burnout_probs']]
            })
            st.bar_chart(prob_df.set_index("Risk Level"))
        
        st.info(get_recommendation(results['prod_score'], results['persona'], results['burnout_label']))

# -----------------------------
# 7. PAGE: GLOBAL INSIGHTS (Optimized)
# -----------------------------
elif page == "📈 Global Insights":
    st.title("📈 Behavioral Dashboard (Global Insights)")
    
    # Cache expensive chart computations
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def create_persona_charts(df_filtered, radar_features, radar_labels):
        """Create and cache dashboard charts"""
        # Persona Distribution
        persona_dist = (
            df_global['StudentType']
            .value_counts(normalize=True)
            .reset_index()
        )
        persona_dist.columns = ['StudentType', 'Percentage']
        persona_dist['Percentage'] *= 100
        
        fig_pie = px.pie(
            persona_dist,
            names='StudentType',
            values='Percentage',
            hole=0.55,
            title="🧑‍🎓 Student Persona Distribution (%)"
        )
        
        # Coffee Consumption
        coffee_df = (
            df_filtered
            .groupby('StudentType')['coffee_consumption_per_day']
            .mean()
            .reset_index()
        )
        
        fig_coffee = px.bar(
            coffee_df,
            x='coffee_consumption_per_day',
            y='StudentType',
            orientation='h',
            text_auto='.2f',
            title="☕ Avg Coffee Consumption by Persona"
        )
        
        # Radar Chart
        radar_df = (
            df_filtered
            .groupby('StudentType')[radar_features]
            .mean()
            .reset_index()
        )
        
        radar_scaled = radar_df.copy()
        for col in radar_features:
            radar_scaled[col] = (
                radar_scaled[col] - radar_scaled[col].min()
            ) / (radar_scaled[col].max() - radar_scaled[col].min()) * 10
        
        fig_radar = go.Figure()
        for _, row in radar_scaled.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=row[radar_features].values,
                theta=radar_labels,
                fill='toself',
                name=row['StudentType']
            ))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
            height=520,
            showlegend=True
        )
        
        return fig_pie, fig_coffee, fig_radar
    
    # Persona Filter
    persona_filter = st.multiselect(
        "Select Personas to Compare",
        options=df_global['StudentType'].unique(),
        default=["Grinders", "Zen Masters", "Overwhelmed"]
    )
    
    df_filtered = df_global[df_global['StudentType'].isin(persona_filter)]
    
    # KPI ROW (Lightweight)
    st.subheader("📊 Persona-Level Summary")
    k1, k2, k3, k4, k5 = st.columns(5)
    
    with k1:
        st.metric("Avg Productivity", f"{df_filtered['actual_productivity_score'].mean():.2f}")
    with k2:
        st.metric("Avg Stress", f"{df_filtered['stress_level'].mean():.2f}")
    with k3:
        st.metric("Avg Sleep (hrs)", f"{df_filtered['sleep_hours'].mean():.2f}")
    with k4:
        st.metric("Avg Satisfaction", f"{df_filtered['job_satisfaction_score'].mean():.2f}")
    with k5:
        st.metric("Avg Coffee (cups)", f"{df_filtered['coffee_consumption_per_day'].mean():.2f}")
    
    st.divider()
    
    # Generate cached charts
    radar_features = [
        'work_hours_per_day',
        'sleep_hours',
        'stress_level',
        'job_satisfaction_score',
        'actual_productivity_score',
        'daily_social_media_time'
    ]
    
    radar_labels = [
        'Work Intensity',
        'Sleep Quality',
        'Stress Level',
        'Satisfaction',
        'Productivity',
        'Social Media Usage'
    ]
    
    fig_pie, fig_coffee, fig_radar = create_persona_charts(df_filtered, radar_features, radar_labels)
    
    # ROW 1 — DISTRIBUTION + COFFEE
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_pie, use_container_width=True)
    with c2:
        st.plotly_chart(fig_coffee, use_container_width=True)
    
    st.divider()
    
    # RADAR CHART
    st.subheader("🧭 Persona Behavioral DNA")
    st.plotly_chart(fig_radar, use_container_width=True)
    
    st.divider()
    
    # ROW 2 — STRESS + SLEEP (Cached)
    c3, c4 = st.columns(2)
    
    with st.spinner("Generating visualizations..."):
        fig_scatter = px.scatter(
            df_filtered,
            x='stress_level',
            y='actual_productivity_score',
            color='StudentType',
            hover_data=['sleep_hours', 'job_satisfaction_score'],
            title="📉 Stress vs Productivity"
        )
        
        fig_box = px.box(
            df_filtered,
            x='StudentType',
            y='sleep_hours',
            color='StudentType',
            title="😴 Sleep Distribution by Persona"
        )
        
        with c3:
            st.plotly_chart(fig_scatter, use_container_width=True)
        with c4:
            st.plotly_chart(fig_box, use_container_width=True)
    
    st.divider()

# -----------------------------
# 8. PAGE: SCHEDULER (Optimized)
# -----------------------------
elif page == "📅 AI Smart Scheduler":
    st.title("AI Smart Scheduler")
    
    # Current Persona Selection
    current_persona = st.selectbox("Current Persona Profile", list(cluster_names.values()), 
                                   key="persona_select_scheduler")
    
    # Clear tasks only if persona actually changed
    if "prev_persona_scheduler" not in st.session_state:
        st.session_state.prev_persona_scheduler = current_persona
    
    if st.session_state.prev_persona_scheduler != current_persona:
        st.session_state.tasks = []
        st.session_state.prev_persona_scheduler = current_persona
        st.warning(f"Persona changed to {current_persona}. Schedule cleared to apply new logic.")
    
    # Task Management
    with st.expander("➕ Add New Task", expanded=True if not st.session_state.tasks else False):
        t_name = st.text_input("Task Description", key="task_name")
        t_dur = st.number_input("Duration (hrs)", 0.5, 8.0, 1.0, 0.5, key="task_duration")
        t_prio = st.selectbox("Priority", ["High", "Medium", "Low"], key="task_priority")
        
        col_add, _ = st.columns([1, 3])
        with col_add:
            if st.button("Append Task", key="add_task_button"):
                st.session_state.tasks.append({"Task": t_name, "Duration": t_dur, "Priority": t_prio})
                st.rerun()
    
    # Display and Edit Tasks
    if st.session_state.tasks:
        st.subheader("Current Task List")
        
        # Use a form for task editing to prevent constant re-runs
        with st.form("task_editor_form"):
            edited_tasks = st.data_editor(st.session_state.tasks, num_rows="dynamic", 
                                          key="task_editor", use_container_width=True)
            
            col_save, col_clear = st.columns(2)
            with col_save:
                save_changes = st.form_submit_button("💾 Save Changes", type="primary")
            with col_clear:
                if st.form_submit_button("🗑️ Clear All Tasks"):
                    st.session_state.tasks = []
                    st.rerun()
            
            if save_changes:
                st.session_state.tasks = edited_tasks
                st.success("Tasks updated!")
                st.rerun()
        
        # Generate Schedule Button (only shown if tasks exist)
        if st.button("🚀 Generate AI Timeline", type="primary", key="generate_schedule"):
            start_time = datetime.strptime("09:00", "%H:%M")
            schedule_list = []
            
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
                    st.warning(f"Task '{task['Task']}' exceeds recommended focus for your persona. Consider splitting it!")
                
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
            
            # Display Schedule
            st.subheader("📅 Generated Schedule")
            st.table(schedule_list)
            
            # Export
            output = io.BytesIO()
            pd.DataFrame(schedule_list).to_excel(output, index=False)
            st.download_button("📥 Export to Excel", data=output.getvalue(), 
                             file_name="student_schedule.xlsx", mime="application/vnd.ms-excel")