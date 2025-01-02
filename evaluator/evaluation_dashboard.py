import streamlit as st
import plotly.express as px
from logger import MetricsLogger

def render_evaluation_dashboard(metrics_logger: MetricsLogger):
    st.title("Physics Chatbot Evaluation Dashboard")
    
    # Get metrics summary
    df = metrics_logger.get_metrics_summary()
    
    # Response Time Trends
    st.subheader("Response Time Trends")
    fig_time = px.line(df, x='timestamp', y='response_time', 
                      title='Response Time Over Time')
    st.plotly_chart(fig_time)
    
    # Content Quality Metrics
    st.subheader("Content Quality Metrics")
    quality_metrics = px.bar(df, 
                           x='timestamp', 
                           y=['question_similarity', 'context_similarity'],
                           title='Response Quality Metrics')
    st.plotly_chart(quality_metrics)
    
    # Physics Content Analysis
    st.subheader("Physics Content Analysis")
    physics_metrics = px.bar(df,
                           x='timestamp',
                           y=['equations_count', 'units_count', 'steps_count'],
                           title='Physics Content Metrics')
    st.plotly_chart(physics_metrics)
