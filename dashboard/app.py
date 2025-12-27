"""
Real-time Email Trust & Safety Dashboard
Monitors classification metrics, abuse patterns, and system health
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json
from collections import Counter

# Page config
st.set_page_config(
    page_title="Email Trust & Safety Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# API endpoint
API_URL = "http://localhost:8000"

# Title
st.title("🛡️ Email Trust & Safety Dashboard")
st.markdown("Real-time monitoring of email classification and abuse detection")

# Sidebar
st.sidebar.header("Configuration")
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 5, 60, 30)
show_details = st.sidebar.checkbox("Show Detailed Logs", value=True)

# Health Check
try:
    health = requests.get(f"{API_URL}/health", timeout=2).json()
    st.sidebar.success(f"✓ API Status: {health.get('status', 'unknown').upper()}")
    st.sidebar.info(f"Model: {health.get('model', 'unknown')}")
except:
    st.sidebar.error("✗ API Offline")

# Main metrics
col1, col2, col3, col4 = st.columns(4)

# Simulate real-time metrics (in production, fetch from database)
with col1:
    st.metric("Total Emails Processed", "15,847", "+234 (24h)")

with col2:
    st.metric("Phishing Detected", "1,294", "+12 (24h)")

with col3:
    st.metric("Spam Blocked", "3,521", "+89 (24h)")

with col4:
    st.metric("False Positive Rate", "0.02%", "-0.01% (24h)")

st.divider()

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Live Classification", 
    "🔍 Abuse Patterns",
    "📈 Performance Metrics",
    "🚨 Alerts",
    "🧪 Test Email"
])

# Tab 1: Live Classification
with tab1:
    st.header("Live Email Classification")
    
    # Classification distribution
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        labels = ['Ham', 'Spam', 'Phishing']
        values = [11032, 3521, 1294]
        
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
        fig.update_layout(title="Classification Distribution (Last 24h)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Time series
        hours = list(range(24))
        ham = [450 + i*10 for i in hours]
        spam = [150 - i*2 for i in hours]
        phish = [50 + i*1 for i in hours]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hours, y=ham, name='Ham', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=hours, y=spam, name='Spam', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=hours, y=phish, name='Phishing', line=dict(color='red')))
        fig.update_layout(title="Emails per Hour", xaxis_title="Hour", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

# Tab 2: Abuse Patterns
with tab2:
    st.header("Detected Abuse Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Malicious Domains")
        
        domains_data = {
            "Domain": [
                "mega-lottery-international.xyz",
                "bank-verify.ru",
                "secure-payment.top",
                "prize-claim.cn",
                "urgent-account.ml"
            ],
            "Threat Count": [847, 623, 512, 498, 389],
            "Risk Level": ["Critical", "Critical", "High", "High", "High"]
        }
        df_domains = pd.DataFrame(domains_data)
        
        # Color code by risk level
        def color_risk(val):
            if val == "Critical":
                return 'background-color: #ff4444; color: white'
            elif val == "High":
                return 'background-color: #ff8800; color: white'
            return ''
        
        st.dataframe(
            df_domains.style.applymap(color_risk, subset=['Risk Level']),
            use_container_width=True
        )
    
    with col2:
        st.subheader("Attack Vectors")
        
        vectors = ["Lottery Scam", "Account Phishing", "CEO Fraud", "Invoice Scam", "Tech Support"]
        counts = [847, 623, 312, 289, 223]
        
        fig = go.Figure(data=[
            go.Bar(x=vectors, y=counts, marker_color=['#ff4444', '#ff6644', '#ff8844', '#ffaa44', '#ffcc44'])
        ])
        fig.update_layout(title="Attack Vector Distribution", xaxis_title="Type", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Recent High-Risk Detections")
    
    recent_threats = {
        "Timestamp": [
            "2025-12-20 15:45:23",
            "2025-12-20 15:42:11",
            "2025-12-20 15:38:55",
            "2025-12-20 15:35:42"
        ],
        "From": [
            "winner-alert@mega-lottery-international.xyz",
            "security@bank-verify.ru",
            "admin@urgent-account.ml",
            "prize@claim-now.top"
        ],
        "Subject": [
            "CONGRATULATIONS! You've WON $5,000,000",
            "Account Suspended - Verify Now",
            "URGENT: Update Your Information",
            "Claim Your Prize Before Expiry"
        ],
        "Risk Score": [170, 95, 88, 82],
        "Classification": ["Phishing", "Phishing", "Phishing", "Phishing"]
    }
    df_threats = pd.DataFrame(recent_threats)
    st.dataframe(df_threats, use_container_width=True)

# Tab 3: Performance Metrics
with tab3:
    st.header("Model Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Precision", "98.5%", "+0.2%")
        st.metric("Recall", "97.8%", "+0.1%")
    
    with col2:
        st.metric("F1-Score", "98.1%", "+0.15%")
        st.metric("Accuracy", "98.3%", "+0.18%")
    
    with col3:
        st.metric("AUC-ROC", "0.992", "+0.003")
        st.metric("Avg Response Time", "45ms", "-2ms")
    
    st.subheader("Confusion Matrix")
    
    # Confusion matrix heatmap
    confusion_data = pd.DataFrame(
        [[10832, 150, 50],
         [120, 3401, 0],
         [45, 15, 1234]],
        columns=['Pred Ham', 'Pred Spam', 'Pred Phishing'],
        index=['True Ham', 'True Spam', 'True Phishing']
    )
    
    fig = px.imshow(confusion_data, text_auto=True, color_continuous_scale='Blues')
    fig.update_layout(title="Model Confusion Matrix (Last 7 Days)")
    st.plotly_chart(fig, use_container_width=True)

# Tab 4: Alerts
with tab4:
    st.header("🚨 Active Alerts")
    
    alerts = [
        {
            "severity": "critical",
            "title": "Coordinated Phishing Attack Detected",
            "description": "15+ emails from lottery-prize.xyz domain targeting multiple users",
            "time": "2 minutes ago"
        },
        {
            "severity": "high",
            "title": "New Malicious Domain Detected",
            "description": "bank-security-alert.ru sending account suspension emails",
            "time": "15 minutes ago"
        },
        {
            "severity": "medium",
            "title": "Unusual Spike in Spam",
            "description": "Spam volume increased 25% in last hour",
            "time": "1 hour ago"
        }
    ]
    
    for alert in alerts:
        if alert["severity"] == "critical":
            st.error(f"🚨 **{alert['title']}**\n\n{alert['description']}\n\n*{alert['time']}*")
        elif alert["severity"] == "high":
            st.warning(f"⚠️ **{alert['title']}**\n\n{alert['description']}\n\n*{alert['time']}*")
        else:
            st.info(f"ℹ️ **{alert['title']}**\n\n{alert['description']}\n\n*{alert['time']}*")

# Tab 5: Test Email
with tab5:
    st.header("🧪 Test Email Classification")
    
    with st.form("test_email_form"):
        from_email = st.text_input("From Email", "test@example.com")
        to_email = st.text_input("To Email", "recipient@example.com")
        subject = st.text_input("Subject", "Test Email")
        body = st.text_area("Body", "This is a test email message.", height=150)
        
        submit = st.form_submit_button("Classify Email")
        
        if submit:
            with st.spinner("Classifying email..."):
                try:
                    # payload = {
                    #     "message_id": "test-" + datetime.now().strftime("%Y%m%d%H%M%S"),
                    #     "from": from_email,
                    #     "to": [to_email],
                    #     "subject": subject,
                    #     "body": body,
                    #     "headers": {}
                    # }
                    payload = {
                        "message_id": "test-" + datetime.now().strftime("%Y%m%d%H%M%S"),
                        "from": from_email.strip(),  # ✅ Use "from" not "from_email"
                        "to": [to_email.strip()],
                        "subject": subject.strip(),
                        "body": body.strip(),
                        "headers": {}
                                                    }

                    response = requests.post(
                        f"{API_URL}/score_email",
                        json=payload,
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display results
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            label = result["predicted"]["predicted_label"]
                            if label == "ham":
                                st.success(f"✅ Classification: **{label.upper()}**")
                            elif label == "spam":
                                st.warning(f"⚠️ Classification: **{label.upper()}**")
                            else:
                                st.error(f"🚨 Classification: **{label.upper()}**")
                        
                        with col2:
                            st.metric("Risk Score", f"{result['risk_analysis']['risk_score']}/100")
                        
                        with col3:
                            st.metric("Confidence", f"{result['predicted']['model_confidence']:.2%}")
                        
                        # Show scores
                        st.subheader("Prediction Scores")
                        scores_df = pd.DataFrame({
                            "Class": ["Ham", "Spam", "Phishing"],
                            "Score": [
                                result["predicted"]["ham_score"],
                                result["predicted"]["spam_score"],
                                result["predicted"]["phishing_score"]
                            ]
                        })
                        fig = px.bar(scores_df, x="Class", y="Score", color="Class",
                                    color_discrete_map={"Ham": "green", "Spam": "orange", "Phishing": "red"})
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show risk factors
                        if result.get("reasons"):
                            st.subheader("Risk Factors Detected")
                            for reason in result["reasons"]:
                                if reason["weight"] == "critical":
                                    st.error(f"🚨 **{reason['type']}**: {reason['description']}")
                                elif reason["weight"] == "high":
                                    st.warning(f"⚠️ **{reason['type']}**: {reason['description']}")
                                else:
                                    st.info(f"ℹ️ **{reason['type']}**: {reason['description']}")
                        
                        # Show explanation
                        if result.get("explanation"):
                            st.subheader("Explanation")
                            st.info(result["explanation"])
                        
                        # Show raw JSON
                        with st.expander("View Raw JSON Response"):
                            st.json(result)
                    
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                
                except Exception as e:
                    st.error(f"Failed to classify email: {str(e)}")

# Footer
st.divider()
st.caption("Email Trust & Safety Dashboard | Real-time Abuse Detection & Monitoring")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")