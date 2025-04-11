import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="Mall Customer Predictor",
    page_icon="🛍️",
    layout="wide"
)

# Title and description
st.title("🛍️ Mall Customer Segmentation Predictor")
st.markdown("""
Predict customer segments based on demographics and spending habits.
""")

# Cluster interpretation
CLUSTER_DESCRIPTIONS = {
    0: {
        "name": "Budget-Conscious Shoppers",
        "description": "Moderate income, careful spending habits",
        "characteristics": ["Income: $30k-60k", "Age: 30-50", "Spending Score: 40-60"]
    },
    1: {
        "name": "High-Spending Professionals",
        "description": "High income, high spending",
        "characteristics": ["Income: $70k+", "Age: 25-45", "Spending Score: 80+"]
    },
    2: {
        "name": "Young Big Spenders",
        "description": "Younger customers who spend generously",
        "characteristics": ["Income: $40k-70k", "Age: 18-30", "Spending Score: 70+"]
    },
    3: {
        "name": "Conservative Spenders",
        "description": "High income but low spending",
        "characteristics": ["Income: $70k+", "Age: 35-60", "Spending Score: <40"]
    },
    4: {
        "name": "Low-Engagement Customers",
        "description": "Low income and low spending",
        "characteristics": ["Income: <$40k", "Age: Any", "Spending Score: <40"]
    }
}

# Sidebar for user inputs
with st.sidebar:
    st.header("Customer Details")
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 70, 30)
    annual_income = st.slider("Annual Income (k$)", 15, 150, 60)
    spending_score = st.slider("Spending Score (1-100)", 1, 100, 50)
    predict_button = st.button("Predict Segment", type="primary")

# Load or train model
@st.cache_resource
def load_or_train_model():
    DATA_PATH = "data/raw/mall_customers.csv"
    MODEL_PATH = "kmeans_model.pkl"
    
    if Path(MODEL_PATH).exists():
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    else:
        df = pd.read_csv(DATA_PATH)
        X = df[['Age', 'Annual_Income', 'Spending_Score']]
        X = (X - X.mean()) / X.std()
        model = KMeans(n_clusters=5, random_state=42)
        model.fit(X)
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
    
    return model

def show_prediction_results(model, raw_data, scaled_data):
    """Display prediction results and visualizations"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Customer Profile")
        st.write(f"**Gender:** {raw_data['Gender'][0]}")
        st.write(f"**Age:** {raw_data['Age'][0]}")
        st.write(f"**Annual Income:** ${raw_data['Annual_Income'][0]}k")
        st.write(f"**Spending Score:** {raw_data['Spending_Score'][0]}/100")
        
        cluster = model.predict(scaled_data[['Age', 'Annual_Income', 'Spending_Score']])[0]
        segment = CLUSTER_DESCRIPTIONS[cluster]
        
        st.subheader("🎯 Predicted Segment")
        st.success(f"Cluster {cluster}: {segment['name']}")
        st.info(segment['description'])
        
        with st.expander("Segment Characteristics"):
            for char in segment['characteristics']:
                st.write(f"- {char}")
    
    with col2:
        st.subheader("📈 Cluster Visualization")
        if Path("data/raw/mall_customers.csv").exists():
            df = pd.read_csv("data/raw/mall_customers.csv")
            X_full = df[['Age', 'Annual_Income', 'Spending_Score']]
            X_full_scaled = (X_full - X_full.mean()) / X_full.std()
            labels = model.predict(X_full_scaled)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(
                df['Annual_Income'], 
                df['Spending_Score'], 
                c=labels, 
                cmap='viridis',
                alpha=0.6
            )
            
            # Highlight new customer
            ax.scatter(
                raw_data['Annual_Income'][0], 
                raw_data['Spending_Score'][0], 
                c='red', 
                s=200,
                marker='X',
                edgecolors='black',
                label='Your Customer'
            )
            
            ax.set_xlabel("Annual Income (k$)")
            ax.set_ylabel("Spending Score (1-100)")
            ax.set_title("Customer Segments")
            ax.legend()
            st.pyplot(fig)

def show_all_segments():
    st.subheader("🔍 All Segment Profiles")
    cols = st.columns(len(CLUSTER_DESCRIPTIONS))
    
    for i, (cluster_num, segment) in enumerate(CLUSTER_DESCRIPTIONS.items()):
        with cols[i]:
            container = st.container(border=True)
            container.subheader(f"Segment {cluster_num}")
            container.write(f"**{segment['name']}**")
            container.caption(segment['description'])
            
            with container.expander("Details"):
                for char in segment['characteristics']:
                    st.write(f"- {char}")

def main():
    model = load_or_train_model()
    
    # Create customer data from inputs
    raw_data = pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'Annual_Income': [annual_income],
        'Spending_Score': [spending_score]
    })
    
    # Create scaled version for model prediction
    scaled_data = raw_data.copy()
    if Path("data/raw/mall_customers.csv").exists():
        df = pd.read_csv("data/raw/mall_customers.csv")
        train_stats = df[['Age', 'Annual_Income', 'Spending_Score']].mean()
        train_std = df[['Age', 'Annual_Income', 'Spending_Score']].std()
        scaled_data[['Age', 'Annual_Income', 'Spending_Score']] = (
            (scaled_data[['Age', 'Annual_Income', 'Spending_Score']] - train_stats) / train_std
        )
    
    if predict_button:
        show_prediction_results(model, raw_data, scaled_data)
        show_all_segments()
    else:
        st.info("👈 Enter customer details and click 'Predict Segment'")
        show_all_segments()

if __name__ == "__main__":
    main()