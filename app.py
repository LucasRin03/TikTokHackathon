# app.py
import streamlit as st
import pandas as pd

# Configure the default settings of the page
st.set_page_config(
    page_title="TikTok Trend Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add a title and description
st.title("🚀 TikTok Hackathon: Trend Prediction Dashboard")
st.markdown("""
This interactive dashboard allows you to explore our model's predictions on which TikTok videos will become trends.
**👈 Use the sidebar and tabs below** to navigate different sections.
""")

# Create a sidebar for navigation/inputs
with st.sidebar:
    st.header("Controls")
    st.info("Adjust settings and inputs here.")
    # Example of a simple input in the sidebar
    num_samples = st.slider('Number of samples to load', 5, 100, 20)

# Create tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Explorer", "🔮 Prediction", "📈 Model Performance", "ℹ️ About"])

with tab1:
    st.header("Explore the Dataset")
    st.write("Here you can inspect the data used to train our model.")
    
    # Placeholder for loading data - YOU WILL CONNECT THIS LATER
    if st.button('Load Sample Data', key='load_data'):
        # This is sample data. Replace with code from your data_loader.py!
        sample_df = pd.DataFrame({
            'video_id': range(1, num_samples+1),
            'view_count': [i * 1000 for i in range(1, num_samples+1)],
            'like_count': [i * 100 for i in range(1, num_samples+1)],
            'is_trend': ['Yes' if i % 4 == 0 else 'No' for i in range(1, num_samples+1)]
        })
        st.dataframe(sample_df)
        st.metric(label="Total Samples Loaded", value=num_samples)

with tab2:
    st.header("Make a Prediction")
    st.write("Simulate a prediction on new video data.")
    
    # Example input fields that match what your model expects
    col1, col2 = st.columns(2)
    with col1:
        video_length = st.slider('Video Length (seconds)', 5, 300, 60)
        upload_time = st.selectbox('Upload Time', ['Morning', 'Afternoon', 'Evening', 'Night'])
    with col2:
        likes_ratio = st.slider('Likes/Views Ratio', 0.0, 1.0, 0.2)
        comments_count = st.number_input('Comments Count', min_value=0, value=50)
    
    # Placeholder for making a prediction - YOU WILL CONNECT THIS LATER
    if st.button('Predict Trend', key='predict', type='primary'):
        # This is a placeholder. Replace with code from your models.py!
        with st.spinner('Analyzing video features...'):
            # Simulate a prediction process
            import time
            time.sleep(1) # Simulate computation time
            # Replace this with: prediction = your_model_predict(...)
            prediction = "YES" if (likes_ratio > 0.15 and comments_count > 30) else "NO"
            confidence = 0.85 if prediction == "YES" else 0.72
            
        if prediction == "YES":
            st.success(f'🎉 Prediction: This video WILL trend! (Confidence: {confidence:.0%})')
        else:
            st.error(f'👀 Prediction: This video will NOT trend. (Confidence: {confidence:.0%})')

with tab3:
    st.header("Model Evaluation")
    st.write("See how our model performs on test data.")
    # Placeholder for evaluation metrics
    st.image("https://via.placeholder.com/600x300/2c3e50/ffffff?text=Model+Performance+Metrics+Chart", use_column_width=True)

with tab4:
    st.header("About This Project")
    st.write("""
    **TikTok Hackathon Project**
    - **Team:** [Your Team Name]
    - **Goal:** Predict viral TikTok trends using machine learning.
    - **Models Used:** [List your models, e.g., Gradient Boosting, Neural Networks]
    """)
    st.write("Check out our code and documentation on [GitHub](https://github.com/LucasRin03/TikTokHackathon).")

# Footer
st.markdown("---")
st.caption("Built with Streamlit for the TikTok Hackathon • © 2025")