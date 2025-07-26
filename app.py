import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Calories Burnt Predictor", page_icon="🔥")
st.title("🔥 Calories Burnt Predictor")

# File uploader
uploaded_file = st.file_uploader("C:\\Users\\yashi\\OneDrive\\Desktop\\calories_burnt\\sample_calories_data.csv", type=["csv"])

if uploaded_file:
    st.success("✅ File uploaded successfully!")
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Input Data Preview")
    st.dataframe(df.head())

    # Checking required columns
    required_columns = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']
    if all(col in df.columns for col in required_columns):
        # Input/Output split
        X = df[['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']]
        y = df['Calories']

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict on test set
        predictions = model.predict(X_test)
        result_df = X_test.copy()
        result_df['Actual Calories'] = y_test.values
        result_df['Predicted Calories'] = predictions.round(2)

        st.subheader("🔮 Predicted vs Actual Calories Burnt (on test data)")
        st.dataframe(result_df.head(10))

        # Optional: Download CSV
        csv_download = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Prediction Results", csv_download, "calorie_predictions.csv", "text/csv")
    else:
        st.error(f"❌ Dataset must contain these columns: {', '.join(required_columns)}")
