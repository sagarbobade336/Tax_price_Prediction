import streamlit as st
import pandas as pd
import joblib

# Load the trained model
try:
    model = joblib.load('best_taxi_model.pkl')
except FileNotFoundError:
    st.error("Error: The 'best_taxi_model.pkl' file was not found. Please make sure it's in the same directory.")
    st.stop()

def predict_fare(features):
    """
    Predicts the taxi fare using the loaded model.
    The function expects a dictionary of features.
    """
    
    # The model was trained on a specific feature order and one-hot encoded columns.
    # We need to ensure the input data has the same structure.
    
    # Define all possible feature columns as they were in the training data
    column_order = [
        'Trip_Distance_km', 'Passenger_Count', 'Base_Fare', 'Per_Km_Rate',
        'Per_Minute_Rate', 'Trip_Duration_Minutes', 'Time_of_Day_Evening',
        'Time_of_Day_Morning', 'Time_of_Day_Night', 'Day_of_Week_Weekend',
        'Traffic_Conditions_Low', 'Traffic_Conditions_Medium',
        'Weather_Rain', 'Weather_Snow'
    ]
    
    # Create a DataFrame from the input features
    input_df = pd.DataFrame([features])
    
    # Reindex the DataFrame to match the training data's column order and fill missing columns with 0
    # This is crucial for one-hot encoded categorical features
    input_df = input_df.reindex(columns=column_order, fill_value=0)
    
    # Make the prediction
    prediction = model.predict(input_df)
    
    return prediction[0]

# --- Streamlit UI ---
st.set_page_config(page_title="Taxi Fare Predictor", layout="centered")
st.title('🚖 Taxi Fare Prediction')
st.markdown('Enter the trip details below to get an estimated fare.')

# --- Input widgets ---
st.header('Trip Information')

col1, col2 = st.columns(2)

with col1:
    trip_distance = st.number_input('Trip Distance (km)', min_value=0.1, max_value=200.0, value=10.0, step=1.0)
    passenger_count = st.slider('Passenger Count', min_value=1, max_value=6, value=1)
    base_fare = st.number_input('Base Fare ($)', min_value=1.0, max_value=10.0, value=3.0, step=0.5)
    per_km_rate = st.number_input('Rate per km ($)', min_value=0.1, max_value=5.0, value=0.8, step=0.1)
    
with col2:
    per_minute_rate = st.number_input('Rate per Minute ($)', min_value=0.1, max_value=5.0, value=0.3, step=0.1)
    trip_duration = st.number_input('Trip Duration (minutes)', min_value=1, max_value=300, value=20)
    
    time_of_day = st.selectbox('Time of Day', ['Morning', 'Afternoon', 'Evening', 'Night'])
    day_of_week = st.selectbox('Day of Week', ['Weekday', 'Weekend'])

st.header('Environmental Conditions')
col3, col4 = st.columns(2)

with col3:
    traffic_conditions = st.selectbox('Traffic Conditions', ['Low', 'Medium', 'High'])
    
with col4:
    weather_conditions = st.selectbox('Weather', ['Clear', 'Rain', 'Snow'])

# --- Prediction button and result display ---
if st.button('Predict Fare', use_container_width=True):
    # Prepare the input data dictionary for the prediction function
    input_data = {
        'Trip_Distance_km': trip_distance,
        'Passenger_Count': float(passenger_count),
        'Base_Fare': base_fare,
        'Per_Km_Rate': per_km_rate,
        'Per_Minute_Rate': per_minute_rate,
        'Trip_Duration_Minutes': float(trip_duration)
    }
    
    # One-hot encode the categorical features
    if time_of_day == 'Morning':
        input_data['Time_of_Day_Morning'] = 1
    elif time_of_day == 'Evening':
        input_data['Time_of_Day_Evening'] = 1
    elif time_of_day == 'Night':
        input_data['Time_of_Day_Night'] = 1
    
    if day_of_week == 'Weekend':
        input_data['Day_of_Week_Weekend'] = 1
        
    if traffic_conditions == 'Low':
        input_data['Traffic_Conditions_Low'] = 1
    elif traffic_conditions == 'Medium':
        input_data['Traffic_Conditions_Medium'] = 1
    
    if weather_conditions == 'Rain':
        input_data['Weather_Rain'] = 1
    elif weather_conditions == 'Snow':
        input_data['Weather_Snow'] = 1
        
    # Make the prediction
    predicted_price = predict_fare(input_data)
    
    # Display the result
    st.success(f'### Estimated Trip Price: ${predicted_price:,.2f}')