import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the pre-trained Decision Tree model
@st.cache_resource
def load_model():
    return joblib.load("dt.sav")

model = load_model()

# Title and Description
st.title("🚚 Delivery Delay Prediction App")
st.write("This interactive application predicts whether a delivery will be **On-Time** or **Delayed** based on logistic and physical characteristics using our highly accurate Decision Tree model.")

st.sidebar.header("Input Delivery Parameters")

# Features based on df.columns:
# ['Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition', 'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age', 'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency', 'Warehouse_Processing_Time']

delivery_distance = st.sidebar.slider("Delivery Distance (miles)", 0.0, 100.0, 25.0)
traffic_congestion = st.sidebar.slider("Traffic Congestion Score (1-5)", 1, 5, 3)
weather_condition = st.sidebar.selectbox("Weather Condition", options=[1, 2, 3], format_func=lambda x: {1: "Clear (1)", 2: "Cloudy (2)", 3: "Rainy/Stormy (3)"}[x])
delivery_slot = st.sidebar.selectbox("Delivery Slot", options=[1, 2, 3], format_func=lambda x: {1: "Morning (1)", 2: "Afternoon (2)", 3: "Evening (3)"}[x])
driver_experience = st.sidebar.slider("Driver Experience (years)", 0, 40, 10)
num_stops = st.sidebar.slider("Number of Stops", 0, 15, 4)
vehicle_age = st.sidebar.slider("Vehicle Age (years)", 0, 20, 5)
road_condition_score = st.sidebar.slider("Road Condition Score (1-5)", 1, 5, 3)
package_weight = st.sidebar.slider("Package Weight (lbs)", 0.1, 100.0, 15.0)
fuel_efficiency = st.sidebar.slider("Fuel Efficiency (mpg)", 5.0, 40.0, 15.0)
warehouse_processing_time = st.sidebar.slider("Warehouse Processing Time (minutes)", 5, 180, 45)

# Prediction logic
if st.button("Predict Delivery Status"):
    # Construct the input array match feature order exactly
    input_features = pd.DataFrame([{
        'Delivery_Distance': delivery_distance,
        'Traffic_Congestion': traffic_congestion,
        'Weather_Condition': weather_condition,
        'Delivery_Slot': delivery_slot,
        'Driver_Experience': driver_experience,
        'Num_Stops': num_stops,
        'Vehicle_Age': vehicle_age,
        'Road_Condition_Score': road_condition_score,
        'Package_Weight': package_weight,
        'Fuel_Efficiency': fuel_efficiency,
        'Warehouse_Processing_Time': warehouse_processing_time
    }])

    # Predict
    prediction = model.predict(input_features)[0]
    
    st.subheader("Prediction Result:")
    if prediction == 1:
         st.error("🔴 Warning: This delivery is predicted to be **DELAYED**.")
    else:
         st.success("🟢 Good News: This delivery is predicted to be **ON-TIME**.")
