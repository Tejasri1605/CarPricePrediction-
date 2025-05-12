import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model and preprocessing steps
model = joblib.load('car_price_prediction_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
scalers = joblib.load('scalers.pkl')

# Load dataset for filtering and identifying similar data
data = pd.read_csv('car_dekho_cleaned_dataset_Raw.csv')

# Set pandas option to handle future downcasting behavior
pd.set_option('future.no_silent_downcasting', True)

# Features used for training
features = ['ft', 'bt', 'km', 'transmission', 'ownerNo', 'oem', 'model', 'modelYear', 'variantName', 'City', 'mileage', 'Seats', 'car_age', 'brand_popularity', 'mileage_normalized']

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = "Home"  # Default page

# Function to navigate between pages
def navigate_to(page_name):
    st.session_state.page = page_name

# Function to filter data based on user selections
def filter_data(oem=None, model=None, body_type=None, fuel_type=None, seats=None):
    filtered_data = data.copy()
    if oem:
        filtered_data = filtered_data[filtered_data['oem'] == oem]
    if model:
        filtered_data = filtered_data[filtered_data['model'] == model]
    if body_type:
        filtered_data = filtered_data[filtered_data['bt'] == body_type]
    if fuel_type:
        filtered_data = filtered_data[filtered_data['ft'] == fuel_type]
    if seats:
        filtered_data = filtered_data[filtered_data['Seats'] == seats]
    return filtered_data

# Preprocessing function for user input
def preprocess_input(df):
    df['car_age'] = 2024 - df['modelYear']
    brand_popularity = data.groupby('oem')['price'].mean().to_dict()
    df['brand_popularity'] = df['oem'].map(brand_popularity)
    df['mileage_normalized'] = df['mileage'] / df['car_age']

    # Apply label encoding
    for column in ['ft', 'bt', 'transmission', 'oem', 'model', 'variantName', 'City']:
        if column in df.columns and column in label_encoders:
            df[column] = df[column].apply(lambda x: label_encoders[column].transform([x])[0])

    # Apply min-max scaling
    for column in ['km', 'ownerNo', 'modelYear']:
        if column in df.columns and column in scalers:
            df[column] = scalers[column].transform(df[[column]])

    return df
home_background_color = "white"  # Background color for the home page
predictor_background_color = "black"  # Background color for the prediction page


home_background_img = '''
<style>
.stApp {
    background: url("https://wallpapercave.com/wp/wp6404592.jpg") no-repeat center center fixed;
    background-size: cover;
}
</style>
'''
predictor_background_color = '''
<style>
.stApp {
     background-color: black !important;
    background-size: cover !important;
}
</style>
'''

# Home Page
if st.session_state.page == "Home":
    st.markdown(home_background_img, unsafe_allow_html=True)  # Apply background image for home page
    st.title("Welcome to the Car Price Prediction App!")
    st.write("Easily estimate the price of your car using advanced AI.")
    
    if st.button("Get Started"):
        navigate_to("Car Price Predictor")


# Car Price Predictor Page
elif st.session_state.page == "Car Price Predictor":
    st.title("Car Price Predictor")
    st.sidebar.header('Input Car Features')
    app_background_color = predictor_background_color

    # Get user inputs
    selected_oem = st.sidebar.selectbox('1. Original Equipment Manufacturer (OEM)', data['oem'].unique())
    filtered_data = filter_data(oem=selected_oem)

    selected_model = st.sidebar.selectbox('2. Car Model', filtered_data['model'].unique())
    filtered_data = filter_data(oem=selected_oem, model=selected_model)

    body_type = st.sidebar.selectbox('3. Body Type', filtered_data['bt'].unique())
    filtered_data = filter_data(oem=selected_oem, model=selected_model, body_type=body_type)

    fuel_type = st.sidebar.selectbox('4. Fuel Type', filtered_data['ft'].unique())
    transmission = st.sidebar.selectbox('5. Transmission Type', filtered_data['transmission'].unique())

    seat_count = st.sidebar.selectbox('6. Seats', filtered_data['Seats'].unique())
    selected_variant = st.sidebar.selectbox('7. Variant Name', filtered_data['variantName'].unique())

    modelYear = st.sidebar.number_input('8. Year of Manufacture', min_value=1980, max_value=2024, value=2015)
    ownerNo = st.sidebar.number_input('9. Number of Previous Owners', min_value=0, max_value=10, value=1)
    km = st.sidebar.number_input('10. Kilometers Driven', min_value=0, max_value=500000, value=10000)

    mileage = st.sidebar.slider('11. Mileage (kmpl)', min_value=5.0, max_value=30.0, value=15.0, step=0.5)
    city = st.sidebar.selectbox('12. City', data['City'].unique())

    # Create a DataFrame for user input
    user_input_data = {
        'ft': [fuel_type],
        'bt': [body_type],
        'km': [km],
        'transmission': [transmission],
        'ownerNo': [ownerNo],
        'oem': [selected_oem],
        'model': [selected_model],
        'modelYear': [modelYear],
        'variantName': [selected_variant],
        'City': [city],
        'mileage': [mileage],
        'Seats': [seat_count],
        'car_age': [2024 - modelYear],
        'brand_popularity': [data.groupby('oem')['price'].mean().to_dict().get(selected_oem)],
        'mileage_normalized': [mileage / (2024 - modelYear)]
    }

    user_df = pd.DataFrame(user_input_data)
    user_df = user_df[features]
    user_df = preprocess_input(user_df)

    # Button to trigger prediction
    if st.sidebar.button('Predict'):
        if user_df.notnull().all().all():
            try:
                prediction = model.predict(user_df)
                st.markdown(f"""
                    <div style="background-color:#FFF8E7;
                    padding:20px;
                    border-radius:10px;
                    text-align:center;">
                        <h2 style="color:maroon;">Predicted Car Price</h2>
                        <p style="font-size:36px;font-weight:bold;color:maroon;">₹{prediction[0]:,.2f}</p>
                        <p>Car Age: {user_df['car_age'][0]} years</p>
                        <p>Efficiency Score: {user_df['mileage_normalized'][0]:,.2f} km/year</p>
                    </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error in prediction: {e}")
        else:
            missing_fields = [col for col in user_df.columns if user_df[col].isnull().any()]
            st.error(f"Missing fields: {', '.join(missing_fields)}. Please fill all required fields.")

    # Back to Home button
    if st.sidebar.button("Back to Home"):
        navigate_to("Home")