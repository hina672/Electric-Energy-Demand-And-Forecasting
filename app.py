import datetime
import os
import re
import uuid
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import logging
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from prophet import Prophet
from werkzeug.utils import secure_filename
import pmdarima as pm
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ---------------- Flask app setup ----------------
app = Flask(__name__)
app.secret_key = 'EnergiXspecialKey.'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///energy_forecast.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.debug("Flask app and configurations set up.")

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the database
db = SQLAlchemy(app)

# ---------------- Models ----------------
class User(db.Model): 
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)  # (In production, use hashed passwords!)

class EnergyData(db.Model):
    __tablename__ = 'energy_data'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    file_path = db.Column(db.String(200), nullable=False)

class AnalysisSession(db.Model):
    __tablename__ = 'analysis_session'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    csv_filename = db.Column(db.String(150), nullable=False)
    place = db.Column(db.String(100), nullable=False)
    country = db.Column(db.String(100), nullable=False)
    consumer_type = db.Column(db.String(50), nullable=False)  # e.g. household, office, etc.
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    file_path = db.Column(db.String(200), nullable=False)  # Path to stored CSV
    model_choice = db.Column(db.String(20), nullable=True)  # "prophet" or "lstm"
    unique_id = db.Column(db.String(10), nullable=True)       # Unique identifier for graph file naming

with app.app_context():
    db.create_all()
logger.debug("Database initialized.")


# ---------------- Forecasting Functions ----------------

# Prophet Forecasting & Evaluation

def forecast_with_prophet(df, steps=30):
    logger.debug("Starting Prophet forecasting (daily)...")
    try:
        df = df.copy()
        df.rename(columns={'datetime': 'ds', 'EnergyConsumption': 'y'}, inplace=True)
        model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True,
                        changepoint_prior_scale=0.05)
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        if 'covid' in df.columns:
            model.add_regressor('covid')
        model.fit(df)
        future = model.make_future_dataframe(periods=steps, freq='D')
        if 'covid' in df.columns:
            future['covid'] = df['covid'].iloc[-1]
        forecast = model.predict(future)
        logger.debug("Prophet forecasting (daily) completed.")
        return forecast[['ds', 'yhat']].tail(steps)
    except Exception as e:
        logger.error(f"Prophet Error: {e}")
        return None

def evaluate_prophet_daily(df, steps=30):
    """
    Evaluates Prophet forecasting performance on a holdout set.
    Splits the DataFrame into train and test (last `steps` days as test),
    fits the model on train, forecasts on test, and returns both forecast and metrics.
    """
    logger.debug("Starting Prophet evaluation (daily)...")
    df = df.copy().reset_index()  # ensure datetime is a column
    df.rename(columns={'datetime': 'ds', 'EnergyConsumption': 'y'}, inplace=True)
    
    if len(df) < steps + 1:
        logger.error("Not enough data to evaluate Prophet with the given steps.")
        return None, None
    
    train = df.iloc[:-steps]
    test = df.iloc[-steps:]
    
    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True,
                    changepoint_prior_scale=0.05)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    if 'covid' in train.columns:
        model.add_regressor('covid')
    model.fit(train)
    
    future = model.make_future_dataframe(periods=steps, freq='D')
    if 'covid' in train.columns:
        future['covid'] = train['covid'].iloc[-1]
    forecast = model.predict(future)
    forecast_test = forecast.tail(steps)
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = mean_absolute_error(test['y'], forecast_test['yhat'])
    rmse = np.sqrt(mean_squared_error(test['y'], forecast_test['yhat']))
    rel_mae = (mae / test['y'].mean()) * 100
    rel_rmse = (rmse / test['y'].mean()) * 100
    metrics = {'MAE': mae, 'RMSE': rmse, 'Rel_MAE_pct': rel_mae, 'Rel_RMSE_pct': rel_rmse}
    
    logger.debug("Prophet evaluation (daily) completed.")
    return forecast_test[['ds','yhat']], metrics

def forecast_monthly_with_prophet(df, steps=12):
    logger.debug("Starting Prophet forecasting (monthly)...")
    try:
        df = df.copy()
        df.rename(columns={'datetime': 'ds', 'EnergyConsumption': 'y'}, inplace=True)
        model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True,
                        changepoint_prior_scale=0.05)
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        if 'covid' in df.columns:
            model.add_regressor('covid')
        model.fit(df)
        future = model.make_future_dataframe(periods=steps, freq='M')
        if 'covid' in df.columns:
            future['covid'] = df['covid'].iloc[-1]
        forecast = model.predict(future)
        logger.debug("Prophet forecasting (monthly) completed.")
        return forecast[['ds', 'yhat']].tail(steps)
    except Exception as e:
        logger.error(f"Prophet Monthly Error: {e}")
        return None

def evaluate_prophet_monthly(df, steps=12):
    """
    Evaluates Prophet forecasting performance on monthly aggregated data.
    Splits the DataFrame into train and test (last `steps` months as test),
    fits the model on train, forecasts on test, and returns both forecast and metrics.
    """
    logger.debug("Starting Prophet evaluation (monthly)...")
    df = df.copy().reset_index()
    df.rename(columns={'datetime': 'ds', 'EnergyConsumption': 'y'}, inplace=True)
    
    if len(df) < steps + 1:
        logger.error("Not enough data to evaluate Prophet monthly with the given steps.")
        return None, None
    
    train = df.iloc[:-steps]
    test = df.iloc[-steps:]
    
    model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True,
                    changepoint_prior_scale=0.05)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    if 'covid' in train.columns:
        model.add_regressor('covid')
    model.fit(train)
    
    future = model.make_future_dataframe(periods=steps, freq='M')
    if 'covid' in train.columns:
        future['covid'] = train['covid'].iloc[-1]
    forecast = model.predict(future)
    forecast_test = forecast.tail(steps)
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = mean_absolute_error(test['y'], forecast_test['yhat'])
    rmse = np.sqrt(mean_squared_error(test['y'], forecast_test['yhat']))
    rel_mae = (mae / test['y'].mean()) * 100
    rel_rmse = (rmse / test['y'].mean()) * 100
    metrics = {'MAE': mae, 'RMSE': rmse, 'Rel_MAE_pct': rel_mae, 'Rel_RMSE_pct': rel_rmse}
    
    logger.debug("Prophet evaluation (monthly) completed.")
    return forecast_test[['ds','yhat']], metrics


# LSTM Forecasting

def forecast_with_lstm(df, steps=30, seq_length=30, epochs=50, batch_size=16):
    logger.debug("Starting LSTM forecasting (daily)...")
    df = df.copy()
    data = df['EnergyConsumption'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i - seq_length:i, 0])
        y.append(scaled_data[i, 0])
    X = np.array(X)
    y = np.array(y)
    if X.size == 0:
        logger.error("Not enough data for daily LSTM forecast.")
        return None, {'MAE': None, 'RMSE': None, 'Rel_MAE_pct': None, 'Rel_RMSE_pct': None}
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
    model.add(LSTM(50))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    logger.debug("Training LSTM model...")
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)
    logger.debug("LSTM model training completed.")
    y_pred = model.predict(X_test)
    y_pred_inverse = scaler.inverse_transform(y_pred)
    y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = mean_absolute_error(y_test_inverse, y_pred_inverse)
    rmse = np.sqrt(mean_squared_error(y_test_inverse, y_pred_inverse))
    mean_actual = y_test_inverse.mean()
    rel_mae = (mae / mean_actual) * 100
    rel_rmse = (rmse / mean_actual) * 100
    metrics = {'MAE': mae, 'RMSE': rmse, 'Rel_MAE_pct': rel_mae, 'Rel_RMSE_pct': rel_rmse}
    forecast_input = scaled_data[-seq_length:]
    forecast_input = np.reshape(forecast_input, (1, seq_length, 1))
    forecast_list = []
    for _ in range(steps):
        pred = model.predict(forecast_input)
        forecast_list.append(pred[0, 0])
        forecast_input = np.append(forecast_input[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
    forecast_values = scaler.inverse_transform(np.array(forecast_list).reshape(-1, 1)).flatten()
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=steps)
    forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': forecast_values})
    logger.debug("LSTM forecasting (daily) completed.")
    return forecast_df, metrics

def forecast_with_lstm_monthly(df, steps=12, seq_length=12, epochs=50, batch_size=16):
    logger.debug("Starting LSTM forecasting (monthly)...")
    df = df.copy()
    data = df['EnergyConsumption'].values.reshape(-1, 1)
    if len(data) < seq_length:
        logger.error("Not enough data for monthly LSTM forecast.")
        return None, {'MAE': None, 'RMSE': None, 'Rel_MAE_pct': None, 'Rel_RMSE_pct': None}
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i - seq_length:i, 0])
        y.append(scaled_data[i, 0])
    X = np.array(X)
    y = np.array(y)
    if X.size == 0:
        logger.error("Not enough data after processing for monthly LSTM forecast.")
        return None, {'MAE': None, 'RMSE': None, 'Rel_MAE_pct': None, 'Rel_RMSE_pct': None}
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    train_size = int(len(X) * 0.8)
    if train_size == 0:
         logger.error("Not enough training data for monthly LSTM forecast.")
         return None, {'MAE': None, 'RMSE': None, 'Rel_MAE_pct': None, 'Rel_RMSE_pct': None}
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
    model.add(LSTM(50))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    logger.debug("Training LSTM model (monthly)...")
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)
    logger.debug("LSTM monthly model training completed.")
    y_pred = model.predict(X_test)
    y_pred_inverse = scaler.inverse_transform(y_pred)
    y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = mean_absolute_error(y_test_inverse, y_pred_inverse)
    rmse = np.sqrt(mean_squared_error(y_test_inverse, y_pred_inverse))
    mean_actual = y_test_inverse.mean()
    rel_mae = (mae / mean_actual) * 100
    rel_rmse = (rmse / mean_actual) * 100
    metrics = {'MAE': mae, 'RMSE': rmse, 'Rel_MAE_pct': rel_mae, 'Rel_RMSE_pct': rel_rmse}
    forecast_input = scaled_data[-seq_length:]
    forecast_input = np.reshape(forecast_input, (1, seq_length, 1))
    forecast_list = []
    for _ in range(steps):
        pred = model.predict(forecast_input)
        forecast_list.append(pred[0, 0])
        forecast_input = np.append(forecast_input[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
    forecast_values = scaler.inverse_transform(np.array(forecast_list).reshape(-1, 1)).flatten()
    future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=steps, freq='M')
    forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': forecast_values})
    logger.debug("LSTM forecasting (monthly) completed.")
    return forecast_df, metrics


# ---------------- Interactive Plot Functions (Plotly) ----------------

def generate_interactive_energy_plot(df, user_id, unique_id):
    logger.debug("Generating interactive energy plot...")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['EnergyConsumption'],
        mode='lines+markers',
        name='Energy Data',
        hovertemplate="Date: %{x}<br>Energy: %{y:.2f} kWh<extra></extra>"
    ))
    fig.update_layout(
        title="Energy Consumption Data",
        xaxis_title="Date",
        yaxis_title="Energy Consumption (kWh)",
        template="plotly_dark"
    )
    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], f"user_{user_id}")
    os.makedirs(user_folder, exist_ok=True)
    interactive_file = os.path.join(user_folder, f'energy_data_{unique_id}.html')
    fig.write_html(interactive_file)
    logger.debug("Interactive energy plot generated.")
    if os.path.exists(interactive_file):
        logger.debug(f"File exists: {interactive_file}")
    else:
        logger.error(f"File not found: {interactive_file}")
    logger.debug("Interactive energy plot generated.")
    return f"uploads/user_{user_id}/energy_data_{unique_id}.html"

def generate_interactive_forecast_plot(forecast_df, user_id, unique_id, model_name):
    logger.debug(f"Generating interactive forecast plot for {model_name}...")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat'],
        mode='lines+markers',
        name=f'{model_name} Forecast',
        hovertemplate="Date: %{x}<br>Energy: %{y:.2f} kWh<extra></extra>"
    ))
    fig.update_layout(
        title=f"Predicted Daily Energy Consumption (kWh) using {model_name}",
        xaxis_title="Date",
        yaxis_title="Energy Consumption (kWh)",
        template="plotly_dark"
    )
    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], f"user_{user_id}")
    os.makedirs(user_folder, exist_ok=True)
    interactive_file = os.path.join(user_folder, f'daily_forecast_{unique_id}.html')
    fig.write_html(interactive_file)
    logger.debug("Interactive energy plot generated.")
    if os.path.exists(interactive_file):
        logger.debug(f"File exists: {interactive_file}")
    else:
        logger.error(f"File not found: {interactive_file}")
    logger.debug("Interactive forecast plot generated.")
    return f"uploads/user_{user_id}/daily_forecast_{unique_id}.html"

def generate_interactive_monthly_plot(prophet_forecast, user_id, unique_id):
    logger.debug("Generating interactive monthly forecast plot...")
    fig = go.Figure()
    if prophet_forecast is not None:
        fig.add_trace(go.Scatter(
            x=prophet_forecast['ds'],
            y=prophet_forecast['yhat'],
            mode='lines+markers',
            name='Prophet Forecast',
            hovertemplate="Date: %{x}<br>Energy: %{y:.2f} kWh<extra></extra>"
        ))
    fig.update_layout(
        title="Predicted Monthly Energy Consumption (kWh) using Prophet",
        xaxis_title="Date",
        yaxis_title="Energy Consumption (kWh)",
        template="plotly_dark"
    )
    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], f"user_{user_id}")
    os.makedirs(user_folder, exist_ok=True)
    interactive_file = os.path.join(user_folder, f'monthly_forecast_{unique_id}.html')
    fig.write_html(interactive_file)
    logger.debug("Interactive energy plot generated.")
    if os.path.exists(interactive_file):
        logger.debug(f"File exists: {interactive_file}")
    else:
        logger.error(f"File not found: {interactive_file}")
    logger.debug("Interactive monthly forecast plot generated.")
    return f"uploads/user_{user_id}/monthly_forecast_{unique_id}.html"

# ---------------- Utility Functions for Data Parsing ----------------

def preprocess_datetime_str(dt_str, default_year=None):
    dt_str = dt_str.strip()
    if default_year is None:
        default_year = datetime.datetime.now().year
    if not re.search(r'\b\d{4}\b', dt_str):
        dt_str = dt_str + " " + str(default_year)
    return dt_str

def parse_uploaded_csv(file):
    logger.debug("Parsing uploaded CSV file...")
    try:
        df = pd.read_csv(file)
    except Exception as e:
        file.seek(0)
        df = pd.read_csv(file, sep=';')
    
    if {'datetime', 'EnergyConsumption'}.issubset(df.columns):
        cols = ['datetime', 'EnergyConsumption']
        if 'covid' in df.columns:
            cols.append('covid')
        df = df[cols]
    elif {'date', 'consumption'}.issubset(df.columns):
        cols = ['date', 'consumption']
        if 'covid' in df.columns:
            cols.append('covid')
        df = df[cols]
        df.rename(columns={'date': 'datetime', 'consumption': 'EnergyConsumption'}, inplace=True)
    else:
        date_col = next((col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()), None)
        cons_col = next((col for col in df.columns if 'consum' in col.lower()), None)
        if date_col and cons_col:
            cols = [date_col, cons_col]
            if 'covid' in df.columns:
                cols.append('covid')
            df = df[cols]
            df.rename(columns={date_col: 'datetime', cons_col: 'EnergyConsumption'}, inplace=True)
        else:
            raise ValueError("CSV does not contain recognizable datetime and consumption columns.")
    
    df['datetime'] = df['datetime'].apply(lambda s: preprocess_datetime_str(s, default_year=2023))
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime'])
    df.sort_values('datetime', inplace=True)
    df.set_index('datetime', inplace=True)
    
    if 'covid' in df.columns:
        df = df[['EnergyConsumption', 'covid']]
    else:
        df = df[['EnergyConsumption']]
    
    logger.debug(f"CSV parsing complete. DataFrame shape: {df.shape}")
    return df

def infer_data_frequency(df):
    time_diffs = df.index.to_series().diff().dropna()
    if time_diffs.empty:
        return None
    median_diff = time_diffs.median()
    if median_diff <= pd.Timedelta(minutes=1):
        return 'T'
    elif median_diff <= pd.Timedelta(minutes=10):
        return '10T'
    elif median_diff <= pd.Timedelta(hours=1):
        return 'H'
    elif median_diff <= pd.Timedelta(days=1):
        return 'D'
    else:
        return 'M'

# ---------------- Web Data Retrieval Functions ----------------

SERP_API_KEY = "7f3a30c58441947068a1ff3b9213d7d6fe898671bd1b2277531343f47fd787a9"  # Replace with your actual SerpAPI key

def google_search(query):
    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "hl": "en",
        "gl": "us",
        "api_key": SERP_API_KEY
    }
    response = requests.get(url, params=params)
    data = response.json()
    if "organic_results" in data:
        results = [item.get("snippet", "No data found.") for item in data["organic_results"][:3]]
        return " ".join(results)
    return "No data found."

def get_energy_trends(country):
    query = f"per year increase in electricity consumption in {country}"
    return google_search(query)

def get_electricity_cost(city, country):
    query = f"cost of electricity per kWh in {city}, {country}"
    return google_search(query)

# ---------------- Flask Routes ----------------

@app.route('/')
def home():
    logger.debug("Home route accessed.")
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        logger.debug("Signup form submitted.")
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        logger.debug(f"Received signup data: Username={username}, Email={email}")
        
        # Email pattern validation
        email_pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
        if not re.match(email_pattern, email):
            flash('Invalid email format!', 'danger')
            return redirect(url_for('signup'))
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists!', 'danger')
            return redirect(url_for('signup'))
        if User.query.filter_by(email=email).first():
            flash('Email already registered!', 'danger')
            return redirect(url_for('signup'))
        
        new_user = User(username=username, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        logger.debug("User added to the database.")
        
        flash('Signup successful! You can now log in.', 'success')
        return redirect(url_for('signin'))
    
    return render_template('signup.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email, password=password).first()
        
        if user:
            session.clear()
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials!', 'danger')
    
    return render_template('signin.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('home'))

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    logger.debug("Dashboard route accessed.")
    if 'user_id' not in session:
        flash('Please log in to access the dashboard.', 'danger')
        return redirect(url_for('signin'))
    user_id = session['user_id']
    user_files = EnergyData.query.filter_by(user_id=user_id).all()
    archives = AnalysisSession.query.filter_by(user_id=user_id).order_by(AnalysisSession.timestamp.desc()).all()

    if request.method == 'POST' and 'upload_csv' in request.form:
        logger.debug("Processing CSV upload...")
        if 'file' not in request.files:
            flash('No file uploaded!', 'danger')
            return redirect(url_for('dashboard'))
        file = request.files['file']
        if file.filename == '':
            flash('No file selected!', 'danger')
            return redirect(url_for('dashboard'))
        if not file.filename.endswith('.csv'):
            flash('Invalid file format! Please upload a CSV file.', 'danger')
            return redirect(url_for('dashboard'))
        try:
            pd.read_csv(file)
            logger.debug("CSV file read successfully.")
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            flash(f'Error reading file: {e}', 'danger')
            return redirect(url_for('dashboard'))
        country = request.form.get('country')
        city = request.form.get('city')
        consumer_type = request.form.get('consumer_type')
        model_choice = request.form.get('model_choice')
        
        user_folder = os.path.join(app.config['UPLOAD_FOLDER'], f"user_{user_id}")
        os.makedirs(user_folder, exist_ok=True)
        filename = secure_filename(file.filename)
        file_path = os.path.join(user_folder, filename)
        try:
            file.seek(0)
            file.save(file_path)
            logger.debug("CSV file saved successfully.")
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            flash(f'Error saving file: {e}', 'danger')
            return redirect(url_for('dashboard'))
        
        new_file = EnergyData(user_id=user_id, file_path=file_path)
        db.session.add(new_file)
        db.session.commit()
        
        unique_id = str(uuid.uuid4())[:8]
        new_session = AnalysisSession(user_id=user_id, csv_filename=filename,
                                      place=city, country=country,
                                      consumer_type=consumer_type, file_path=file_path,
                                      model_choice=model_choice, unique_id=unique_id)
        db.session.add(new_session)
        db.session.commit()
        logger.debug("New analysis session recorded in database.")
        
        try:
            logger.debug("Parsing CSV file for forecasting...")
            df = parse_uploaded_csv(file_path)
            logger.debug(f"CSV file parsed. Data shape: {df.shape}")
        except Exception as e:
            logger.error(f"Error processing CSV file: {e}")
            flash(f'Error processing CSV file: {e}', 'danger')
            return redirect(url_for('dashboard'))
        
                # Aggregation: detect if data is monthly-only
        freq = infer_data_frequency(df)
        logger.debug(f"Inferred frequency: {freq}")
        if freq == 'M':
            # Data is already monthly
            monthly_data = df.copy()
            daily_data = None
        else:
            # For minute, 10-min, hourly or daily data, aggregate into daily and monthly sums.
            daily_data = df.resample('D').sum()
            monthly_data = df.resample('M').sum()
        logger.debug(f"After aggregation: Daily shape {daily_data.shape if daily_data is not None else 'None'}, Monthly shape {monthly_data.shape}")
        
        unique_id = str(uuid.uuid4())[:8]
        energy_data_plot = generate_interactive_energy_plot(df, user_id, unique_id)
        
        if model_choice == 'lstm':
            # LSTM branch remains unchanged.
            if freq == 'M':
                forecast_df_monthly, monthly_metrics = forecast_with_lstm_monthly(monthly_data, steps=12)
                monthly_forecast_plot = generate_interactive_monthly_plot(forecast_df_monthly, user_id, unique_id)
                daily_forecast_plot = None
            else:
                forecast_df_daily, daily_metrics = forecast_with_lstm(daily_data, steps=30)
                daily_forecast_plot = generate_interactive_forecast_plot(forecast_df_daily, user_id, unique_id, "LSTM Daily")
                forecast_df_monthly, monthly_metrics = forecast_with_lstm_monthly(monthly_data, steps=12)
                monthly_forecast_plot = generate_interactive_monthly_plot(forecast_df_monthly, user_id, unique_id)
        else:
            # Prophet branch: use evaluation functions to get error metrics.
            if freq == 'M':
                forecast_df_monthly, prophet_monthly_metrics = evaluate_prophet_monthly(monthly_data.reset_index()[['datetime', 'EnergyConsumption']], steps=12)
                monthly_forecast_plot = generate_interactive_monthly_plot(forecast_df_monthly, user_id, unique_id)
                daily_forecast_plot = None
            else:
                forecast_df_daily, prophet_daily_metrics = evaluate_prophet_daily(daily_data.reset_index()[['datetime', 'EnergyConsumption']], steps=30)
                daily_forecast_plot = generate_interactive_forecast_plot(forecast_df_daily, user_id, unique_id, "Prophet")
                forecast_df_monthly, prophet_monthly_metrics = evaluate_prophet_monthly(monthly_data.reset_index()[['datetime', 'EnergyConsumption']], steps=12)
                monthly_forecast_plot = generate_interactive_monthly_plot(forecast_df_monthly, user_id, unique_id)
        
        logger.debug("Forecasting and plotting completed after CSV upload.")
        return render_template('dashboard.html', 
                               energy_data_plot=energy_data_plot,
                               daily_forecast_plot=daily_forecast_plot,
                               monthly_forecast_plot=monthly_forecast_plot,
                               archives=archives,
                               lstm_daily_metrics=daily_metrics if model_choice=='lstm' and daily_data is not None else None,
                               lstm_monthly_metrics=monthly_metrics if model_choice=='lstm' else None,
                               prophet_daily_metrics=prophet_daily_metrics if model_choice!='lstm' and daily_data is not None else None,
                               prophet_monthly_metrics=prophet_monthly_metrics if model_choice!='lstm' else None)
    
    energy_data_plot = daily_forecast_plot = monthly_forecast_plot = None
    if archives:
        # Use the unique_id from the latest AnalysisSession in the archive
        latest_session = archives[0]
        unique_id = latest_session.unique_id  
        file_path = latest_session.file_path
        try:
            df = parse_uploaded_csv(file_path)
            logger.debug(f"Parsed CSV from latest archive. Shape: {df.shape}")
        except Exception as e:
            logger.error(f"Error processing CSV file: {e}")
            flash(f'Error processing CSV file: {e}', 'danger')
            return redirect(url_for('dashboard'))
        
        freq = infer_data_frequency(df)
        logger.debug(f"Inferred frequency: {freq}")
        if freq == 'M':
            monthly_data = df.copy()
            daily_data = None
        else:
            daily_data = df.resample('D').sum()
            monthly_data = df.resample('M').sum()
        logger.debug(f"After aggregation: Daily shape {daily_data.shape if daily_data is not None else 'None'}, Monthly shape {monthly_data.shape}")
        
        # Generate the energy data plot using the archived unique_id
        energy_data_plot = generate_interactive_energy_plot(df, user_id, unique_id)
        if daily_data is not None and len(daily_data) > 0:
            # For Prophet (adjust as needed for LSTM)
            prophet_daily_forecast = forecast_with_prophet(daily_data.reset_index()[['datetime', 'EnergyConsumption']], steps=30)
            daily_forecast_plot = generate_interactive_forecast_plot(prophet_daily_forecast, user_id, unique_id, "Prophet")
        if monthly_data is not None and len(monthly_data) > 0:
            prophet_monthly_forecast = forecast_monthly_with_prophet(monthly_data.reset_index()[['datetime', 'EnergyConsumption']], steps=12)
            monthly_forecast_plot = generate_interactive_monthly_plot(prophet_monthly_forecast, user_id, unique_id)
    
    return render_template('dashboard.html', 
                           energy_data_plot=energy_data_plot,
                           daily_forecast_plot=daily_forecast_plot, 
                           monthly_forecast_plot=monthly_forecast_plot,
                           archives=archives)



@app.route('/change_username', methods=['POST'])
def change_username():
    if 'user_id' not in session:
        flash('Please log in to update your username.', 'danger')
        return redirect(url_for('signin'))
    new_username = request.form.get('new_username')
    if not new_username:
        flash('Please enter a new username.', 'danger')
        return redirect(url_for('dashboard'))
    if User.query.filter_by(username=new_username).first():
        flash('Username already exists!', 'danger')
        return redirect(url_for('dashboard'))
    user = User.query.get(session['user_id'])
    user.username = new_username
    db.session.commit()
    session['username'] = new_username
    flash('Username updated successfully!', 'success')
    return redirect(url_for('dashboard'))

@app.route('/archive')
def archive():
    if 'user_id' not in session:
        flash('Please log in to view your archives.', 'danger')
        return redirect(url_for('signin'))
    user_id = session['user_id']
    archives = AnalysisSession.query.filter_by(user_id=user_id).order_by(AnalysisSession.timestamp.desc()).all()
    return render_template('archive.html', archives=archives)

@app.route('/archive/delete/<int:session_id>', methods=['POST'])
def delete_archive(session_id):
    if 'user_id' not in session:
        flash('Please log in.', 'danger')
        return redirect(url_for('signin'))
    archive_entry = AnalysisSession.query.get_or_404(session_id)
    if os.path.exists(archive_entry.file_path):
        try:
            os.remove(archive_entry.file_path)
        except Exception as e:
            flash(f'Error deleting file: {e}', 'danger')
            return redirect(url_for('archive'))
    db.session.delete(archive_entry)
    db.session.commit()
    flash('Archive deleted successfully.', 'success')
    return redirect(url_for('archive'))

@app.route('/results/<int:session_id>')
def results(session_id):
    if 'user_id' not in session:
        flash('Please log in.', 'danger')
        return redirect(url_for('signin'))
    session_entry = AnalysisSession.query.get_or_404(session_id)
    try:
        df = parse_uploaded_csv(session_entry.file_path)
    except Exception as e:
        flash(f'Error processing archived CSV: {e}', 'danger')
        return redirect(url_for('archive'))
    
    daily_data = df.resample('D').sum()
    prophet_daily_forecast = forecast_with_prophet(daily_data.reset_index()[['datetime', 'EnergyConsumption']], steps=30)
    
    energy_trend = get_energy_trends(session_entry.country)
    electricity_cost = get_electricity_cost(session_entry.place, session_entry.country)
    
    try:
        annual_increase_percent = float(''.join([c for c in energy_trend if c.isdigit() or c == '.']))
    except ValueError:
        annual_increase_percent = 5.0
    try:
        price_per_kwh = float(''.join([c for c in electricity_cost if c.isdigit() or c == '.']))
    except ValueError:
        price_per_kwh = 0.15

    future_costs = {}
    if prophet_daily_forecast is not None:
        for idx, row in prophet_daily_forecast.iterrows():
            adjusted_consumption = row['yhat'] * (1 + (annual_increase_percent / 100))
            estimated_cost = adjusted_consumption * price_per_kwh
            future_costs[row['ds'].date()] = {
                "forecasted_energy_kwh": adjusted_consumption,
                "estimated_cost": estimated_cost
            }
    
    insights = {
        "forecast": prophet_daily_forecast.to_dict(orient='records') if prophet_daily_forecast is not None else {},
        "regional_insights": {
            "energy_trend": energy_trend,
            "electricity_cost": electricity_cost,
            "adjusted_forecast": future_costs
        }
    }
    return render_template('results.html', session_entry=session_entry, insights=insights)

@app.route('/delete_account', methods=['POST'])
def delete_account():
    if 'user_id' not in session:
        flash('You need to be logged in to delete your account.', 'danger')
        return redirect(url_for('signin'))
    user_id = session['user_id']
    user_files = EnergyData.query.filter_by(user_id=user_id).all()
    for file_record in user_files:
        if os.path.exists(file_record.file_path):
            try:
                os.remove(file_record.file_path)
            except Exception as e:
                flash(f'Error deleting file {file_record.file_path}: {e}', 'danger')
    EnergyData.query.filter_by(user_id=user_id).delete()
    AnalysisSession.query.filter_by(user_id=user_id).delete()
    User.query.filter_by(id=user_id).delete()
    db.session.commit()
    session.pop('user_id', None)
    session.pop('username', None)
    flash('Your account and all related data have been deleted.', 'success')
    return redirect(url_for('signin'))

@app.route('/test')
def test():
    print("Test route reached")
    return "Server is working!"

if __name__ == '__main__':
    logger.debug("Starting Flask app...")
    app.run(debug=True)
