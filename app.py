import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import ta

st.set_page_config(page_title="Tesla Stock Predictor", layout="wide")

st.title("📈 Tesla Stock Price Prediction App")
st.markdown("Predict Tesla's closing price using Deep Learning models.")

# Sidebar Settings
st.sidebar.header("Configuration")

# File Uploader
uploaded_file = st.sidebar.file_uploader("Upload TSLA.csv", type=["csv"])

# Model Selection
model_options = ['SimpleRNN', 'LSTM', 'StackedLSTM', 'TunedLSTM']
selected_model = st.sidebar.selectbox("Select Model", model_options)

# Horizon
horizon_options = [1, 5, 10]
selected_horizon = st.sidebar.selectbox("Prediction Horizon (Days)", horizon_options)

# Lookback Window
lookback_window = st.sidebar.slider("Lookback Window (Days)", min_value=30, max_value=90, value=60, step=10)

def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col='Date')
    
    st.subheader("Raw Data Preview")
    st.dataframe(df.tail())
    
    # Preprocessing
    df.drop_duplicates(inplace=True)
    df.ffill(inplace=True)
    
    # Feature Engineering
    try:
        df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
        macd = ta.trend.MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        bb = ta.volatility.BollingerBands(close=df['Close'], window=20)
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()
        df['SMA_7'] = ta.trend.SMAIndicator(close=df['Close'], window=7).sma_indicator()
        df['SMA_21'] = ta.trend.SMAIndicator(close=df['Close'], window=21).sma_indicator()
        df.dropna(inplace=True)
    except Exception as e:
        st.error(f"Error calculating technical indicators: {e}")
        st.stop()
        
    features_array = df.values
    target_array = df[['Adj Close']].values
    
    feature_scaler = MinMaxScaler(feature_range=(0,1))
    scaled_features = feature_scaler.fit_transform(features_array)
    
    target_scaler = MinMaxScaler(feature_range=(0,1))
    scaled_target = target_scaler.fit_transform(target_array)
    
    # We require at least lookback_window + horizon amount of data to make 1 prediction
    if len(df) < lookback_window + selected_horizon:
        st.warning("Not enough data to apply the lookback window and prediction horizon.")
    else:
        st.info("Loading Model...")
        try:
            import tensorflow as tf
            from tensorflow.keras.models import load_model
            model_path = f"models/model_{selected_model}_{selected_horizon}d.h5"
            try:
                # Use compile=False to avoid serialization issues with metrics/optimizers during inference
                model = load_model(model_path, compile=False)
                st.success(f"Successfully loaded {model_path}.")
            except OSError:
                st.error(f"Model file '{model_path}' not found. Please ensure you have run the Jupyter Notebook to train and save the models first.")
                model = None
                
            if model is not None:
                # Prepare sequences for prediction
                def create_sequences(data_features, data_target, window, horiz):
                    X, y = [], []
                    for i in range(window, len(data_features) - horiz + 1):
                        X.append(data_features[i - window:i])
                        y.append(data_target[i + horiz - 1])
                    return np.array(X), np.array(y)
                
                X, y_true_scaled = create_sequences(scaled_features, scaled_target, lookback_window, selected_horizon)
                
                # Make Predictions
                y_pred_scaled = model.predict(X)
                
                # Inverse Transform
                y_pred = target_scaler.inverse_transform(y_pred_scaled).flatten()
                y_true = target_scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
                
                # Align dates
                prediction_dates = df.index[lookback_window + selected_horizon - 1 :]
                
                # Metrics
                st.subheader("Model Performance Metrics")
                rmse, mae, r2 = calculate_metrics(y_true, y_pred)
                m1, m2, m3 = st.columns(3)
                m1.metric("RMSE", f"{rmse:.4f}")
                m2.metric("MAE", f"{mae:.4f}")
                m3.metric("R² Score", f"{r2:.4f}")
                
                # Plotly Chart
                st.subheader(f"Historical Prediction Chart ({selected_model} - {selected_horizon}d)")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=prediction_dates, y=y_true, mode='lines', name='Actual Price', line=dict(color='black')))
                fig.add_trace(go.Scatter(x=prediction_dates, y=y_pred, mode='lines', name=f'{selected_model} Prediction', line=dict(color='blue', dash='dot')))
                fig.update_layout(xaxis_title="Date", yaxis_title="Adj Close Price", hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)
                
                # Predict Future Unknown Days
                st.subheader("Predict Future Days")
                if st.button("Predict Next N Days"):
                    # We need the last `lookback_window` available scaled features
                    last_sequence = scaled_features[-lookback_window:]
                    last_sequence = last_sequence.reshape(1, lookback_window, scaled_features.shape[1])
                    
                    future_pred_scaled = model.predict(last_sequence)
                    future_pred = target_scaler.inverse_transform(future_pred_scaled).flatten()[0]
                    
                    target_date = df.index[-1] + pd.Timedelta(days=selected_horizon)
                    st.success(f"Predicted Adj Close for {target_date.strftime('%Y-%m-%d')} is **${future_pred:.2f}**")
                    
        except ImportError as e:
            st.error("TensorFlow is not available. The local environment's DLL failed to initialize. Please try running the setup in an environment where TensorFlow operates correctly.")
else:
    st.info("Please upload TSLA.csv using the sidebar to continue.")
