import os
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.utils
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
import warnings
import datetime
warnings.filterwarnings('ignore')

# Set HuggingFace mirror for China
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from model import Kronos, KronosTokenizer, KronosPredictor
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("Warning: Kronos model cannot be imported, will use simulated data for demonstration")

from db import save_prediction, get_predictions_history, get_prediction_by_id, get_predictions_by_ids, delete_prediction, save_llm_analysis, get_llm_analysis, save_chat_message, get_chat_history
from llm_service import analyze_trend, analyze_parameter_impact, chat_with_context, check_llm_available, get_llm_config

app = Flask(__name__)
CORS(app)

# Global variables to store models
tokenizer = None
model = None
predictor = None

# Available model configurations
AVAILABLE_MODELS = {
    'kronos-mini': {
        'name': 'Kronos-mini',
        'model_id': 'NeoQuasar/Kronos-mini',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-2k',
        'context_length': 2048,
        'params': '4.1M',
        'description': 'Lightweight model, suitable for fast prediction'
    },
    'kronos-small': {
        'name': 'Kronos-small',
        'model_id': 'NeoQuasar/Kronos-small',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '24.7M',
        'description': 'Small model, balanced performance and speed'
    },
    'kronos-base': {
        'name': 'Kronos-base',
        'model_id': 'NeoQuasar/Kronos-base',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '102.3M',
        'description': 'Base model, provides better prediction quality'
    }
}

def load_data_files():
    """Scan data directory and return available data files"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    data_files = []
    
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith(('.csv', '.feather')):
                file_path = os.path.join(data_dir, file)
                file_size = os.path.getsize(file_path)
                data_files.append({
                    'name': file,
                    'path': file_path,
                    'size': f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.1f} MB"
                })
    
    return data_files

def load_data_file(file_path):
    """Load data file"""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.feather'):
            df = pd.read_feather(file_path)
        else:
            return None, "Unsupported file format"
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return None, f"Missing required columns: {required_cols}"
        
        # Process timestamp column
        if 'timestamps' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamps'])
        elif 'timestamp' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamp'])
        elif 'date' in df.columns:
            # If column name is 'date', rename it to 'timestamps'
            df['timestamps'] = pd.to_datetime(df['date'])
        else:
            # If no timestamp column exists, create one
            df['timestamps'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1H')
        
        # Ensure numeric columns are numeric type
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Process volume column (optional)
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # Process amount column (optional, but not used for prediction)
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        # Remove rows containing NaN values
        df = df.dropna()
        
        return df, None
        
    except Exception as e:
        return None, f"Failed to load file: {str(e)}"

def save_prediction_results(file_path, prediction_type, prediction_results, actual_data, input_data, prediction_params):
    """Save prediction results to file"""
    try:
        # Create prediction results directory
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prediction_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'prediction_{timestamp}.json'
        filepath = os.path.join(results_dir, filename)
        
        # Prepare data for saving
        save_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'file_path': file_path,
            'prediction_type': prediction_type,
            'prediction_params': prediction_params,
            'input_data_summary': {
                'rows': len(input_data),
                'columns': list(input_data.columns),
                'price_range': {
                    'open': {'min': float(input_data['open'].min()), 'max': float(input_data['open'].max())},
                    'high': {'min': float(input_data['high'].min()), 'max': float(input_data['high'].max())},
                    'low': {'min': float(input_data['low'].min()), 'max': float(input_data['low'].max())},
                    'close': {'min': float(input_data['close'].min()), 'max': float(input_data['close'].max())}
                },
                'last_values': {
                    'open': float(input_data['open'].iloc[-1]),
                    'high': float(input_data['high'].iloc[-1]),
                    'low': float(input_data['low'].iloc[-1]),
                    'close': float(input_data['close'].iloc[-1])
                }
            },
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'analysis': {}
        }
        
        # If actual data exists, perform comparison analysis
        if actual_data and len(actual_data) > 0:
            # Calculate continuity analysis
            if len(prediction_results) > 0 and len(actual_data) > 0:
                last_pred = prediction_results[0]  # First prediction point
            first_actual = actual_data[0]      # First actual point
                
            save_data['analysis']['continuity'] = {
                    'last_prediction': {
                        'open': last_pred['open'],
                        'high': last_pred['high'],
                        'low': last_pred['low'],
                        'close': last_pred['close']
                    },
                    'first_actual': {
                        'open': first_actual['open'],
                        'high': first_actual['high'],
                        'low': first_actual['low'],
                        'close': first_actual['close']
                    },
                    'gaps': {
                        'open_gap': abs(last_pred['open'] - first_actual['open']),
                        'high_gap': abs(last_pred['high'] - first_actual['high']),
                        'low_gap': abs(last_pred['low'] - first_actual['low']),
                        'close_gap': abs(last_pred['close'] - first_actual['close'])
                    },
                    'gap_percentages': {
                        'open_gap_pct': (abs(last_pred['open'] - first_actual['open']) / first_actual['open']) * 100,
                        'high_gap_pct': (abs(last_pred['high'] - first_actual['high']) / first_actual['high']) * 100,
                        'low_gap_pct': (abs(last_pred['low'] - first_actual['low']) / first_actual['low']) * 100,
                        'close_gap_pct': (abs(last_pred['close'] - first_actual['close']) / first_actual['close']) * 100
                    }
                }
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"Prediction results saved to: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"Failed to save prediction results: {e}")
        return None

def create_prediction_chart(df, pred_df, lookback, pred_len, actual_df=None, historical_start_idx=0):
    """Create prediction chart"""
    # Use specified historical data start position, not always from the beginning of df
    if historical_start_idx + lookback + pred_len <= len(df):
        # Display lookback historical points + pred_len prediction points starting from specified position
        historical_df = df.iloc[historical_start_idx:historical_start_idx+lookback]
        prediction_range = range(historical_start_idx+lookback, historical_start_idx+lookback+pred_len)
    else:
        # If data is insufficient, adjust to maximum available range
        available_lookback = min(lookback, len(df) - historical_start_idx)
        available_pred_len = min(pred_len, max(0, len(df) - historical_start_idx - available_lookback))
        historical_df = df.iloc[historical_start_idx:historical_start_idx+available_lookback]
        prediction_range = range(historical_start_idx+available_lookback, historical_start_idx+available_lookback+available_pred_len)
    
    # Create chart
    fig = go.Figure()
    
    # Add historical data (candlestick chart)
    fig.add_trace(go.Candlestick(
        x=list(historical_df['timestamps']) if 'timestamps' in historical_df.columns else list(historical_df.index),
        open=list(historical_df['open']),
        high=list(historical_df['high']),
        low=list(historical_df['low']),
        close=list(historical_df['close']),
        name='历史数据',
        increasing_line_color='#26A69A',
        decreasing_line_color='#EF5350'
    ))
    
    # Add prediction data (candlestick chart)
    if pred_df is not None and len(pred_df) > 0:
        # Calculate prediction data timestamps - ensure continuity with historical data
        if 'timestamps' in df.columns and len(historical_df) > 0:
            # Start from the last timestamp of historical data, create prediction timestamps with the same time interval
            last_timestamp = historical_df['timestamps'].iloc[-1]
            time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0] if len(df) > 1 else pd.Timedelta(hours=1)
            
            pred_timestamps = pd.date_range(
                start=last_timestamp + time_diff,
                periods=len(pred_df),
                freq=time_diff
            )
        else:
            # If no timestamps, use index
            pred_timestamps = range(len(historical_df), len(historical_df) + len(pred_df))
        
        fig.add_trace(go.Candlestick(
            x=list(pred_timestamps),
            open=list(pred_df['open']),
            high=list(pred_df['high']),
            low=list(pred_df['low']),
            close=list(pred_df['close']),
            name='预测数据',
            increasing_line_color='#66BB6A',
            decreasing_line_color='#FF7043'
        ))
    
    # Add actual data for comparison (if exists)
    if actual_df is not None and len(actual_df) > 0:
        # Actual data should be in the same time period as prediction data
        if 'timestamps' in df.columns:
            # Actual data should use the same timestamps as prediction data to ensure time alignment
            if 'pred_timestamps' in locals():
                actual_timestamps = pred_timestamps
            else:
                # If no prediction timestamps, calculate from the last timestamp of historical data
                if len(historical_df) > 0:
                    last_timestamp = historical_df['timestamps'].iloc[-1]
                    time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0] if len(df) > 1 else pd.Timedelta(hours=1)
                    actual_timestamps = pd.date_range(
                        start=last_timestamp + time_diff,
                        periods=len(actual_df),
                        freq=time_diff
                    )
                else:
                    actual_timestamps = range(len(historical_df), len(historical_df) + len(actual_df))
        else:
            actual_timestamps = range(len(historical_df), len(historical_df) + len(actual_df))
        
        fig.add_trace(go.Candlestick(
            x=list(actual_timestamps),
            open=list(actual_df['open']),
            high=list(actual_df['high']),
            low=list(actual_df['low']),
            close=list(actual_df['close']),
            name='实际数据',
            increasing_line_color='#FF9800',
            decreasing_line_color='#F44336'
        ))
    
    # Update layout
    fig.update_layout(
        title='Kronos 金融预测结果',
        xaxis_title='时间',
        yaxis_title='价格',
        template='plotly_white',
        height=600,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    fig.update_xaxes(type='date')
    fig.update_yaxes(fixedrange=False)
    
    # Set Y-axis range based on all price data
    all_prices = []
    if len(historical_df) > 0:
        all_prices.extend(historical_df['open'].tolist())
        all_prices.extend(historical_df['high'].tolist())
        all_prices.extend(historical_df['low'].tolist())
        all_prices.extend(historical_df['close'].tolist())
    if pred_df is not None and len(pred_df) > 0:
        all_prices.extend(pred_df['open'].tolist())
        all_prices.extend(pred_df['high'].tolist())
        all_prices.extend(pred_df['low'].tolist())
        all_prices.extend(pred_df['close'].tolist())
    if actual_df is not None and len(actual_df) > 0:
        all_prices.extend(actual_df['open'].tolist())
        all_prices.extend(actual_df['high'].tolist())
        all_prices.extend(actual_df['low'].tolist())
        all_prices.extend(actual_df['close'].tolist())
    
    if all_prices:
        min_price = min(all_prices)
        max_price = max(all_prices)
        padding = (max_price - min_price) * 0.1
        fig.update_yaxes(range=[min_price - padding, max_price + padding])
    
    # Ensure x-axis time continuity
    if 'timestamps' in historical_df.columns:
        # Get all timestamps and sort them
        all_timestamps = []
        if len(historical_df) > 0:
            all_timestamps.extend(historical_df['timestamps'])
        if 'pred_timestamps' in locals():
            all_timestamps.extend(pred_timestamps)
        if 'actual_timestamps' in locals():
            all_timestamps.extend(actual_timestamps)
        
        if all_timestamps:
            all_timestamps = sorted(all_timestamps)
            fig.update_xaxes(
                range=[all_timestamps[0], all_timestamps[-1]],
                rangeslider_visible=False,
                type='date'
            )
    
    return fig.to_json()

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/api/data-files')
def get_data_files():
    """Get available data file list"""
    data_files = load_data_files()
    return jsonify(data_files)

@app.route('/api/load-data', methods=['POST'])
def load_data():
    """Load data file"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        
        if not file_path:
            return jsonify({'error': 'File path cannot be empty'}), 400
        
        df, error = load_data_file(file_path)
        if error:
            return jsonify({'error': error}), 400
        
        # Detect data time frequency
        def detect_timeframe(df):
            if len(df) < 2:
                return "Unknown"
            
            time_diffs = []
            for i in range(1, min(10, len(df))):  # Check first 10 time differences
                diff = df['timestamps'].iloc[i] - df['timestamps'].iloc[i-1]
                time_diffs.append(diff)
            
            if not time_diffs:
                return "Unknown"
            
            # Calculate average time difference
            avg_diff = sum(time_diffs, pd.Timedelta(0)) / len(time_diffs)
            
            # Convert to readable format
            if avg_diff < pd.Timedelta(minutes=1):
                return f"{avg_diff.total_seconds():.0f} seconds"
            elif avg_diff < pd.Timedelta(hours=1):
                return f"{avg_diff.total_seconds() / 60:.0f} minutes"
            elif avg_diff < pd.Timedelta(days=1):
                return f"{avg_diff.total_seconds() / 3600:.0f} hours"
            else:
                return f"{avg_diff.days} days"
        
        # Return data information
        data_info = {
            'rows': len(df),
            'columns': list(df.columns),
            'start_date': df['timestamps'].min().isoformat() if 'timestamps' in df.columns else 'N/A',
            'end_date': df['timestamps'].max().isoformat() if 'timestamps' in df.columns else 'N/A',
            'price_range': {
                'min': float(df[['open', 'high', 'low', 'close']].min().min()),
                'max': float(df[['open', 'high', 'low', 'close']].max().max())
            },
            'prediction_columns': ['open', 'high', 'low', 'close'] + (['volume'] if 'volume' in df.columns else []),
            'timeframe': detect_timeframe(df)
        }
        
        return jsonify({
            'success': True,
            'data_info': data_info,
            'message': f'Successfully loaded data, total {len(df)} rows'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to load data: {str(e)}'}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Perform prediction"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        lookback = int(data.get('lookback', 200))
        pred_len = int(data.get('pred_len', 60))
        prediction_mode = data.get('prediction_mode', 'future')
        
        # Get prediction quality parameters
        temperature = float(data.get('temperature', 1.0))
        top_p = float(data.get('top_p', 0.9))
        sample_count = int(data.get('sample_count', 1))
        
        if not file_path:
            return jsonify({'error': '文件路径不能为空'}), 400
        
        # Load data
        df, error = load_data_file(file_path)
        if error:
            return jsonify({'error': error}), 400
        
        if len(df) < lookback:
            return jsonify({'error': f'数据不足，至少需要 {lookback} 条数据'}), 400
        
        # Perform prediction
        if MODEL_AVAILABLE and predictor is not None:
            try:
                # Use real Kronos model
                required_cols = ['open', 'high', 'low', 'close']
                if 'volume' in df.columns:
                    required_cols.append('volume')
                
                start_date = data.get('start_date')
                
                if prediction_mode == 'future' or not start_date or not start_date.strip():
                    # Predict future - use latest data
                    x_df = df.iloc[-lookback:][required_cols]
                    x_timestamp = df.iloc[-lookback:]['timestamps']
                    
                    last_timestamp = df['timestamps'].iloc[-1]
                    time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0] if len(df) > 1 else pd.Timedelta(days=1)
                    y_timestamp = pd.date_range(
                        start=last_timestamp + time_diff,
                        periods=pred_len,
                        freq=time_diff
                    )
                    y_timestamp = pd.Series(y_timestamp, name='timestamps')
                    
                    prediction_type = f"预测未来{pred_len}天（基于最近{lookback}天数据）"
                    historical_start_idx = len(df) - lookback
                    
                else:
                    # Historical comparison - use selected time window
                    start_dt = pd.to_datetime(start_date)
                    mask = df['timestamps'] >= start_dt
                    time_range_df = df[mask]
                    
                    if len(time_range_df) < lookback + pred_len:
                        return jsonify({'error': f'从选定时间 {start_dt.strftime("%Y-%m-%d")} 开始数据不足，需要 {lookback + pred_len} 条，仅有 {len(time_range_df)} 条'}), 400
                    
                    x_df = time_range_df.iloc[:lookback][required_cols]
                    x_timestamp = time_range_df.iloc[:lookback]['timestamps']
                    y_timestamp = time_range_df.iloc[lookback:lookback+pred_len]['timestamps']
                    
                    prediction_type = f"历史对比验证（{lookback}条历史 + {pred_len}条对比）"
                    historical_start_idx = df[mask].index[0] if len(df[mask]) > 0 else 0
                
                # Ensure timestamps are Series format, not DatetimeIndex
                if isinstance(x_timestamp, pd.DatetimeIndex):
                    x_timestamp = pd.Series(x_timestamp, name='timestamps')
                if isinstance(y_timestamp, pd.DatetimeIndex):
                    y_timestamp = pd.Series(y_timestamp, name='timestamps')
                
                pred_df = predictor.predict(
                    df=x_df,
                    x_timestamp=x_timestamp,
                    y_timestamp=y_timestamp,
                    pred_len=pred_len,
                    T=temperature,
                    top_p=top_p,
                    sample_count=sample_count
                )
                
            except Exception as e:
                return jsonify({'error': f'模型预测失败: {str(e)}'}), 500
        else:
            return jsonify({'error': '模型未加载，请先加载模型'}), 400
        
        # Prepare actual data for comparison (only for historical comparison mode)
        actual_data = []
        actual_df = None
        
        if prediction_mode == 'historical' and start_date and start_date.strip():
            start_dt = pd.to_datetime(start_date)
            mask = df['timestamps'] >= start_dt
            time_range_df = df[mask]
            
            if len(time_range_df) >= lookback + pred_len:
                actual_df = time_range_df.iloc[lookback:lookback+pred_len]
                for _, row in actual_df.iterrows():
                    actual_data.append({
                        'timestamp': row['timestamps'].isoformat(),
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row['volume']) if 'volume' in row else 0,
                        'amount': float(row['amount']) if 'amount' in row else 0
                    })
        
        # Create chart
        chart_json = create_prediction_chart(df, pred_df, lookback, pred_len, actual_df, historical_start_idx)
        
        # Prepare prediction results with timestamps
        prediction_results = []
        
        # Use y_timestamp as prediction timestamps
        if isinstance(y_timestamp, pd.Series):
            pred_timestamps = y_timestamp.tolist()
        else:
            pred_timestamps = list(y_timestamp)
        
        for i, (_, row) in enumerate(pred_df.iterrows()):
            ts = pred_timestamps[i] if i < len(pred_timestamps) else f"T{i}"
            prediction_results.append({
                'timestamp': ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']) if 'volume' in row else 0,
                'amount': float(row['amount']) if 'amount' in row else 0
            })
        
        file_name = os.path.basename(file_path)
        timestamps_data = [ts.isoformat() if hasattr(ts, 'isoformat') else str(ts) for ts in y_timestamp]
        input_start_time = x_timestamp.iloc[0].isoformat() if len(x_timestamp) > 0 else ''
        input_end_time = x_timestamp.iloc[-1].isoformat() if len(x_timestamp) > 0 else ''
        
        current_model_name = 'unknown'
        if predictor is not None:
            for key, config in AVAILABLE_MODELS.items():
                if config['tokenizer_id'] in str(type(tokenizer)):
                    current_model_name = config['name']
                    break
        
        prediction_id = save_prediction(
            file_path=file_path,
            file_name=file_name,
            model=current_model_name,
            lookback=lookback,
            pred_len=pred_len,
            temperature=temperature,
            top_p=top_p,
            sample_count=sample_count,
            prediction_mode=prediction_mode,
            prediction_type=prediction_type,
            predictions_data=prediction_results,
            timestamps_data=timestamps_data,
            actual_data=actual_data,
            input_start_time=input_start_time,
            input_end_time=input_end_time
        )
        
        return jsonify({
            'success': True,
            'prediction_id': prediction_id,
            'prediction_type': prediction_type,
            'chart': chart_json,
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'has_comparison': len(actual_data) > 0,
            'message': f'预测完成，生成 {pred_len} 个预测点' + (f'，包含 {len(actual_data)} 个实际数据用于对比' if len(actual_data) > 0 else '')
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/load-model', methods=['POST'])
def load_model():
    """Load Kronos model"""
    global tokenizer, model, predictor
    
    try:
        if not MODEL_AVAILABLE:
            return jsonify({'error': 'Kronos model library not available'}), 400
        
        data = request.get_json()
        model_key = data.get('model_key', 'kronos-small')
        device = data.get('device', 'cpu')
        
        if model_key not in AVAILABLE_MODELS:
            return jsonify({'error': f'Unsupported model: {model_key}'}), 400
        
        model_config = AVAILABLE_MODELS[model_key]
        
        # Load tokenizer and model
        tokenizer = KronosTokenizer.from_pretrained(model_config['tokenizer_id'])
        model = Kronos.from_pretrained(model_config['model_id'])
        
        # Create predictor
        predictor = KronosPredictor(model, tokenizer, device=device, max_context=model_config['context_length'])
        
        return jsonify({
            'success': True,
            'message': f'Model loaded successfully: {model_config["name"]} ({model_config["params"]}) on {device}',
            'model_info': {
                'name': model_config['name'],
                'params': model_config['params'],
                'context_length': model_config['context_length'],
                'description': model_config['description']
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Model loading failed: {str(e)}'}), 500

@app.route('/api/available-models')
def get_available_models():
    """Get available model list"""
    return jsonify({
        'models': AVAILABLE_MODELS,
        'model_available': MODEL_AVAILABLE
    })

@app.route('/api/model-status')
def get_model_status():
    """Get model status"""
    if MODEL_AVAILABLE:
        if predictor is not None:
            return jsonify({
                'available': True,
                'loaded': True,
                'message': 'Kronos model loaded and available',
                'current_model': {
                    'name': predictor.model.__class__.__name__,
                    'device': str(next(predictor.model.parameters()).device)
                }
            })
        else:
            return jsonify({
                'available': True,
                'loaded': False,
                'message': 'Kronos model available but not loaded'
            })
    else:
        return jsonify({
            'available': False,
            'loaded': False,
            'message': 'Kronos model library not available, please install related dependencies'
        })

@app.route('/api/predictions/history')
def get_prediction_history():
    """获取预测历史记录"""
    try:
        limit = int(request.args.get('limit', 50))
        history = get_predictions_history(limit)
        return jsonify({'success': True, 'history': history})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/<int:prediction_id>')
def get_prediction_detail(prediction_id):
    """获取单条预测详情"""
    try:
        prediction = get_prediction_by_id(prediction_id)
        if prediction is None:
            return jsonify({'error': '预测记录不存在'}), 404
        
        # 为历史记录重新生成图表
        chart_json = create_chart_from_stored_data(prediction)
        prediction['chart'] = chart_json
        
        return jsonify({'success': True, 'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def create_chart_from_stored_data(prediction):
    """从存储的预测数据生成图表"""
    fig = go.Figure()
    
    predictions = prediction['predictions_data']
    timestamps = prediction['timestamps_data']
    actual_data = prediction['actual_data']
    
    # 添加预测数据曲线
    if predictions and len(predictions) > 0:
        closes = [p['close'] for p in predictions]
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=closes,
            mode='lines+markers',
            name='预测收盘价',
            line=dict(color='#2196F3', width=2),
            marker=dict(size=4)
        ))
        
        # 添加开盘价、最高价、最低价
        opens = [p['open'] for p in predictions]
        highs = [p['high'] for p in predictions]
        lows = [p['low'] for p in predictions]
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=highs,
            mode='lines',
            name='预测最高价',
            line=dict(color='#4CAF50', width=1, dash='dot'),
            showlegend=True
        ))
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=lows,
            mode='lines',
            name='预测最低价',
            line=dict(color='#EF5350', width=1, dash='dot'),
            showlegend=True
        ))
    
    # 添加实际数据曲线（如果有）
    if actual_data and len(actual_data) > 0:
        actual_timestamps = [a['timestamp'] for a in actual_data]
        actual_closes = [a['close'] for a in actual_data]
        
        fig.add_trace(go.Scatter(
            x=actual_timestamps,
            y=actual_closes,
            mode='lines+markers',
            name='实际收盘价',
            line=dict(color='#FF9800', width=2),
            marker=dict(size=4)
        ))
    
    # 设置图表布局
    title = f"预测记录 #{prediction['id']} - {prediction['prediction_type']}"
    fig.update_layout(
        title=title,
        xaxis_title='时间',
        yaxis_title='价格',
        template='plotly_white',
        height=500,
        showlegend=True,
        hovermode='x unified'
    )
    
    # 计算Y轴范围
    all_prices = []
    if predictions:
        all_prices.extend([p['open'] for p in predictions])
        all_prices.extend([p['high'] for p in predictions])
        all_prices.extend([p['low'] for p in predictions])
        all_prices.extend([p['close'] for p in predictions])
    if actual_data:
        all_prices.extend([a['open'] for a in actual_data])
        all_prices.extend([a['high'] for a in actual_data])
        all_prices.extend([a['low'] for a in actual_data])
        all_prices.extend([a['close'] for a in actual_data])
    
    if all_prices:
        min_price = min(all_prices)
        max_price = max(all_prices)
        padding = (max_price - min_price) * 0.1
        fig.update_yaxes(range=[min_price - padding, max_price + padding])
    
    return fig.to_json()

@app.route('/api/predictions/<int:prediction_id>', methods=['DELETE'])
def delete_prediction_record(prediction_id):
    """删除预测记录"""
    try:
        delete_prediction(prediction_id)
        return jsonify({'success': True, 'message': '预测记录已删除'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/compare', methods=['POST'])
def compare_predictions():
    """对比多次预测结果"""
    try:
        data = request.get_json()
        prediction_ids = data.get('prediction_ids', [])
        
        if len(prediction_ids) < 2:
            return jsonify({'error': '至少需要2条预测记录进行对比'}), 400
        
        predictions = get_predictions_by_ids(prediction_ids)
        
        if len(predictions) < 2:
            return jsonify({'error': '未找到足够的预测记录'}), 404
        
        fig = go.Figure()
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0', '#00BCD4']
        
        for i, pred in enumerate(predictions):
            closes = [p['close'] for p in pred['predictions_data']]
            timestamps = pred['timestamps_data']
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=closes,
                mode='lines+markers',
                name=f"预测#{pred['id']} ({pred['model']}, T={pred.get('temperature', 1.0)})",
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=4)
            ))
        
        fig.update_layout(
            title='预测结果对比',
            xaxis_title='时间',
            yaxis_title='收盘价',
            template='plotly_white',
            height=500,
            showlegend=True,
            hovermode='x unified'
        )
        
        comparison_data = []
        timestamps_common = predictions[0]['timestamps_data']
        for i, ts in enumerate(timestamps_common):
            row = {'timestamp': ts}
            for pred in predictions:
                if i < len(pred['predictions_data']):
                    row[f'预测#{pred["id"]}'] = pred['predictions_data'][i]['close']
            comparison_data.append(row)
        
        return jsonify({
            'success': True,
            'chart': fig.to_json(),
            'predictions': predictions,
            'comparison_data': comparison_data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/llm/status')
def get_llm_status():
    """获取LLM服务状态"""
    return jsonify(get_llm_config())

@app.route('/api/llm/analyze', methods=['POST'])
def llm_analyze():
    """LLM金融分析"""
    try:
        data = request.get_json()
        prediction_id = data.get('prediction_id')
        analysis_type = data.get('analysis_type', 'trend')
        
        if prediction_id is None:
            return jsonify({'error': '缺少prediction_id'}), 400
        
        prediction = get_prediction_by_id(prediction_id)
        if prediction is None:
            return jsonify({'error': '预测记录不存在'}), 404
        
        if analysis_type == 'trend':
            content = analyze_trend(prediction['predictions_data'], prediction['actual_data'])
        elif analysis_type == 'parameter':
            prediction_ids = data.get('prediction_ids', [prediction_id])
            predictions = get_predictions_by_ids(prediction_ids)
            content = analyze_parameter_impact(predictions)
        else:
            return jsonify({'error': '不支持的分析类型'}), 400
        
        save_llm_analysis(prediction_id, analysis_type, content)
        
        return jsonify({
            'success': True,
            'analysis_type': analysis_type,
            'content': content
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/llm/chat', methods=['POST'])
def llm_chat():
    """LLM金融对话"""
    try:
        data = request.get_json()
        prediction_id = data.get('prediction_id')
        message = data.get('message')
        
        if prediction_id is None or message is None:
            return jsonify({'error': '缺少prediction_id或message'}), 400
        
        prediction = get_prediction_by_id(prediction_id)
        if prediction is None:
            return jsonify({'error': '预测记录不存在'}), 404
        
        chat_history = get_chat_history(prediction_id)
        save_chat_message(prediction_id, 'user', message)
        
        response = chat_with_context(prediction, chat_history, message)
        save_chat_message(prediction_id, 'assistant', response)
        
        return jsonify({
            'success': True,
            'response': response
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/llm/history/<int:prediction_id>')
def get_llm_chat_history(prediction_id):
    """获取LLM对话历史"""
    try:
        history = get_chat_history(prediction_id)
        analyses = get_llm_analysis(prediction_id)
        return jsonify({
            'success': True,
            'chat_history': history,
            'analyses': analyses
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Kronos Web UI...")
    print(f"Model availability: {MODEL_AVAILABLE}")
    if MODEL_AVAILABLE:
        print("Tip: You can load Kronos model through /api/load-model endpoint")
    else:
        print("Tip: Will use simulated data for demonstration")
    
    app.run(debug=True, host='0.0.0.0', port=3005)
