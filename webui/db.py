import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'predictions.db')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_path TEXT,
        file_name TEXT,
        model TEXT,
        lookback INTEGER,
        pred_len INTEGER,
        temperature REAL,
        top_p REAL,
        sample_count INTEGER,
        prediction_mode TEXT,
        prediction_type TEXT,
        predictions_data TEXT,
        timestamps_data TEXT,
        actual_data TEXT,
        input_start_time TEXT,
        input_end_time TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS llm_analysis (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prediction_id INTEGER,
        analysis_type TEXT,
        content TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (prediction_id) REFERENCES predictions(id)
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS llm_chat (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prediction_id INTEGER,
        role TEXT,
        message TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (prediction_id) REFERENCES predictions(id)
    )''')
    
    conn.commit()
    conn.close()

def save_prediction(file_path, file_name, model, lookback, pred_len, temperature, 
                    top_p, sample_count, prediction_mode, prediction_type,
                    predictions_data, timestamps_data, actual_data, 
                    input_start_time, input_end_time):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''INSERT INTO predictions 
        (file_path, file_name, model, lookback, pred_len, temperature, top_p, 
         sample_count, prediction_mode, prediction_type, predictions_data, 
         timestamps_data, actual_data, input_start_time, input_end_time, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (file_path, file_name, model, lookback, pred_len, temperature, top_p,
         sample_count, prediction_mode, prediction_type, 
         json.dumps(predictions_data), json.dumps(timestamps_data),
         json.dumps(actual_data) if actual_data else None,
         input_start_time, input_end_time, datetime.now().isoformat()))
    
    prediction_id = c.lastrowid
    conn.commit()
    conn.close()
    
    return prediction_id

def get_predictions_history(limit=50):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''SELECT id, file_name, model, lookback, pred_len, temperature, 
                        top_p, sample_count, prediction_mode, prediction_type,
                        input_start_time, input_end_time, created_at
                 FROM predictions 
                 ORDER BY created_at DESC 
                 LIMIT ?''', (limit,))
    
    rows = c.fetchall()
    conn.close()
    
    history = []
    for row in rows:
        history.append({
            'id': row[0],
            'file_name': row[1],
            'model': row[2],
            'lookback': row[3],
            'pred_len': row[4],
            'temperature': row[5],
            'top_p': row[6],
            'sample_count': row[7],
            'prediction_mode': row[8],
            'prediction_type': row[9],
            'input_start_time': row[10],
            'input_end_time': row[11],
            'created_at': row[12]
        })
    
    return history

def get_prediction_by_id(prediction_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''SELECT id, file_path, file_name, model, lookback, pred_len, 
                        temperature, top_p, sample_count, prediction_mode, 
                        prediction_type, predictions_data, timestamps_data, 
                        actual_data, input_start_time, input_end_time, created_at
                 FROM predictions 
                 WHERE id = ?''', (prediction_id,))
    
    row = c.fetchone()
    conn.close()
    
    if row is None:
        return None
    
    return {
        'id': row[0],
        'file_path': row[1],
        'file_name': row[2],
        'model': row[3],
        'lookback': row[4],
        'pred_len': row[5],
        'temperature': row[6],
        'top_p': row[7],
        'sample_count': row[8],
        'prediction_mode': row[9],
        'prediction_type': row[10],
        'predictions_data': json.loads(row[11]) if row[11] else [],
        'timestamps_data': json.loads(row[12]) if row[12] else [],
        'actual_data': json.loads(row[13]) if row[13] else [],
        'input_start_time': row[14],
        'input_end_time': row[15],
        'created_at': row[16]
    }

def get_predictions_by_ids(prediction_ids):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''SELECT id, file_name, model, prediction_type, predictions_data, 
                        timestamps_data, actual_data, input_start_time, input_end_time
                 FROM predictions 
                 WHERE id IN ({})'''.format(','.join(map(str, prediction_ids))))
    
    rows = c.fetchall()
    conn.close()
    
    predictions = []
    for row in rows:
        predictions.append({
            'id': row[0],
            'file_name': row[1],
            'model': row[2],
            'prediction_type': row[3],
            'predictions_data': json.loads(row[4]) if row[4] else [],
            'timestamps_data': json.loads(row[5]) if row[5] else [],
            'actual_data': json.loads(row[6]) if row[6] else [],
            'input_start_time': row[7],
            'input_end_time': row[8]
        })
    
    return predictions

def delete_prediction(prediction_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('DELETE FROM llm_analysis WHERE prediction_id = ?', (prediction_id,))
    c.execute('DELETE FROM llm_chat WHERE prediction_id = ?', (prediction_id,))
    c.execute('DELETE FROM predictions WHERE id = ?', (prediction_id,))
    
    conn.commit()
    conn.close()

def save_llm_analysis(prediction_id, analysis_type, content):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''INSERT INTO llm_analysis 
        (prediction_id, analysis_type, content, created_at)
        VALUES (?, ?, ?, ?)''',
        (prediction_id, analysis_type, content, datetime.now().isoformat()))
    
    analysis_id = c.lastrowid
    conn.commit()
    conn.close()
    
    return analysis_id

def get_llm_analysis(prediction_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''SELECT id, analysis_type, content, created_at
                 FROM llm_analysis 
                 WHERE prediction_id = ?
                 ORDER BY created_at DESC''', (prediction_id,))
    
    rows = c.fetchall()
    conn.close()
    
    analyses = []
    for row in rows:
        analyses.append({
            'id': row[0],
            'analysis_type': row[1],
            'content': row[2],
            'created_at': row[3]
        })
    
    return analyses

def save_chat_message(prediction_id, role, message):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''INSERT INTO llm_chat 
        (prediction_id, role, message, created_at)
        VALUES (?, ?, ?, ?)''',
        (prediction_id, role, message, datetime.now().isoformat()))
    
    conn.commit()
    conn.close()

def get_chat_history(prediction_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''SELECT role, message, created_at
                 FROM llm_chat 
                 WHERE prediction_id = ?
                 ORDER BY created_at ASC''', (prediction_id,))
    
    rows = c.fetchall()
    conn.close()
    
    history = []
    for row in rows:
        history.append({
            'role': row[0],
            'message': row[1],
            'created_at': row[2]
        })
    
    return history

init_db()