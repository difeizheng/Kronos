import os
import json
import openai
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

LLM_PROVIDER = os.environ.get('LLM_PROVIDER', 'siliconflow')

if LLM_PROVIDER == 'siliconflow':
    API_KEY = os.environ.get('SILICONFLOW_API_KEY', '')
    BASE_URL = os.environ.get('SILICONFLOW_BASE_URL', 'https://api.siliconflow.cn/v1')
    MODEL = os.environ.get('SILICONFLOW_MODEL', 'Qwen/Qwen2.5-72B-Instruct')
else:
    API_KEY = os.environ.get('OPENAI_API_KEY', '')
    BASE_URL = os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1')
    MODEL = os.environ.get('OPENAI_MODEL', 'gpt-4o')

def get_client():
    from openai import OpenAI
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)

def analyze_trend(predictions: List[Dict], actual_data: Optional[List[Dict]] = None) -> str:
    if not API_KEY:
        return f"LLM服务未配置，请在.env文件中设置 {LLM_PROVIDER.upper()}_API_KEY"
    
    close_prices = [p['close'] for p in predictions]
    timestamps = [p['timestamp'].split('T')[0] for p in predictions]
    
    analysis_context = {
        '预测时间范围': f"{timestamps[0]} 至 {timestamps[-1]}",
        '预测收盘价序列': close_prices,
        '起始价格': close_prices[0],
        '结束价格': close_prices[-1],
        '最高价格': max(close_prices),
        '最低价格': min(close_prices),
        '平均价格': sum(close_prices) / len(close_prices),
        '价格变化幅度': ((close_prices[-1] - close_prices[0]) / close_prices[0] * 100)
    }
    
    if actual_data and len(actual_data) > 0:
        actual_closes = [a['close'] for a in actual_data]
        analysis_context['实际收盘价序列'] = actual_closes
        analysis_context['预测误差MAPE'] = sum(abs(close_prices[i] - actual_closes[i]) / actual_closes[i] for i in range(min(len(close_prices), len(actual_closes)))) / min(len(close_prices), len(actual_closes)) * 100
    
    prompt = f"""你是一位专业的金融分析师，请根据以下Kronos模型预测数据进行趋势分析。

预测数据摘要：
{json.dumps(analysis_context, indent=2, ensure_ascii=False)}

请从以下维度进行分析：

1. **趋势判断**
   - 整体趋势方向（上涨/下跌/震荡）
   - 趋势强度评估
   - 关键转折点识别

2. **风险评估**
   - 价格波动率分析
   - 支撑位和阻力位判断
   - 置信区间建议

3. **投资建议**
   - 基于预测结果的操作建议
   - 风险提示
   - 关注事项

请用中文回答，保持专业、客观、谨慎。"""

    try:
        client = get_client()
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"LLM分析失败: {str(e)}"

def analyze_parameter_impact(predictions_list: List[Dict]) -> str:
    if not API_KEY:
        return f"LLM服务未配置，请在.env文件中设置 {LLM_PROVIDER.upper()}_API_KEY"
    
    comparison_data = []
    for pred in predictions_list:
        close_prices = [p['close'] for p in pred['predictions_data']]
        comparison_data.append({
            '预测ID': pred['id'],
            '模型': pred['model'],
            '温度': pred['temperature'],
            'top_p': pred['top_p'],
            '采样次数': pred['sample_count'],
            '预测收盘价': close_prices,
            '起始价格': close_prices[0],
            '结束价格': close_prices[-1],
            '变化幅度': ((close_prices[-1] - close_prices[0]) / close_prices[0] * 100)
        })
    
    prompt = f"""你是一位专业的量化分析师，请分析不同预测参数对Kronos模型预测结果的影响。

对比数据：
{json.dumps(comparison_data, indent=2, ensure_ascii=False)}

请从以下角度分析：

1. **参数敏感性分析**
   - Temperature变化对预测的影响
   - Top_p参数的影响
   - 采样次数的影响

2. **预测差异分析**
   - 各预测结果的分歧点
   - 共识区域（各预测一致的部分）
   - 分歧区域（各预测不一致的部分）

3. **综合判断**
   - 最可信的预测配置建议
   - 参数调优建议

请用中文回答，保持专业、客观。"""

    try:
        client = get_client()
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"LLM分析失败: {str(e)}"

def chat_with_context(prediction_data: Dict, chat_history: List[Dict], user_message: str) -> str:
    if not API_KEY:
        return f"LLM服务未配置，请在.env文件中设置 {LLM_PROVIDER.upper()}_API_KEY"
    
    close_prices = [p['close'] for p in prediction_data['predictions_data']]
    timestamps = [p['timestamp'].split('T')[0] for p in prediction_data['predictions_data']]
    
    system_prompt = f"""你是一位专业的金融分析师AI助手，正在帮助用户分析Kronos模型的预测结果。

当前预测数据摘要：
- 数据文件: {prediction_data['file_name']}
- 模型: {prediction_data['model']}
- 预测模式: {prediction_data['prediction_mode']}
- 预测时间范围: {timestamps[0]} 至 {timestamps[-1]}
- 预测收盘价: {close_prices[:10]}...（共{len(close_prices)}个数据点）
- 参数: lookback={prediction_data['lookback']}, pred_len={prediction_data['pred_len']}, T={prediction_data['temperature']}, top_p={prediction_data['top_p']}

请根据用户的问题，结合预测数据给出专业、客观、谨慎的回答。回答时：
1. 以专业金融分析师的视角分析
2. 注意风险提示
3. 不做确定性承诺，使用"可能"、"预计"等表述"""

    messages = [{"role": "system", "content": system_prompt}]
    for chat in chat_history:
        messages.append({"role": chat['role'], "content": chat['message']})
    messages.append({"role": "user", "content": user_message})
    
    try:
        client = get_client()
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"LLM对话失败: {str(e)}"

def check_llm_available() -> bool:
    return bool(API_KEY)

def get_llm_config() -> Dict:
    return {
        'available': check_llm_available(),
        'provider': LLM_PROVIDER,
        'model': MODEL,
        'base_url': BASE_URL
    }