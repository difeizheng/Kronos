# AGENTS.md

Repo-specific guidance for OpenCode sessions.

## Testing

```bash
pytest tests/
```

Regression tests use pinned model revisions (see `tests/test_kronos_regression.py:30-31`) and validate MSE tolerance. Tests run on CPU by default.

## Fine-tuning Order

Tokenizer must be trained before predictor. Two pipelines:

**Qlib (Chinese A-share):**
```bash
pip install pyqlib  # prerequisite
python finetune/qlib_data_preprocess.py  # step 1: prepare data
torchrun --standalone --nproc_per_node=N finetune/train_tokenizer.py  # step 2
torchrun --standalone --nproc_per_node=N finetune/train_predictor.py  # step 3
python finetune/qlib_test.py --device cuda:0  # step 4: backtest
```

Edit paths in `finetune/config.py` before running.

**Custom CSV:**
```bash
python finetune_csv/train_sequential.py --config configs/your.yaml
```

CSV requires columns: `timestamps, open, high, low, close, volume, amount` (volume/amount can be 0).

## Model Constraints

- **max_context**: Kronos-mini=2048, Kronos-small/base=512. Exceeding truncates silently.
- **Batch prediction**: All series must have identical historical length and `pred_len`.
- **DataFrame columns**: `open, high, low, close` required; `volume, amount` optional (filled with zeros).

## Token Architecture

Hierarchical two-part tokens: `s1` (pre) → `s2` (post, conditioned on s1 via DependencyAwareLayer). Don't treat them as independent.

## WebUI

```bash
cd webui && python run.py  # http://localhost:7070
```

**预测历史功能：**
- SQLite数据库 `webui/predictions.db` 自动存储预测记录
- 支持历史查看、多预测对比、LLM分析对话

**LLM配置（webui/.env）：**
```env
LLM_PROVIDER=siliconflow  # 或 openai
SILICONFLOW_API_KEY=your-key
SILICONFLOW_MODEL=Qwen/Qwen2.5-72B-Instruct
```

**常见问题：**
- Plotly图表不显示：Pandas Series需转`list`格式
- 未来预测报错：`y_timestamp`不能为None，需生成虚拟时间戳
- LLM分析结果折叠失效：CSS `display:none`需配合`.expanded`类

**文件结构：**
- `webui/app.py` — Flask后端，预测API、历史API、LLM API
- `webui/db.py` — SQLite操作封装
- `webui/llm_service.py` — LLM调用(SiliconFlow/OpenAI)
- `webui/.env` — LLM API配置(不提交git)

## Key Entry Points

- `model/__init__.py` — KronosTokenizer, Kronos, KronosPredictor exports
- `model/module.py` — BSQuantizer, HierarchicalEmbedding, DependencyAwareLayer
- `model/kronos.py` — Core model implementation