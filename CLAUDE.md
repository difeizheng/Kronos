# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kronos is the first open-source foundation model for financial candlesticks (K-lines), trained on data from over 45 global exchanges. It uses a two-stage architecture:
1. **KronosTokenizer**: A specialized tokenizer that quantizes continuous OHLCV data into hierarchical discrete tokens using Binary Spherical Quantization (BSQ)
2. **Kronos**: A decoder-only Transformer pre-trained on these tokens for autoregressive forecasting

The project was accepted to AAAI 2026. Paper: https://arxiv.org/abs/2508.02739

## Key Commands

### Installation
```bash
pip install -r requirements.txt
```

### Testing
```bash
pytest tests/
```

The regression tests use pinned HuggingFace model revisions and compare predictions against stored CSV fixtures. Tests run on CPU with argmax decoding (`top_k=1, top_p=1.0`) for determinism.

### Fine-tuning (Qlib-based, for Chinese A-share market)
Requires `pip install pyqlib` first.

```bash
# 1. Prepare data
python finetune/qlib_data_preprocess.py

# 2. Fine-tune tokenizer (multi-GPU)
torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_tokenizer.py

# 3. Fine-tune predictor (multi-GPU)
torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_predictor.py

# 4. Backtest
python finetune/qlib_test.py --device cuda:0
```

Configuration is centralized in `finetune/config.py` - update paths before running. This is a Python class with hardcoded values (edit the source file directly).

### Fine-tuning on Custom CSV Data
```bash
# Sequential training (recommended)
python finetune_csv/train_sequential.py --config configs/config_ali09988_candle-5min.yaml

# DDP training
DIST_BACKEND=nccl torchrun --standalone --nproc_per_node=8 train_sequential.py --config configs/your_config.yaml
```

CSV data requires columns: `timestamps`, `open`, `high`, `low`, `close`, `volume`, `amount`. Configuration uses YAML files (see `finetune_csv/config_loader.py`).

### WebUI
```bash
cd webui
python run.py  # or python app.py
# Access at http://localhost:7070
```

## Architecture

### Core Model API (`model/__init__.py`)
- `KronosTokenizer` - Tokenizes OHLCV data into discrete tokens
- `Kronos` - Main decoder-only Transformer model
- `KronosPredictor` - High-level prediction interface handling normalization, prediction, and denormalization

### Token Architecture
The tokenizer produces **hierarchical two-part tokens**:
- **s1 (pre tokens)**: Primary quantized representation (first `s1_bits`, typically 10 bits → vocab 1024)
- **s2 (post tokens)**: Fine-grained refinement tokens (remaining `s2_bits`, typically 10 bits → vocab 1024)

The Kronos model decodes these in sequence: first predicts s1, then s2 conditioned on sampled s1 via a `DependencyAwareLayer`. This hierarchical approach avoids a single 2^20 vocab (1M classes) in favor of two 2^10 vocabs with cross-attention conditioning.

### Key Modules (`model/module.py`)
- `BinarySphericalQuantizer`: BSQ quantizer that maps to bipolar vectors {+1, -1}^D on the unit hypersphere. Gives 2^D codebook entries with only D parameters (vs 2^D × D for standard VQ).
- `HierarchicalEmbedding`: Separate embeddings for s1 and s2 tokens, concatenated and projected via `Linear(d_model*2, d_model)`.
- `TransformerBlock`: Decoder block with RoPE attention and SwiGLU FFN (`w2(silu(w1(x)) * w3(x))`).
- `DependencyAwareLayer`: Cross-attention for s2 decoding conditioned on sampled s1. Query = sampled s1 embeddings, Key/Value = transformer context. Uses non-causal cross-attention at inference (all s1 tokens available).
- `TemporalEmbedding`: Time features (minute, hour, weekday, day, month) added to token embeddings.

### Prediction Flow
`KronosPredictor.predict()`:
1. Normalizes input OHLCV data (per-instance z-score using lookback-only stats to prevent leakage)
2. Tokenizes via `KronosTokenizer.encode(half=True)` → returns (s1_ids, s2_ids)
3. Autoregressive inference: `decode_s1()` → sample s1 → `decode_s2()` with DependencyAwareLayer → sample s2
4. Decodes tokens back to continuous values via `KronosTokenizer.decode(half=True)`
5. Denormalizes to original scale

The `half=True` parameter is always used in practice — it treats s1 and s2 as separate token streams. The non-half path (single 20-bit token) exists but is unused.

### Sample-Count Averaging
At inference, `sample_count` parallel samples are generated and averaged (`preds = np.mean(preds, axis=1)`). This ensemble approach leverages sampling stochasticity within a single forward pass.

### Model Zoo (HuggingFace Hub: `NeoQuasar`)
| Model | Params | Context | Tokenizer |
|-------|--------|---------|-----------|
| Kronos-mini | 4.1M | 2048 | Kronos-Tokenizer-2k |
| Kronos-small | 24.7M | 512 | Kronos-Tokenizer-base |
| Kronos-base | 102.3M | 512 | Kronos-Tokenizer-base |

### Batch Prediction
`KronosPredictor.predict_batch()` processes multiple series in parallel along the batch dimension. All series must have identical historical length and prediction length.

## Training Pipelines

### Pipeline A: `finetune/` (Qlib-based, Chinese A-shares)
- **Config**: Python class in `finetune/config.py` (edit source directly)
- **Dataset**: `QlibDataset` - multi-symbol sliding window with pre-computed valid indices
- **Data split**: Explicit date ranges in config
- **Tokenizer training**: `loss = (recon_pre + recon_all + bsq_loss) / 2`
- **Predictor training**: Cross-entropy on s1+s2 tokens, generated on-the-fly via tokenizer
- **Logging**: Comet.ml integration
- **Scheduler**: OneCycleLR with 3% warmup

### Pipeline B: `finetune_csv/` (CSV-based, any market)
- **Config**: YAML files via `config_loader.py` → `CustomFinetuneConfig`
- **Dataset**: `CustomKlineDataset` - single-symbol, time-based split
- **Data split**: Ratio-based (70/15/15 by default)
- **Orchestrator**: `SequentialTrainer` in `train_sequential.py` automates tokenizer → predictor training
- **Flexibility**: Supports `--skip-tokenizer`, `--skip-basemodel`, `--skip-existing` flags

## WebUI Architecture (`webui/`)

Flask application with SQLite persistence and optional LLM integration:
- **Models**: Loaded lazily via `/api/load-model` (not at startup)
- **Persistence**: Predictions saved to both SQLite (`predictions.db`) and JSON files in `prediction_results/`
- **Database**: Three tables - `predictions`, `llm_analysis`, `llm_chat`
- **LLM service**: Provider-agnostic (SiliconFlow or OpenAI API) configured via `.env`
- **Prediction modes**: "future" (extrapolate from latest) and "historical" (compare against actuals)
- **Charts**: Plotly candlestick charts via `create_prediction_chart()`

## Important Constraints

- **max_context**: Kronos-small/base have max_context=512; Kronos-mini has max_context=2048. Input lookback should not exceed this.
- `predict()` requires DataFrame with columns: `open`, `high`, `low`, `close`. `volume` and `amount` are optional (filled with zeros if missing).
- Batch prediction requires all series to have the same historical and prediction lengths.
- **Normalization**: Per-instance z-score using lookback-only statistics (no future leakage).
- **Causal attention**: Self-attention in transformer blocks is strictly causal, but `DependencyAwareLayer` cross-attention uses `is_causal=self.training` (non-causal at inference since all s1 tokens are already sampled).