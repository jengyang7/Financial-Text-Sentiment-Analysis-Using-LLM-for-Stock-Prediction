# FYP: Financial Text Sentiment Analysis Using LLM for Stock Market Prediction

**Author:** Kong Jeng Yang (22115126)  
**Institution:** Faculty of Computer Science & Information Technology, Universiti Malaya  
**Program:** Master of Data Science  
**Project Supervisor:** Dr. Lim Chee Kau  
**Year:** 2025

## 📌 Abstract
This research explores the integration of Large Language Models (LLMs) with financial time-series forecasting. The study proposes a novel hybrid approach that fine-tunes domain-specific LLMs (**Llama 3 8B**, **Gemma 7B**) on the *Financial PhraseBank* dataset to extract nuanced sentiment from historical financial news. These sentiment scores are then integrated with historical market data (OHLCV) to predict stock price movements using **LSTM**, **Random Forest**, and **XGBoost**.

## 🚀 Key Features
* **LLM Fine-Tuning**: Comparative analysis of Llama 3 (8B), Gemma (7B), and FinBERT for financial sentiment analysis.
    * *Result:* Llama 3 fine-tuned on the "100% Agreement" dataset achieved the highest accuracy (**99.3%**).
* **Advanced Feature Engineering**: Engineered rolling sentiment features (3-day, 7-day, 10-day averages) to capture long-term market psychology.
* **Hybrid Prediction Models**: Evaluated the impact of sentiment on Deep Learning (LSTM) vs. Ensemble methods (Random Forest, XGBoost).
* **Case Study**: Detailed performance analysis on **Tesla (TSLA)** stock over a 4-year period (2021-2025).

## 🛠️ Tech Stack
* **LLM Frameworks**: PyTorch, Hugging Face Transformers, PEFT (LoRA), BitsAndBytes (QLoRA).
* **Machine Learning**: TensorFlow/Keras (LSTM), Scikit-Learn, XGBoost.
* **Data Sources**: 
    * *News*: Polygon.io (160k+ articles filtered to 35k high-quality samples).
    * *Market Data*: yFinance (US Top 20 Stocks).
* **Experiment Tracking**: Optuna (Hyperparameter Tuning).

## 📊 Methodology

The project is executed in two primary phases:

### Phase 1: Fine-Tuning Financial LLMs
We fine-tuned general-purpose LLMs to specialize in financial sentiment (Positive, Neutral, Negative).
* **Dataset**: Financial PhraseBank (split by annotator agreement levels: 50%, 66%, 75%, 100%).
* **Findings**: Data quality matters more than quantity. Models trained on the smaller **100% Agreement** subset significantly outperformed those trained on the full dataset.
* **Best Model**: **Llama 3 8B** (Fine-tuned), achieving 0.99 F1-Score.

### Phase 2: Stock Trend Prediction
We merged the predicted sentiment scores with daily stock data to forecast closing prices.
* **Input Features**: Open, High, Low, Close, Volume + `sentiment_7d_avg` (7-day rolling average).
* **Models Tested**:
    1.  **LSTM (Long Short-Term Memory)**: Captures sequential dependencies.
    2.  **Random Forest & XGBoost**: Strong baselines for structured data.

## 📈 Key Findings & Results

### 1. LLM Performance
| Model | Accuracy (100% Agreement Data) | F1 Score |
| :--- | :--- | :--- |
| **Llama 3 8B (Fine-tuned)** | **0.993** | **0.99** |
| Gemma 7B (Fine-tuned) | 0.978 | 0.98 |
| FinBERT | 0.970 | 0.95 |

### 2. Stock Prediction (TSLA Case Study)
Does sentiment help predict stock prices?
* **LSTM**: **YES.** Integrating the **7-day sentiment average** reduced the RMSE by **10.9%** (from $27.71 to $24.68) and improved R² by 4.4%.
* **Random Forest / XGBoost**: **NO.** These models showed minimal or negative improvement when sentiment features were added, suggesting they treat sentiment as noise rather than a temporal signal.

> **Conclusion:** Deep learning models like LSTM are superior at leveraging sequential sentiment trends for price forecasting compared to tree-based models.

## 🔗 Project Resources & Model Checkpoints

The following repositories and model checkpoints are associated with this research project, comprising the trained models, implementation code, and relevant datasets:

* **Fine-tuned LLaMA 3 (Financial Sentiment)**:  
    [huggingface.co/jengyang/trained-llama3-sentences_allagree-financial-sentiment](https://huggingface.co/jengyang/trained-llama3-sentences_allagree-financial-sentiment)

* **Fine-tuned Gemma (Financial Sentiment)**:  
    [huggingface.co/jengyang/trained-gemma-sentences_allagree-financial-sentiment](https://huggingface.co/jengyang/trained-gemma-sentences_allagree-financial-sentiment)

* **LSTM Stock Prediction Model**:  
    [huggingface.co/jengyang/lstm-stock-prediction-model](https://huggingface.co/jengyang/lstm-stock-prediction-model)
    *(Includes models with and without sentiment features)*

## 📂 Repository Structure
```text
├── fine_tune_llm/
│   ├── script/
│   │   ├── fine-tune-llama-3.ipynb       # Llama 3 QLoRA implementation
│   │   ├── fine-tune-gemma-7b.ipynb      # Gemma 7B QLoRA implementation
│   │   └── Financial_Phrasebank_EDA.ipynb # Analysis of agreement levels
│   └── dataset/                          # Financial PhraseBank subsets
├── stock_prediction/
│   ├── lstm_single_latest_final.ipynb    # LSTM model (Base vs Optimized)
│   ├── xgboost_rf_final.ipynb            # Tree-based model comparison
│   ├── create_stock_price_with_sentiment # Merging Polygon.io news with yfinance
│   └── Llama3_Hyperparameter_Tuning.ipynb
├── requirements.txt
└── README.md

