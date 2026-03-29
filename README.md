# 🐦 Twitter Sentiment Analysis using BERT

A deep learning project that fine-tunes a BERT model to classify tweets as **POSITIVE** or **NEGATIVE** with high confidence.

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | **79%** |
| Macro Avg F1 | 0.79 |
| Weighted Avg F1 | 0.79 |
| Test Samples | 4,973 |

---

## 🔍 Sample Predictions

| Tweet | Prediction | Confidence |
|-------|-----------|------------|
| I absolutely love this, it's amazing! | ✅ POSITIVE | 98.73% |
| Worst experience ever, totally disappointed. | ❌ NEGATIVE | 99.01% |
| So excited for the weekend ahead! | ✅ POSITIVE | 98.13% |
| Can't believe how bad the service was. | ❌ NEGATIVE | 91.99% |
| The weather today is just okay. | ✅ POSITIVE | 82.49% |

---

## 🧠 Model Architecture

- **Base Model:** BERT (`bert-base-uncased`)
- **Task:** Binary Sentiment Classification
- **Framework:** PyTorch + HuggingFace Transformers
- **Dataset:** Sentiment140 (1.6M tweets)

---

## 🗂️ Project Structure

```
twitter-sentiment-analysis/
│
├── twitter_sentiment_analysis.py       # Training & evaluation script
├── twitter_sentiment_analysis.ipynb    # Jupyter notebook version
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/deeksha26052003/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis
```

### 2. Install dependencies
```bash
pip install transformers torch pandas scikit-learn numpy
```

### 3. Download the dataset
Download [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) and place `training.1600000.processed.noemoticon.csv` in the project root.

### 4. Run the training script
```bash
python twitter_sentiment_analysis.py
```

---

## 📦 Dependencies

- `transformers`
- `torch`
- `pandas`
- `scikit-learn`
- `numpy`

---

## 👩‍💻 Author

**Deeksha Bankapur**  
[GitHub](https://github.com/deeksha26052003)
