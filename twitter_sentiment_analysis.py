# ============================================================
# Twitter Sentiment Analysis
# Stack: PyTorch, BERT, HuggingFace, NLTK, pandas, sklearn, Plotly
# ============================================================

# ── 1. Install dependencies ──────────────────────────────────
import ssl
import nltk
import torch
import os
import zipfile
import urllib.request
import re
from transformers import pipeline as hf_pipeline
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from transformers import (
    BertTokenizerFast, BertForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import subprocess
import sys
for pkg in ['transformers', 'torch', 'nltk', 'pandas', 'scikit-learn',
            'plotly', 'accelerate', 'tqdm']:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

# ── 2. Imports ───────────────────────────────────────────────


# Fix SSL for Mac
try:
    _create_unverified = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified

# NLTK downloads
for r in ['stopwords', 'punkt', 'wordnet', 'omw-1.4', 'punkt_tab']:
    nltk.download(r, quiet=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── 3. Download & Load Sentiment140 CSV ──────────────────────
CSV_PATH = 'training.1600000.processed.noemoticon.csv'
ZIP_PATH = 'sentiment140.zip'
DATA_URL = 'https://huggingface.co/datasets/stanfordnlp/sentiment140/resolve/main/data/training.1600000.processed.noemoticon.csv.zip'

if not os.path.exists(CSV_PATH):
    print('Downloading Sentiment140 (~80MB)...')
    urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        z.extractall('.')
    os.remove(ZIP_PATH)
    print('Download complete.')

print('Loading dataset...')
df_full = pd.read_csv(CSV_PATH, encoding='latin-1', header=None)
df_full.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']
df_full['label'] = df_full['sentiment'].map({0: 0, 4: 1})
df_full = df_full[['text', 'label']].dropna()

# Sample 25K per class = 50K total
df = pd.concat([
    df_full[df_full['label'] == 0].sample(25000, random_state=SEED),
    df_full[df_full['label'] == 1].sample(25000, random_state=SEED)
]).sample(frac=1, random_state=SEED).reset_index(drop=True)

print(f'Dataset: {df.shape}')
print(df['label'].value_counts())

# ── 4. EDA plots ─────────────────────────────────────────────
label_counts = df['label'].value_counts().reset_index()
label_counts.columns = ['label', 'count']
label_counts['sentiment'] = label_counts['label'].map(
    {0: 'Negative', 1: 'Positive'})

fig = px.bar(label_counts, x='sentiment', y='count',
             color='sentiment',
             color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c'},
             title='Label Distribution in 50K Sample', text='count')
fig.update_traces(textposition='outside')
fig.show()

df['tweet_length'] = df['text'].apply(len)
fig2 = px.histogram(df, x='tweet_length',
                    color=df['label'].map({0: 'Negative', 1: 'Positive'}),
                    barmode='overlay', nbins=60, opacity=0.7,
                    title='Tweet Length Distribution by Sentiment',
                    color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c'})
fig2.show()

# ── 5. Preprocessing ─────────────────────────────────────────
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_tweet(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)


tqdm.pandas(desc='Cleaning tweets')
df['clean_text'] = df['text'].progress_apply(clean_tweet)
df = df[df['clean_text'].str.strip().astype(bool)].reset_index(drop=True)
print(f'After cleaning: {df.shape}')

# ── 6. Train / Val / Test split ──────────────────────────────
X, y = df['clean_text'].tolist(), df['label'].tolist()
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2,
                                                    random_state=SEED, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5,
                                                random_state=SEED, stratify=y_temp)
print(f'Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}')

# ── 7. Tokenization ──────────────────────────────────────────
MODEL_NAME = 'bert-base-uncased'
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)


class TweetDataset(Dataset):
    def __init__(self, texts, labels, tok, max_len=128):
        self.enc = tok(texts, truncation=True, padding='max_length',
                       max_length=max_len, return_tensors='pt')
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids':      self.enc['input_ids'][idx],
            'attention_mask': self.enc['attention_mask'][idx],
            'token_type_ids': self.enc['token_type_ids'][idx],
            'labels':         self.labels[idx]
        }


print('Tokenizing...')
train_ds = TweetDataset(X_train, y_train, tokenizer)
val_ds = TweetDataset(X_val,   y_val,   tokenizer)
test_ds = TweetDataset(X_test,  y_test,  tokenizer)
print('Done.')

# ── 8. Fine-tune BERT ────────────────────────────────────────
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2,
    id2label={0: 'NEGATIVE', 1: 'POSITIVE'},
    label2id={'NEGATIVE': 0, 'POSITIVE': 1}
)
model.to(device)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    report = classification_report(
        labels, preds, output_dict=True, zero_division=0)
    return {
        'accuracy':  accuracy_score(labels, preds),
        'precision': report['weighted avg']['precision'],
        'recall':    report['weighted avg']['recall'],
        'f1':        report['weighted avg']['f1-score']
    }


args = TrainingArguments(
    output_dir='./bert_sentiment',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=200,
    weight_decay=0.01,
    learning_rate=2e-5,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    logging_dir='./logs',
    logging_steps=100,
    seed=SEED,
    fp16=False,
    report_to='none'
)

trainer = Trainer(
    model=model, args=args,
    train_dataset=train_ds, eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print('Fine-tuning BERT... (this will take a while on CPU)')
trainer.train()

# ── 9. Evaluate on test set ──────────────────────────────────
print('Evaluating...')
out = trainer.predict(test_ds)
y_pred = np.argmax(out.predictions, axis=-1)

acc = accuracy_score(y_test, y_pred)
print(f'\nTest Accuracy: {acc:.4f} ({acc*100:.2f}%)')
print(classification_report(y_test, y_pred,
      target_names=['Negative', 'Positive']))

# Training curves
logs = trainer.state.log_history
train_logs = [l for l in logs if 'loss' in l and 'eval_loss' not in l]
eval_logs = [l for l in logs if 'eval_accuracy' in l]

fig3 = make_subplots(rows=1, cols=2,
                     subplot_titles=('Training Loss', 'Validation Accuracy'))
fig3.add_trace(go.Scatter(x=[l['step'] for l in train_logs],
                          y=[l['loss'] for l in train_logs],
                          mode='lines', name='Train Loss',
                          line=dict(color='#e74c3c')), row=1, col=1)
fig3.add_trace(go.Scatter(x=[l['epoch'] for l in eval_logs],
                          y=[l['eval_accuracy'] for l in eval_logs],
                          mode='lines+markers', name='Val Accuracy',
                          line=dict(color='#2ecc71')), row=1, col=2)
fig3.update_layout(title='BERT Fine-Tuning Metrics', height=400)
fig3.show()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig4 = px.imshow(cm, labels=dict(x='Predicted', y='Actual', color='Count'),
                 x=['Negative', 'Positive'], y=['Negative', 'Positive'],
                 color_continuous_scale='Blues',
                 title='Confusion Matrix', text_auto=True)
fig4.show()

# ── 10. Save model ───────────────────────────────────────────
SAVE_DIR = './bert_sentiment_final'
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f'Model saved to {SAVE_DIR}')

# ── 11. Inference ────────────────────────────────────────────

pipe = hf_pipeline('text-classification', model=SAVE_DIR, tokenizer=SAVE_DIR,
                   device=0 if torch.cuda.is_available() else -1)

samples = [
    "I absolutely love this, it's amazing!",
    "Worst experience ever, totally disappointed.",
    "So excited for the weekend ahead!",
    "Can't believe how bad the service was.",
    "The weather today is just okay."
]
results = pipe(samples)
out_df = pd.DataFrame({
    'tweet':      samples,
    'prediction': [r['label'] for r in results],
    'confidence': [round(r['score'], 4) for r in results]
})
print(out_df.to_string(index=False))
