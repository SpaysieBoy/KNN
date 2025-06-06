import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.base import BaseEstimator, TransformerMixin

# === BERT-embeddings transformer
class BertEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='emilyalsentzer/Bio_ClinicalBERT', max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.model.eval()
        embeddings = []
        with torch.no_grad():
            for text in X:
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length
                )
                outputs = self.model(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
                embeddings.append(cls_embedding.numpy())
        return np.vstack(embeddings)

# === Externe testbestanden
external_test_paths = {
    'feedback by hands': 'Text_File/Analyses/Subsets/Feedback_by_hands_sentiment.csv',
    'Second hospital': 'Text_File/Analyses/Subsets/Subset_Second_Hospital_sentiment.csv',
    'Amazon': 'Text_File/Analyses/Subsets/Subset_Amazon_reviews_sentiment.csv'
}

# === Hoofdloop over 10 runs
for i in tqdm(range(10), desc="Runs voltooid"):
    input_train_path = f"Text_File/Analyses/Subsets/generated_reviews_papagAIo1.0.csv"
    output_path = f"Dannyfolder/Output/Full_PapagAIo_RF+pipeline{i+1}.txt"

    df = pd.read_csv(input_train_path, sep=";")
    df.columns = ['Sentiment', 'Review']
    df['Review'] = df['Review'].astype(str).str.strip().str.replace('"""', '')
    df['Sentiment'] = df['Sentiment'].str.strip()
    df = df[df['Review'].str.len() > 10]
    if df['Sentiment'].value_counts().get('Neutraal', 0) < 10:
        df = df[df['Sentiment'] != 'Neutraal']

    # === Train/testsplit: 70% train, 30% test
    X = df['Review']
    y = df['Sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42+i)

    classes = np.unique(y_train)

    # === Pipeline met BERT + Random Forest
    pipeline = Pipeline([
        ('bert', BertEmbeddingTransformer(model_name='emilyalsentzer/Bio_ClinicalBERT')),
        ('clf', RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42+i,
            n_jobs=-1
        ))
    ])
    pipeline.fit(X_train, y_train)

    # === Evaluatie op PapagAIo testdata
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    conf_matrix_text = f"\n\nConfusion Matrix (PapagAIo):\n{pd.DataFrame(cm, index=classes, columns=classes).to_string()}"

    auc_text = ""
    if len(classes) == 2:
        y_scores = pipeline.predict_proba(X_test)[:, 1]
        y_test_bin = LabelBinarizer().fit_transform(y_test).ravel()
        auc_score = roc_auc_score(y_test_bin, y_scores)
        auc_text = f"\n\nAUC-ROC score (PapagAIo): {auc_score:.4f}"

    results = ["=== EVALUATIE OP PAPAGAIO TESTDATA (Random Forest) ===\n", report, conf_matrix_text, auc_text, "\n"]

    # === Evaluatie op 100% externe testdata met getrainde pipeline
    for label, path in external_test_paths.items():
        df_ext = pd.read_csv(path)
        df_ext.columns = ['Sentiment', 'Review']
        df_ext['Review'] = df_ext['Review'].astype(str).str.strip().str.replace('"""', '')
        df_ext['Sentiment'] = df_ext['Sentiment'].str.strip()
        df_ext = df_ext[df_ext['Review'].str.len() > 10]
        if df_ext['Sentiment'].value_counts().get('Neutraal', 0) < 10:
            df_ext = df_ext[df_ext['Sentiment'] != 'Neutraal']

        X_ext = df_ext['Review']
        y_ext = df_ext['Sentiment']
        y_ext_pred = pipeline.predict(X_ext)
        ext_report = classification_report(y_ext, y_ext_pred, zero_division=0)

        valid_labels = sorted(list(set(y_ext.unique()) & set(classes)))
        if not valid_labels:
            ext_cm = f"\n\nConfusion Matrix {label}: ⚠️ Geen overlappende klassen met PapagAIo."
        else:
            cm = confusion_matrix(y_ext, y_ext_pred, labels=valid_labels)
            ext_cm = f"\n\nConfusion Matrix {label}:\n{pd.DataFrame(cm, index=valid_labels, columns=valid_labels).to_string()}"

        auc_ext = ""
        if len(classes) == 2:
            y_ext_scores = pipeline.predict_proba(X_ext)[:, 1]
            y_ext_bin = LabelBinarizer().fit_transform(y_ext).ravel()
            if len(y_ext_scores) == len(y_ext_bin):
                auc_value = roc_auc_score(y_ext_bin, y_ext_scores)
                auc_ext = f"\n\nAUC-ROC score {label}: {auc_value:.4f}"

        results.extend([f"\n--- Evaluatie op {label} (100% test) ---\n", ext_report, ext_cm, auc_ext])

    # === Opslaan van resultaten
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in results:
            f.write(line)

    print(f"✅ Evaluatie Run {i+1} (Random Forest) opgeslagen in: {output_path}")
