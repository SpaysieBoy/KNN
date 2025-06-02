import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelBinarizer
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import ttest_ind

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
                cls_emb = outputs.last_hidden_state[:, 0, :].squeeze(0)
                embeddings.append(cls_emb.cpu().numpy())
        return np.vstack(embeddings)

# === Dataset locations
dataset_paths = {
    'PapagAIo': 'Text_File/Analyses/Subsets/generated_reviews_papagAIo1.0.csv',
    'Feedback': 'Text_File/Analyses/Subsets/Feedback_by_hands_sentiment.csv',
    'SecondHospital': 'Text_File/Analyses/Subsets/Subset_Second_Hospital_sentiment.csv',
    'Amazon': 'Text_File/Analyses/Subsets/Subset_Amazon_reviews_sentiment.csv'
}

# Store accuracies per dataset over runs
accuracies = {name: [] for name in dataset_paths}

# === Main loop over 10 runs and all datasets
for i in tqdm(range(10), desc="Total runs"):  # i = 0..9
    run_results = []
    run_results.append(f"=== Run {i+1} Evaluaties ===")
    for name, path in dataset_paths.items():
        # Load data
        sep = ';' if name == 'PapagAIo' else ','
        df = pd.read_csv(path, sep=sep)
        df.columns = ['Sentiment', 'Review']
        df['Review'] = df['Review'].astype(str).str.strip().str.replace('"""', '', regex=False)
        df['Sentiment'] = df['Sentiment'].str.strip()
        df = df[df['Review'].str.len() > 10]
        if df['Sentiment'].value_counts().get('Neutraal', 0) < 10:
            df = df[df['Sentiment'] != 'Neutraal']

        # Train/test split (70/30)
        X_train, X_test, y_train, y_test = train_test_split(
            df['Review'], df['Sentiment'], test_size=0.3,
            stratify=df['Sentiment'], random_state=42 + i
        )

        # Compute class weights
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes, y_train)
        cw = dict(zip(classes, weights))

        # Pipeline and train with Logistic Regression
        pipeline = Pipeline([
            ('bert', BertEmbeddingTransformer()),
            ('clf', LogisticRegression(
                class_weight=cw,
                max_iter=1000,
                solver='lbfgs',
                random_state=42 + i
            ))
        ])
        pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred = pipeline.predict(X_test)
        acc = np.mean(y_pred == y_test)
        accuracies[name].append(acc)

        report = classification_report(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        cm_df = pd.DataFrame(cm, index=classes, columns=classes)

        run_results.append(f"--- Evaluatie {name} (70/30 split) ---")
        run_results.append(report)
        run_results.append("Confusion Matrix:")
        run_results.append(cm_df.to_string())

        # AUC for binary
        if len(classes) == 2:
            y_scores = pipeline.predict_proba(X_test)[:, 1]
            y_bin = LabelBinarizer().fit_transform(y_test).ravel()
            auc = roc_auc_score(y_bin, y_scores)
            run_results.append(f"AUC-ROC: {auc:.4f}")
        run_results.append("\n")

    # Save per-run file
    save_path = f"Text_File/Train in pipeline 2.0/REGRESSION/Full_MultiDataset_LogReg_Run{i+1}.txt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(run_results))
    print(f"Saved results for Run {i+1}: {save_path}")

# === After all runs: save separate statistical summary
summary_lines = ["=== STATISTISCHE SAMENVATTING OVER 10 RUNS ==="]
# Mean and std per dataset
for name, vals in accuracies.items():
    mean_acc = np.mean(vals)
    std_acc = np.std(vals, ddof=1)
    summary_lines.append(f"{name}: Mean Accuracy = {mean_acc:.4f}, Std Dev = {std_acc:.4f}")

# T-tests
pairs = [
    ('PapagAIo', 'Feedback'),
    ('PapagAIo', 'SecondHospital'),
    ('PapagAIo', 'Amazon'),
    ('Feedback', 'SecondHospital')
]
for a, b in pairs:
    t_stat, p_val = ttest_ind(accuracies[a], accuracies[b], equal_var=False)
    summary_lines.append(f"T-test {a} vs {b}: t = {t_stat:.4f}, p = {p_val:.6f}")

# Save summary file
summary_path = f"Text_File/Train in pipeline 2.0/REGRESSION/Statistical_Summary_LogReg.txt"

