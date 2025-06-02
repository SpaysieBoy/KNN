import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import ttest_ind
from torch.utils.data import TensorDataset, DataLoader

# === Controleer of er een CUDA-device beschikbaar is ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Gebruikt device: {device}")

# === BERT-embeddings transformer met CUDA-ondersteuning ===
class BertEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='emilyalsentzer/Bio_ClinicalBERT', max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
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
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                cls_emb = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()
                embeddings.append(cls_emb.numpy())
        return np.vstack(embeddings)

# === PyTorch Logistic Regression Classifier met CUDA ===
class TorchLogRegClassifier(BaseEstimator):
    def __init__(self,
                 input_dim=768,
                 epochs=5,
                 batch_size=32,
                 lr=1e-3,
                 class_weight=None,
                 random_state=0):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self, X, y):
        self.label_encoder_ = LabelEncoder()
        y_int = self.label_encoder_.fit_transform(y)
        num_classes = len(self.label_encoder_.classes_)

        if self.class_weight is not None:
            weights_list = [self.class_weight[lab] for lab in self.label_encoder_.classes_]
            weight_tensor = torch.tensor(weights_list, dtype=torch.float, device=device)
        else:
            weight_tensor = None

        class Net(nn.Module):
            def __init__(self, input_dim, num_classes):
                super(Net, self).__init__()
                self.linear = nn.Linear(input_dim, num_classes)

            def forward(self, x):
                return self.linear(x)

        torch.manual_seed(self.random_state)
        self.model_ = Net(self.input_dim, num_classes).to(device)

        if weight_tensor is not None:
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

        X_tensor = torch.tensor(X, dtype=torch.float).to(device)
        y_tensor = torch.tensor(y_int, dtype=torch.long).to(device)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                logits = self.model_(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float).to(device)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_tensor)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return self.label_encoder_.inverse_transform(preds)

    def predict_proba(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float).to(device)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

# === Dataset locaties ===
dataset_paths = {
    'PapagAIo': 'Text_File/Analyses/Subsets/generated_reviews_papagAIo1.0.csv',
    'Feedback': 'Text_File/Analyses/Subsets/Feedback_by_hands_sentiment.csv',
    'SecondHospital': 'Text_File/Analyses/Subsets/Subset_Second_Hospital_sentiment.csv',
    'Amazon': 'Text_File/Analyses/Subsets/Subset_Amazon_reviews_sentiment.csv'
}

accuracies = {name: [] for name in dataset_paths}

# === Hoofdloop over 10 runs en alle datasets ===
for i in tqdm(range(10), desc="Total runs"):
    run_results = []
    run_results.append(f"=== Run {i+1} Evaluaties ===")

    for name, path in dataset_paths.items():
        sep = ';' if name == 'PapagAIo' else ','
        df = pd.read_csv(path, sep=sep)
        df.columns = ['Sentiment', 'Review']
        df['Review'] = df['Review'].astype(str).str.strip().str.replace('"""', '', regex=False)
        df['Sentiment'] = df['Sentiment'].str.strip()
        df = df[df['Review'].str.len() > 10]
        if df['Sentiment'].value_counts().get('Neutraal', 0) < 10:
            df = df[df['Sentiment'] != 'Neutraal']

        X_train, X_test, y_train, y_test = train_test_split(
            df['Review'], df['Sentiment'],
            test_size=0.3,
            stratify=df['Sentiment'],
            random_state=42 + i
        )

        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes, y_train)
        cw = dict(zip(classes, weights))

        pipeline = Pipeline([
            ('bert', BertEmbeddingTransformer()),
            ('logreg', TorchLogRegClassifier(
                input_dim=768,
                epochs=5,
                batch_size=32,
                lr=1e-3,
                class_weight=cw,
                random_state=42 + i
            ))
        ])
        pipeline.fit(X_train, y_train)

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

        if len(classes) == 2:
            y_scores = pipeline.predict_proba(X_test)[:, 1]
            y_bin = LabelBinarizer().fit_transform(y_test).ravel()
            auc = roc_auc_score(y_bin, y_scores)
            run_results.append(f"AUC-ROC: {auc:.4f}")
        run_results.append("\n")

    save_path = f"Text_File/Train in pipeline 2.0/REGRESSION/Full_MultiDataset_LogReg_CUDA_Run{i+1}.txt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(run_results))
    print(f"Saved results for Run {i+1}: {save_path}")

# === Na alle runs: statistische samenvatting opslaan ===
summary_lines = ["=== STATISTISCHE SAMENVATTING OVER 10 RUNS ==="]
for name, vals in accuracies.items():
    mean_acc = np.mean(vals)
    std_acc = np.std(vals, ddof=1)
    summary_lines.append(f"{name}: Mean Accuracy = {mean_acc:.4f}, Std Dev = {std_acc:.4f}")

pairs = [
    ('PapagAIo', 'Feedback'),
    ('PapagAIo', 'SecondHospital'),
    ('PapagAIo', 'Amazon'),
    ('Feedback', 'SecondHospital')
]
for a, b in pairs:
    t_stat, p_val = ttest_ind(accuracies[a], accuracies[b], equal_var=False)
    summary_lines.append(f"T-test {a} vs {b}: t = {t_stat:.4f}, p = {p_val:.6f}")

summary_path = f"Text_File/Train in pipeline 2.0/REGRESSION/Statistical_Summary_LogReg_CUDA.txt"
os.makedirs(os.path.dirname(summary_path), exist_ok=True)
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(summary_lines))
print(f"Saved statistical summary at: {summary_path}")
