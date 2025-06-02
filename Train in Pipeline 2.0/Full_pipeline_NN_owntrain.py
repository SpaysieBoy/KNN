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

# === BERT-embeddings transformer ===
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


# === Torch-based Neural Network Classifier ===
class TorchNNClassifier(BaseEstimator):
    def __init__(self,
                 input_dim=768,
                 hidden_dim=256,
                 epochs=5,
                 batch_size=32,
                 lr=1e-3,
                 class_weight=None,
                 random_state=0):
        """
        A simple feedforward neural network classifier using PyTorch.
        - input_dim: dimensionality van de BERT-embeddings (gewoonlijk 768).
        - hidden_dim: aantal neuronen in de verborgen laag.
        - epochs: aantal trainings-epoches.
        - batch_size: batchgrootte voor DataLoader.
        - lr: learning rate voor de optimizer.
        - class_weight: dict mapping string-labels naar gewichten voor CrossEntropyLoss.
        - random_state: seed voor reproduceerbaarheid.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self, X, y):
        """
        X: numpy array van shape (n_samples, input_dim)
        y: array-like met string-labels
        """
        # Zet de labels om naar gehele indices
        self.label_encoder_ = LabelEncoder()
        y_int = self.label_encoder_.fit_transform(y)
        num_classes = len(self.label_encoder_.classes_)

        # Bepaal class weights als Tensor, in volgorde van label_encoder_.classes_
        if self.class_weight is not None:
            weight_list = []
            for cls_label in self.label_encoder_.classes_:
                weight_list.append(self.class_weight[cls_label])
            weight_tensor = torch.tensor(weight_list, dtype=torch.float)
        else:
            weight_tensor = None

        # Definieer het netwerk
        class Net(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_classes):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(hidden_dim, num_classes)

            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                return x

        # Zet seed voor reproduceerbaarheid
        torch.manual_seed(self.random_state)

        # Instantieer model, verliesfunctie en optimizer
        self.model_ = Net(self.input_dim, self.hidden_dim, num_classes)
        if weight_tensor is not None:
            criterion = nn.CrossEntropyLoss(weight=weight_tensor.to(torch.float))
        else:
            criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

        # Zet data om naar Tensors en maak DataLoader
        X_tensor = torch.tensor(X, dtype=torch.float)
        y_tensor = torch.tensor(y_int, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Trainingsloop
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
        """
        Voorspelt labels (originele string-vormen) voor X.
        """
        X_tensor = torch.tensor(X, dtype=torch.float)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_tensor)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return self.label_encoder_.inverse_transform(preds)

    def predict_proba(self, X):
        """
        Retourneert een numpy array van shape (n_samples, n_classes) met kansschattingen.
        """
        X_tensor = torch.tensor(X, dtype=torch.float)
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

# Voor het verzamelen van accuracies per dataset over alle runs
accuracies = {name: [] for name in dataset_paths}

# === Hoofdloop over 10 runs en alle datasets ===
for i in tqdm(range(10), desc="Total runs"):  # i = 0..9
    run_results = []
    run_results.append(f"=== Run {i+1} Evaluaties ===")

    for name, path in dataset_paths.items():
        # --- Data inlezen en voorbewerken ---
        sep = ';' if name == 'PapagAIo' else ','
        df = pd.read_csv(path, sep=sep)
        df.columns = ['Sentiment', 'Review']
        df['Review'] = df['Review'].astype(str).str.strip().str.replace('"""', '', regex=False)
        df['Sentiment'] = df['Sentiment'].str.strip()
        df = df[df['Review'].str.len() > 10]
        # Indien te weinig "Neutraal"-voorbeelden, drop deze klasse
        if df['Sentiment'].value_counts().get('Neutraal', 0) < 10:
            df = df[df['Sentiment'] != 'Neutraal']

        # --- Train/test split (70/30) met stratificatie ---
        X_train, X_test, y_train, y_test = train_test_split(
            df['Review'], df['Sentiment'],
            test_size=0.3,
            stratify=df['Sentiment'],
            random_state=42 + i
        )

        # --- Klassegewichten berekenen ---
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes, y_train)
        cw = dict(zip(classes, weights))

        # --- Pipeline opbouwen en trainen met TorchNNClassifier ---
        pipeline = Pipeline([
            ('bert', BertEmbeddingTransformer()),
            ('nn', TorchNNClassifier(
                input_dim=768,           # Hidden size BERT (Bio_ClinicalBERT)
                hidden_dim=256,          # Aantal neuronen in de verborgen laag
                epochs=5,                # Aantal trainings-epoches (kan naar wens aangepast)
                batch_size=32,
                lr=1e-3,
                class_weight=cw,         # Klassegewichten voor CrossEntropyLoss
                random_state=42 + i
            ))
        ])
        pipeline.fit(X_train, y_train)

        # --- Evaluatie op de testset ---
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

        # AUC voor binaire classificatie
        if len(classes) == 2:
            y_scores = pipeline.predict_proba(X_test)[:, 1]
            y_bin = LabelBinarizer().fit_transform(y_test).ravel()
            auc = roc_auc_score(y_bin, y_scores)
            run_results.append(f"AUC-ROC: {auc:.4f}")
        run_results.append("\n")

    # --- Per-run bestand opslaan ---
    save_path = f"Text_File/Train in pipeline 2.0/REGRESSION/Full_MultiDataset_NN_Run{i+1}.txt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(run_results))
    print(f"Saved results for Run {i+1}: {save_path}")


# === Na alle runs: statische samenvatting opslaan ===
summary_lines = ["=== STATISTISCHE SAMENVATTING OVER 10 RUNS ==="]
# Gemiddelde en std per dataset
for name, vals in accuracies.items():
    mean_acc = np.mean(vals)
    std_acc = np.std(vals, ddof=1)
    summary_lines.append(f"{name}: Mean Accuracy = {mean_acc:.4f}, Std Dev = {std_acc:.4f}")

# T-toetsen tussen datasetparen
pairs = [
    ('PapagAIo', 'Feedback'),
    ('PapagAIo', 'SecondHospital'),
    ('PapagAIo', 'Amazon'),
    ('Feedback', 'SecondHospital')
]
for a, b in pairs:
    t_stat, p_val = ttest_ind(accuracies[a], accuracies[b], equal_var=False)
    summary_lines.append(f"T-test {a} vs {b}: t = {t_stat:.4f}, p = {p_val:.6f}")

# Sla de samenvatting op
summary_path = f"Text_File/Train in pipeline 2.0/REGRESSION/Statistical_Summary_NN.txt"
os.makedirs(os.path.dirname(summary_path), exist_ok=True)
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(summary_lines))
print(f"Saved statistical summary at: {summary_path}")
