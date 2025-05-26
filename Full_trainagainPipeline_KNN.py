import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelBinarizer
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.base import BaseEstimator, TransformerMixin

# === BERT-embeddings transformer
class BertEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='emilyalsentzer/Bio_ClinicalBERT', max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

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
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
                embeddings.append(cls_embedding.cpu().numpy())
        return np.vstack(embeddings)

# === Externe testbestanden
external_test_paths = {
    'feedback by hands': 'Text_File/Analyses/Subsets/Feedback_by_hands_sentiment.csv',
    'Second hospital': 'Text_File/Analyses/Subsets/Subset_Second_Hospital_sentiment.csv',
    'Amazon': 'Text_File/Analyses/Subsets/Subset_Amazon_reviews_sentiment.csv'
}

# === Hoofdloop over 10 runs
for i in tqdm(range(10), desc="Runs voltooid"):
    input_train_path = "Text_File/Analyses/Subsets/generated_reviews_papagAIo1.0.csv"
    df_pap = pd.read_csv(input_train_path, sep=";")
    df_pap.columns = ['Sentiment', 'Review']
    df_pap['Review'] = df_pap['Review'].astype(str).str.strip().str.replace('"""', '')
    df_pap['Sentiment'] = df_pap['Sentiment'].str.strip()
    df_pap = df_pap[df_pap['Review'].str.len() > 10]
    if df_pap['Sentiment'].value_counts().get('Neutraal', 0) < 10:
        df_pap = df_pap[df_pap['Sentiment'] != 'Neutraal']

    X_pap = df_pap['Review']
    y_pap = df_pap['Sentiment']
    X_pap_train, X_pap_test, y_pap_train, y_pap_test = train_test_split(
        X_pap, y_pap, test_size=0.3, stratify=y_pap, random_state=42 + i
    )

    output_path = f"Dannyfolder/Output/PapagAIo_ALL_EXTERNALS_KNN_run{i+1}.txt"
    all_results = []

    for ext_label, ext_path in external_test_paths.items():
        df_ext = pd.read_csv(ext_path)
        df_ext.columns = ['Sentiment', 'Review']
        df_ext['Review'] = df_ext['Review'].astype(str).str.strip().str.replace('"""', '')
        df_ext['Sentiment'] = df_ext['Sentiment'].str.strip()
        df_ext = df_ext[df_ext['Review'].str.len() > 10]
        if df_ext['Sentiment'].value_counts().get('Neutraal', 0) < 10:
            df_ext = df_ext[df_ext['Sentiment'] != 'Neutraal']

        X_ext = df_ext['Review']
        y_ext = df_ext['Sentiment']
        X_ext_train, X_ext_test, y_ext_train, y_ext_test = train_test_split(
            X_ext, y_ext, test_size=0.8, stratify=y_ext, random_state=42 + i
        )

        X_combined_train = pd.concat([X_pap_train, X_ext_train])
        y_combined_train = pd.concat([y_pap_train, y_ext_train])

        classes = np.unique(y_combined_train)

        pipeline = Pipeline([
            ('bert', BertEmbeddingTransformer(model_name='emilyalsentzer/Bio_ClinicalBERT')),
            ('clf', KNeighborsClassifier(n_neighbors=5))
        ])
        pipeline.fit(X_combined_train, y_combined_train)

        # === Evaluatie op PapagAIo testdata
        y_pap_pred = pipeline.predict(X_pap_test)
        report_pap = classification_report(y_pap_test, y_pap_pred, zero_division=0)
        cm_pap = confusion_matrix(y_pap_test, y_pap_pred, labels=classes)
        conf_matrix_pap = f"\n\nConfusion Matrix (PapagAIo):\n{pd.DataFrame(cm_pap, index=classes, columns=classes).to_string()}"

        auc_pap = ""
        if len(classes) == 2:
            y_scores = pipeline.predict_proba(X_pap_test)[:, 1]
            y_test_bin = LabelBinarizer().fit_transform(y_pap_test).ravel()
            auc_score = roc_auc_score(y_test_bin, y_scores)
            auc_pap = f"\n\nAUC-ROC score (PapagAIo): {auc_score:.4f}"

        # === Evaluatie op 80% testdeel van externe dataset
        y_ext_pred = pipeline.predict(X_ext_test)
        report_ext = classification_report(y_ext_test, y_ext_pred, zero_division=0)

        valid_labels = sorted(list(set(y_ext_test.unique()) & set(classes)))
        if not valid_labels:
            conf_matrix_ext = f"\n\nConfusion Matrix ({ext_label}): ⚠️ Geen overlappende klassen met PapagAIo."
        else:
            cm_ext = confusion_matrix(y_ext_test, y_ext_pred, labels=valid_labels)
            conf_matrix_ext = f"\n\nConfusion Matrix ({ext_label}):\n{pd.DataFrame(cm_ext, index=valid_labels, columns=valid_labels).to_string()}"

        auc_ext = ""
        if len(classes) == 2:
            y_ext_scores = pipeline.predict_proba(X_ext_test)[:, 1]
            y_ext_bin = LabelBinarizer().fit_transform(y_ext_test).ravel()
            if len(y_ext_scores) == len(y_ext_bin):
                auc_value = roc_auc_score(y_ext_bin, y_ext_scores)
                auc_ext = f"\n\nAUC-ROC score ({ext_label}): {auc_value:.4f}"

        # === Combineer resultaten
        results = [
            f"\n\n=== Run {i+1} met extra dataset: {ext_label} ===\n",
            "\n--- Evaluatie op PapagAIo testset ---\n", report_pap, conf_matrix_pap, auc_pap,
            f"\n--- Evaluatie op {ext_label} (80% test) ---\n", report_ext, conf_matrix_ext, auc_ext
        ]
        all_results.extend(results)

        print(f"✅ Run {i+1} met {ext_label}: resultaten toegevoegd")

    # === Schrijf alles in één bestand
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in all_results:
            f.write(line)

    print(f"📝 Alle resultaten voor run {i+1} opgeslagen in: {output_path}")
