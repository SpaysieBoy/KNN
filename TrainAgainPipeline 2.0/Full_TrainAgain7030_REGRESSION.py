import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelBinarizer
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import ttest_rel

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

# === Externe testbestanden en hun paden
external_test_paths = {
    'feedback by hands': 'Text_File/Analyses/Subsets/Feedback_by_hands_sentiment.csv',
    'Second hospital': 'Text_File/Analyses/Subsets/Subset_Second_Hospital_sentiment.csv',
    'Amazon': 'Text_File/Analyses/Subsets/Subset_Amazon_reviews_sentiment.csv'
}

# === Structuren om accuracies te verzamelen per run per dataset
# Voor elke externe dataset houden we twee lijsten bij:
#   - pap_acc[ext_label]: accuracies op PapagAIo-testset wanneer getraind met die externe dataset
#   - ext_acc[ext_label]: accuracies op de externe testset zelf
pap_acc = {ext_label: [] for ext_label in external_test_paths}
ext_acc = {ext_label: [] for ext_label in external_test_paths}

# === Hoofdloop over 10 runs
for i in tqdm(range(10), desc="Runs voltooid"):
    # 1) PapagAIo dataset inladen en opschonen
    input_train_path = "Text_File/Analyses/Subsets/generated_reviews_papagAIo1.0.csv"
    df_pap = pd.read_csv(input_train_path, sep=";")
    df_pap.columns = ['Sentiment', 'Review']
    df_pap['Review'] = df_pap['Review'].astype(str).str.strip().str.replace('"""', '', regex=False)
    df_pap['Sentiment'] = df_pap['Sentiment'].str.strip()
    df_pap = df_pap[df_pap['Review'].str.len() > 10]
    if df_pap['Sentiment'].value_counts().get('Neutraal', 0) < 10:
        df_pap = df_pap[df_pap['Sentiment'] != 'Neutraal']

    # 2) Splits PapagAIo: 70% train, 30% test
    X_pap = df_pap['Review']
    y_pap = df_pap['Sentiment']
    X_pap_train, X_pap_test, y_pap_train, y_pap_test = train_test_split(
        X_pap, y_pap, test_size=0.3, stratify=y_pap, random_state=42 + i
    )

    # 3) Bepaal hoe groot de externe trainingssubset moet zijn, zodat PapagAIo:extern = 70:30 verhouding
    n_pap_train = len(X_pap_train)
    n_ext_train = int(round(n_pap_train * (0.3 / 0.7)))

    # Pad voor outputbestand per run
    output_path = f"Dannyfolder/Output/PapagAIo_ALL_EXTERNALS_run{i+1}.txt"
    all_results = []

    # 4) Voor elke externe dataset: inladen, opschonen, stratified sample trekken, trainen en evalueren
    for ext_label, ext_path in external_test_paths.items():
        # 4a) Externe dataset inlezen en opschonen
        df_ext = pd.read_csv(ext_path)
        df_ext.columns = ['Sentiment', 'Review']
        df_ext['Review'] = df_ext['Review'].astype(str).str.strip().str.replace('"""', '', regex=False)
        df_ext['Sentiment'] = df_ext['Sentiment'].str.strip()
        df_ext = df_ext[df_ext['Review'].str.len() > 10]
        if df_ext['Sentiment'].value_counts().get('Neutraal', 0) < 10:
            df_ext = df_ext[df_ext['Sentiment'] != 'Neutraal']

        X_ext = df_ext['Review']
        y_ext = df_ext['Sentiment']

        # 4b) Stratified sample nemen uit de externe dataset: train_size = n_ext_train, rest is test
        max_ext_available = len(X_ext)
        if n_ext_train >= max_ext_available:
            train_frac = 0.9
            X_ext_train, X_ext_test, y_ext_train, y_ext_test = train_test_split(
                X_ext, y_ext, train_size=train_frac, stratify=y_ext, random_state=42 + i
            )
        else:
            X_ext_train, X_ext_test, y_ext_train, y_ext_test = train_test_split(
                X_ext, y_ext, train_size=n_ext_train, stratify=y_ext, random_state=42 + i
            )

        # 4c) Combineer PapagAIo-trainingsset en externe trainingssubset
        X_combined_train = pd.concat([
            X_pap_train.reset_index(drop=True),
            X_ext_train.reset_index(drop=True)
        ])
        y_combined_train = pd.concat([
            y_pap_train.reset_index(drop=True),
            y_ext_train.reset_index(drop=True)
        ])

        # 4d) Bereken class weights voor de Logistic Regression
        classes = np.unique(y_combined_train)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_combined_train
        )
        class_weight_dict = dict(zip(classes, class_weights))

        # 4e) Pipeline bouwen en trainen met Logistic Regression
        pipeline = Pipeline([
            ('bert', BertEmbeddingTransformer(model_name='emilyalsentzer/Bio_ClinicalBERT')),
            ('clf', LogisticRegression(
                class_weight=class_weight_dict,
                solver='lbfgs',
                max_iter=1000,
                multi_class='auto',
                random_state=42 + i
            ))
        ])
        pipeline.fit(X_combined_train, y_combined_train)

        # 5) Evaluatie op PapagAIo-testset
        y_pap_pred = pipeline.predict(X_pap_test)
        report_pap = classification_report(y_pap_test, y_pap_pred, zero_division=0)
        cm_pap = confusion_matrix(y_pap_test, y_pap_pred, labels=classes)
        conf_matrix_pap = (
            f"\n\nConfusion Matrix (PapagAIo):\n"
            f"{pd.DataFrame(cm_pap, index=classes, columns=classes).to_string()}"
        )
        acc_pap = accuracy_score(y_pap_test, y_pap_pred)

        auc_pap = ""
        if len(classes) == 2:
            y_scores = pipeline.predict_proba(X_pap_test)[:, 1]
            y_test_bin = LabelBinarizer().fit_transform(y_pap_test).ravel()
            auc_score = roc_auc_score(y_test_bin, y_scores)
            auc_pap = f"\n\nAUC-ROC score (PapagAIo): {auc_score:.4f}"

        # 6) Evaluatie op externe testset (de rest van de externe data)
        y_ext_pred = pipeline.predict(X_ext_test)
        report_ext = classification_report(y_ext_test, y_ext_pred, zero_division=0)
        acc_ext_val = accuracy_score(y_ext_test, y_ext_pred)

        valid_labels = sorted(list(set(y_ext_test.unique()) & set(classes)))
        if not valid_labels:
            conf_matrix_ext = f"\n\nConfusion Matrix ({ext_label}): Geen overlappende klassen met PapagAIo."
        else:
            cm_ext = confusion_matrix(y_ext_test, y_ext_pred, labels=valid_labels)
            conf_matrix_ext = (
                f"\n\nConfusion Matrix ({ext_label}):\n"
                f"{pd.DataFrame(cm_ext, index=valid_labels, columns=valid_labels).to_string()}"
            )

        auc_ext = ""
        if len(classes) == 2:
            y_ext_scores = pipeline.predict_proba(X_ext_test)[:, 1]
            y_ext_bin = LabelBinarizer().fit_transform(y_ext_test).ravel()
            if len(y_ext_scores) == len(y_ext_bin):
                auc_value = roc_auc_score(y_ext_bin, y_ext_scores)
                auc_ext = f"\n\nAUC-ROC score ({ext_label}): {auc_value:.4f}"

        # 7) Voeg de accuracies toe aan de verzamelstructuren
        pap_acc[ext_label].append(acc_pap)
        ext_acc[ext_label].append(acc_ext_val)

        # 8) Combineer resultaten in een lijst voor dit ext_label
        results = [
            f"\n\n=== Run {i+1} met extra dataset: {ext_label} ===\n",
            "\n--- Evaluatie op PapagAIo testset ---\n", report_pap, conf_matrix_pap, auc_pap,
            f"\n--- Evaluatie op {ext_label} (externe testset) ---\n", report_ext, conf_matrix_ext, auc_ext,
            f"\nAccuracy PapagAIo: {acc_pap:.4f}", f"\nAccuracy {ext_label}: {acc_ext_val:.4f}"
        ]
        all_results.extend(results)

        print(
            f" Run {i+1} met {ext_label}: "
            f"acc_pap={acc_pap:.4f}, acc_ext={acc_ext_val:.4f}"
        )

    # 9) Schrijf alle resultaten naar één bestand per run
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in all_results:
            f.write(line)

    print(f" Alle resultaten voor run {i+1} opgeslagen in: {output_path}")

# === Na de 10 runs: bereken gemiddeldes, standaarddeviaties en voer gepaarde t-test uit voor elke externe dataset
summary_lines = ["=== STATISTISCHE SAMENVATTING OVER 10 RUNS ===\n"]
for ext_label in external_test_paths.keys():
    pap_vals = np.array(pap_acc[ext_label])
    ext_vals = np.array(ext_acc[ext_label])

    mean_pap = np.mean(pap_vals)
    std_pap = np.std(pap_vals, ddof=1)
    mean_ext = np.mean(ext_vals)
    std_ext = np.std(ext_vals, ddof=1)

    # Gepaarde t-test tussen PapagAIo-accuracy en externe accuracy voor deze ext_label
    t_stat, p_value = ttest_rel(pap_vals, ext_vals)

    summary_lines.append(f"--- Vergelijking PapagAIo vs {ext_label} ---")
    summary_lines.append(f"PapagAIo (mean acc)= {mean_pap:.4f}, std= {std_pap:.4f}")
    summary_lines.append(f"{ext_label} (mean acc)= {mean_ext:.4f}, std= {std_ext:.4f}")
    summary_lines.append(f"Gepaarde t-test: t-stat= {t_stat:.4f}, p-value= {p_value:.6f}\n")

# Opslaan van de samenvatting in een eigen bestand
summary_path = "Dannyfolder/Output/PapagAIo_ALL_EXTERNALS_Statistical_Summary.txt"
os.makedirs(os.path.dirname(summary_path), exist_ok=True)
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(summary_lines))

print(f"Statistische samenvatting opgeslagen in: {summary_path}")
