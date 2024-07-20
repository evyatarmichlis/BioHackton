import torch
import esm
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, roc_auc_score
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import xgboost as xgb
from sklearn.manifold import TSNE
import umap.umap_ as umap
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from joblib import dump, load

import optuna

#
# def objective(trial,X,y):
#     # Define the hyperparameters and their ranges
#     param = {
#         'verbosity': 0,
#         'objective': 'binary:logistic',
#         'eval_metric': 'auc',
#         'booster': 'gbtree',
#         'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
#         'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
#         'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.4, 1.0),
#         'subsample': trial.suggest_uniform('subsample', 0.4, 1.0),
#         'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.3),
#         'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
#         'max_depth': trial.suggest_int('max_depth', 1, 12),
#         'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
#         'gamma': trial.suggest_loguniform('gamma', 1e-3, 10.0)
#     }
#
#     # Load your dataset here
#     # Assuming X and y are your features and labels
#     # X, y = load_your_data_function()
#
#     # Example: Load a sample dataset
#     # df = pd.read_csv('your_dataset.csv')
#     # X = df.drop(columns=['target'])
#     # y = df['target']
#
#     # Split the data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     # Create the XGBoost model
#     model = xgb.XGBClassifier(**param)
#
#     # Train the model
#     model.fit(X_train, y_train)
#
#     # Predict on the test set
#     y_pred_proba = model.predict_proba(X_test)[:, 1]
#
#     # Calculate AUC
#     auc = roc_auc_score(y_test, y_pred_proba)
#
#     return auc

# All of ESM-2 pre-trained models by embedding size
ESM_MODELS_DICT = {320: esm.pretrained.esm2_t6_8M_UR50D,
                   480: esm.pretrained.esm2_t12_35M_UR50D,
                   640: esm.pretrained.esm2_t30_150M_UR50D,
                   1280: esm.pretrained.esm2_t33_650M_UR50D,
                   2560: esm.pretrained.esm2_t36_3B_UR50D,
                   5120: esm.pretrained.esm2_t48_15B_UR50D}

def load_peptide_data(data_csv="DB/NesDB_all_CRM1_with_peptides_train.csv", include_nesdoubt=True, include_nodoubt=True, max_peptide_len=22):
    df = pd.read_csv(data_csv).dropna(subset=['Peptide_sequence', 'Negative_sequence', 'Sequence'])
    if not include_nesdoubt:
        df = df[df['is_NesDB_doubt'] != True].reset_index(drop=True)
    if not include_nodoubt:
        df = df[df['is_NesDB_doubt'] != False].reset_index(drop=True)


    pos_pep = []
    neg_pep = []
    data_doubt = []
    peps_hashs = []
    counter = 0

    for index, row in df.iterrows():
        pep = row['Peptide_sequence']
        neg = row['Negative_sequence']
        pep_hash = row["Peptide_hash"]
        if len(pep) <= max_peptide_len and pep != '' and len(neg) <= max_peptide_len and neg != '':
            pos_pep.append((f"{counter}", pep))
            neg_pep.append((f"{counter}", neg))
            peps_hashs.append(pep_hash)
            data_doubt.append(row['is_NesDB_doubt'])
            counter += 1

    return pos_pep, neg_pep, data_doubt,peps_hashs

def get_esm_model(embedding_size=1280):
    if embedding_size not in ESM_MODELS_DICT:
        raise ValueError(f"ERROR: ESM does not have a trained model with embedding size of {embedding_size}.\n "
                         f"Please use one of the following embedding sized: {ESM_MODELS_DICT.keys()}")

    model, alphabet = ESM_MODELS_DICT[embedding_size]()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    device = 'cpu'
    model.to(device)
    print(f"ESM model loaded to {device}")
    return model, alphabet, batch_converter, device

def get_esm_sequence_embedding(pep_tuple_list, esm_model, alphabet, batch_converter, device, embedding_layer=33,per_amino_acid=False):
    batch_labels, batch_strs, batch_tokens = batch_converter(pep_tuple_list)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    with torch.no_grad():
        results = esm_model(batch_tokens.to(device), repr_layers=[embedding_layer])
    token_representations = results["representations"][embedding_layer]

    if per_amino_acid:
        sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1:tokens_len - 1].cpu().numpy())
    else:
        sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(list(token_representations[i, 1: tokens_len - 1].mean(0).cpu().numpy()))



    return sequence_representations
#
# def cluster_embeddings(embeddings, n_clusters=2):
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     cluster_labels = kmeans.fit_predict(embeddings)
#     silhouette_avg = silhouette_score(embeddings, cluster_labels)
#     print(f'Silhouette Score: {silhouette_avg}')
#     return cluster_labels, kmeans
#
# def train_classifier(X_train, y_train):
#     classifier = xgb.XGBClassifier(n_estimators=100, random_state=42)
#     classifier.fit(X_train, y_train)
#     return classifier

def plot_boxplot(data_dict, out_file_path="boxplot.png"):
    plot_data = [data_dict[key] for key in data_dict]
    labels = list(data_dict.keys())
    plt.figure(figsize=(10, 6))
    plt.boxplot(plot_data, patch_artist=True, tick_labels=labels)
    plt.xlabel('Label')
    plt.ylabel('Mean Distance to positive training set')
    plt.title('Boxplot')
    plt.grid(True)
    plt.savefig(out_file_path)

    def plot_roc_curve(y_test, y_scores, out_file_path="roc_curve.png"):
        fpr, tpr, thresholds = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)
        print(f"AUC: {roc_auc}")
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(out_file_path)
        plt.show()

def plot_tsne_umap(embeddings, labels, method='tsne', out_file_path="tsne_umap_plot.png"):
    if method == 'tsne':
        reducer = TSNE(n_components=3, random_state=42)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=3, random_state=42)
    else:
        raise ValueError("Method should be either 'tsne' or 'umap'.")

    reduced_embeddings = reducer.fit_transform(np.array(embeddings))
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    for label in np.unique(labels):
        indices = np.where(labels == label)
        ax.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1], reduced_embeddings[indices, 2], label=f'Class {label}', alpha=0.5)
    ax.legend()
    ax.set_title(f'{method.upper()} Plot')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    plt.grid(True)
    plt.savefig(out_file_path)
    plt.show()


def get_peptide_distances(pos_neg_test_peptides, pos_train_peptides, reduce_func=np.mean):
    distances = cdist(np.array(pos_neg_test_peptides), np.array(pos_train_peptides), metric="euclidean")
    reduced_distances = reduce_func(distances, axis=-1)
    return reduced_distances
#
# if __name__ == '__main__':
#     trail_name = "optuna"
#
#     chosen_embedding_size = 1280
#     chosen_embedding_layer = 33
#     chosen_test_size = 0.1
#     n_clusters = 2
#
#     print("Loading peptide data")
#     positive_pep, negative_pep, doubt_lables,per_hash = load_peptide_data()
#
#     print("Loading ESM-2 model")
#     model_esm, alphabet_esm, batch_converter_esm, device_esm = get_esm_model(embedding_size=chosen_embedding_size)
#
#     print("Getting the ESM-2 embeddings for all the peptide data")
#     positive_esm_emb = get_esm_sequence_embedding(positive_pep, model_esm, alphabet_esm, batch_converter_esm, device_esm, embedding_layer=chosen_embedding_layer)
#     negative_esm_emb = get_esm_sequence_embedding(negative_pep, model_esm, alphabet_esm, batch_converter_esm, device_esm, embedding_layer=chosen_embedding_layer)
#
#     print("Clustering the embeddings")
#     all_embeddings = positive_esm_emb + negative_esm_emb
#     all_labels = [1] * len(positive_esm_emb) + [0] * len(negative_esm_emb)
#     # Shuffle the labels
#     # combined = list(zip(all_embeddings, all_labels))
#
#
#     # all_embeddings, all_labels = zip(*combined)
#     all_labels = np.array(all_labels)
#     # cluster_labels, kmeans = cluster_embeddings(all_embeddings, n_clusters)
#     #
#     # cluster_results = pd.DataFrame({
#     #     'Embedding': all_embeddings,
#     #     'Label': all_labels,
#     #     'Cluster': cluster_labels
#     # })
#
#     # Plot t-SNE and UMAP for visualization
#     # print("Plotting t-SNE and UMAP")
#     # plot_tsne_umap(all_embeddings, all_labels, method='tsne', out_file_path="tsne_plot.png")
#     # plot_tsne_umap(all_embeddings, all_labels, method='umap', out_file_path="umap_plot.png")
#
#     # # Identify which cluster corresponds to positive sequences and which to negative
#     # positive_cluster = 1 if cluster_results[cluster_results['Label'] == 1]['Cluster'].value_counts().idxmax() == 1 else 0
#     # negative_cluster = 1 - positive_cluster
#     #
#     # # Create new labels based on clustering
#     # refined_labels = [1 if cluster == positive_cluster else 0 for cluster in cluster_labels]
#
#     # Split data into train and test sets
#     X = all_embeddings
#     y = all_labels
#     X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(X, y, range(len(X)), test_size=0.2, random_state=42)
#
#     # X_train, X_test, y_train, y_test = train_test_split(all_embeddings, all_labels, test_size=chosen_test_size, random_state=42)
#
#     # study = optuna.create_study(direction='maximize')
#     # study.optimize(lambda trial: objective(trial, all_embeddings, all_labels), n_trials=100)
#     #
#     # # Print the best hyperparameters
#     # print("Best hyperparameters: ", study.best_params)
#     # print("Best AUC: ", study.best_value)
#
#     # Train the final model using the best hyperparameters
#     best_params = {'lambda': 0.022225907943040437, 'alpha': 0.0033409749371986854, 'colsample_bytree': 0.7279431945093569, 'subsample': 0.5578983736704198, 'learning_rate': 0.027037034347036627, 'n_estimators': 597, 'max_depth': 4, 'min_child_weight': 2, 'gamma': 0.10567047820304272}
#
#     final_model = xgb.XGBClassifier(**best_params)
#     classifier = final_model.fit(X_train, y_train)
#     dump(classifier, 'final_model.joblib')
#     classifier = load('final_model.joblib')
#
#     # Train classifier
#     # classifier = train_classifier(X_train, y_train)
#
#     # Predict on test set
#     y_pred = classifier.predict(X_test)
#     y_proba = classifier.predict_proba(X_test)[:, 1]
#     test_hashes = np.array(per_hash)[test_idx]
#     results = pd.DataFrame({
#         'hash': test_hashes,
#         'true_label': y_test,
#         'predicted_label': y_pred
#     })
#     results.to_csv('test_hash_results.csv', index=False)
#     # Evaluation
#     print(classification_report(y_test, y_pred))
#     print(confusion_matrix(y_test, y_pred))
#     plot_roc_curve(y_test, y_proba, f"roc_curve {trail_name}.png")
#
#
# #Xboost -
# #               precision    recall  f1-score   support
# #
# #            0       0.81      0.95      0.88        22
# #            1       0.96      0.84      0.90        31
# #
# #     accuracy                           0.89        53
# #    macro avg       0.89      0.90      0.89        53
# # weighted avg       0.90      0.89      0.89        53
#
#
# #Best hyperparameters:  {'lambda': 0.022225907943040437, 'alpha': 0.0033409749371986854, 'colsample_bytree': 0.7279431945093569, 'subsample': 0.5578983736704198, 'learning_rate': 0.027037034347036627, 'n_estimators': 597, 'max_depth': 4, 'min_child_weight': 2, 'gamma': 0.10567047820304272}
#
# #results was optuna- finding best model.
# # Best AUC:  0.9339285714285714
# #               precision    recall  f1-score   support
# #
# #            0       0.83      0.91      0.87        22
# #            1       0.93      0.87      0.90        31
# #
# #     accuracy                           0.89        53
# #    macro avg       0.88      0.89      0.88        53
# # weighted avg       0.89      0.89      0.89        53