import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm


def plot_confmx(conf_matrix, run_name, classes, bal_acc):
    path_results = "./local_data/results/cfmx/"
    os.makedirs(path_results, exist_ok=True)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f"BALACC = {bal_acc:.4f}")
    path_save = os.path.join(path_results, f"cfmx_{run_name}.png")
    plt.savefig(path_save)
    plt.close()

def shuffle_data(data, labels):
    combined_data = list(zip(data, labels))
    random.shuffle(combined_data)
    data, labels = zip(*combined_data)
    data, labels = list(data), np.array(labels)
    return data, labels

def get_project_data(project):
    if project == "RCC": # Renal Cell Carcinoma (RCC)
        classes = ["Renal cell carcinoma, chromophobe type",
                   "Clear cell adenocarcinoma, NOS",
                   "Papillary adenocarcinoma, NOS"]
        classes_id = ["CHRCC", "CCRCC", "PRCC"]

    elif project == "BRCA": # BReast CAncer (BRCA)
        classes = ["Infiltrating duct carcinoma, NOS", "Lobular carcinoma, NOS"]
        classes_id = ["IDC", "ILC"]

    elif project == "NSCLC": # Non-Small Cell Lung Cancer (NSCLC)
        classes = ["Adenocarcinoma, NOS", "Squamous cell carcinoma, NOS"]
        classes_id = ["LUAD", "LUSC"]

    elif project == "CAMELYON16":
        classes = ["Normal", "Tumor"]
        classes_id = ["Normal", "Tumor"]

    return classes, classes_id

def load_data(folder, project, encoder, classes):
    csv_file = fr"./local_data/csv/{project}.csv"
    data = pd.read_csv(csv_file, delimiter=",")
    list_WSI = data['WSI'].values
    labels = data['GT'].values
    labels = [classes.index(item) for item in labels]  # Map classes_id to labels
    labels = np.array(labels, dtype=np.int64)
    embeddings = [np.load(os.path.join(folder, project, encoder, f"{file_name}.npy")) for file_name in tqdm(list_WSI)]
    return embeddings, labels, list_WSI

def load_data_subset(project, folder, classes, set_data):
    csv_file = os.path.join("./local_data/csv/", project, f"{project}_{set_data}.csv")
    data = pd.read_csv(csv_file, delimiter=",")
    list_WSI = data['WSI'].values
    labels = data['GT'].values
    labels = np.array([classes.index(item) for item in labels], dtype=np.int64)  # Map classes_id to labels
    patch_embd = [np.load(os.path.join(folder, project, file_name + ".npy")) for file_name in tqdm(list_WSI)]
    return patch_embd, labels

def set_random_seeds(seed_value=42):
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def fewshot_sampling(X, Y, k_shots, seed=None):
    if seed is not None:
        np.random.seed(seed)
    ids = np.arange(len(X))
    train_ids = []
    n_classes = len(np.unique(Y))
    for cls in range(n_classes):
        cls_ids = ids[Y == cls]
        train_ids.extend(np.random.choice(cls_ids, size=k_shots, replace=False))
    train_ids, val_ids= np.array(train_ids), np.setdiff1d(ids, train_ids)
    train_data, val_data = [X[i] for i in train_ids], [X[i] for i in val_ids]
    train_labels, val_labels = Y[train_ids], Y[val_ids]
    return train_data, val_data, train_labels, val_labels