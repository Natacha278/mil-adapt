import argparse
import os

import numpy as np
import pandas as pd
import torch

from utils.adapters import ZSMIL, TaskRes, CLIPAdapter, TIPAdapter
from utils.trainer import validate_model, train_model
from utils.utils import set_random_seeds, plot_confmx, get_project_data, load_data, fewshot_sampling

parser = argparse.ArgumentParser(description = "MIL Adapters experimentation")
parser.add_argument('--folder', type=str)
parser.add_argument('--output_results', type=str, default="experiments.xlsx")
# main.py: extend the choices list
parser.add_argument('--project', type=str, choices=["RCC", "NSCLC", "BRCA", "CAMELYON16"], default="NSCLC")
parser.add_argument('--text', type=str, default=None)

parser.add_argument('--aggregator', type=str, choices=["BGAP", "BGMP", "ABMIL", "TransMIL", "WIKGMIL", "ILRAMIL", "RRTMIL"], default="ABMIL")
parser.add_argument('--adapter', type=str, choices=["ZSMIL", "TaskRes", "CLIPAdapter", "TIPAdapter", "LR"], default="TaskRes")
parser.add_argument('--encoder', type=str, choices=["CONCH", "KEEP", "TITAN"], default="CONCH")
parser.add_argument('--init', type=str, choices=["ZS",  "random"], default="random")
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--k_shots', type=int, choices=[2,4,8,16], default=8)
parser.add_argument('--n_seeds', type=int, default=10)
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
parts = [args.project, args.encoder, args.adapter, args.aggregator, args.init, f"k{args.k_shots}" if args.k_shots else None]
run_name = "_".join(str(p) for p in parts if p)

if args.text:
    text_prototypes = np.load(fr"./local_data/prompts/{args.encoder}/{args.project}_{args.text}.npy")  # Zero-shot protoypes
else:
    text_prototypes = np.load(fr"./local_data/prompts/{args.encoder}/{args.project}.npy")  # Zero-shot protoypes
classes, classes_id = get_project_data(project=args.project)
X, Y, WSI = load_data(project=args.project, encoder=args.encoder, folder=args.folder, classes=classes)

for n_seed in range(args.n_seeds):
    set_random_seeds(seed_value=n_seed)  # Reproducibility
    run_name_k = f"{run_name}_seed{n_seed}"
    print(run_name_k)

    train_data, val_data, train_labels, val_labels = fewshot_sampling(X=X, Y=Y, k_shots=args.k_shots,seed=n_seed) # Few-shot sample selection

    # Model initialization
    if args.adapter == "ZSMIL":
        model = ZSMIL(text_embeddings=text_prototypes, aggregator=args.aggregator, init=args.init)
    elif args.adapter == "TaskRes":
        model = TaskRes(text_embeddings=text_prototypes, aggregator=args.aggregator)
    elif args.adapter == "CLIPAdapter":
        model = CLIPAdapter(text_embeddings=text_prototypes, aggregator=args.aggregator)
    elif args.adapter == "TIPAdapter":
        model = TIPAdapter(text_embeddings=text_prototypes, train_data=train_data, train_labels=train_labels, aggregator=args.aggregator)
    model.to(device)

    # Hyperparametre selection
    batch_size, weight_decay, adamw_beta, peak_learning_rate, epochs = 1, 1e-5, (0.9, 0.999), args.lr, args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_learning_rate, betas=adamw_beta, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    # Training & validation loops
    train_model(model,optimizer,criterion,scheduler,train_data,train_labels,epochs) # Model training
    val_cm, val_bacc = validate_model(model, val_data, val_labels) # Model validation
    plot_confmx(conf_matrix=val_cm, run_name=run_name_k, classes=classes_id, bal_acc=val_bacc) # Plot confusion matrix

    # Saving results to EXCEL file
    keys = ['Run Name', 'Project', 'Encoder', 'Adapter', 'Aggregator', 'Init', 'LRate', 'K-shots', 'Seed', 'BACC_Val']
    if os.path.exists(args.output_results):
        df = pd.read_excel(args.output_results)
    else:
        df = pd.DataFrame(columns=keys)
    run_dict = {'Run Name': [run_name_k],
                'Project': [args.project],
                'Encoder': [args.encoder],
                'Adapter': [args.adapter],
                'Aggregator': [args.aggregator],
                'Init': [args.init],
                'LRate': [args.lr],
                'K-shots': [args.k_shots],
                'Seed': [n_seed],
                'BACC_Val': [val_bacc]
                }
    new_entry = pd.DataFrame(run_dict)
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_excel(args.output_results, index=False)