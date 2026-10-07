import numpy as np
import sklearn.metrics
import torch

from utils.utils import shuffle_data


def train_model(model, optimizer, criterion, scheduler, train_data, train_labels, epochs):
    for ep in range(epochs):
        train_data, train_labels = shuffle_data(train_data, train_labels) # Shuffle training data
        model.train()
        run_train_acc, run_train_loss = 0.0, 0.0
        for it, batch in enumerate(train_data):
            optimizer.zero_grad()
            batch = torch.tensor(batch, dtype=torch.float32).cuda()
            train_logits = model(batch)[0]
            loss = criterion(train_logits, torch.tensor(train_labels[it]).cuda())  # CE loss
            loss.backward()  # Gradient calculation
            optimizer.step()  # Gradient propagation
            train_pred = train_logits.softmax(axis=0).argmax(axis=0)
            run_train_acc += (train_pred == train_labels[it]).item()  # Train ACC
            run_train_loss += loss.item()
        train_acc = run_train_acc / len(train_data)
        train_loss = run_train_loss / len(train_data)
        scheduler.step()
        print(f'Epoch {ep+1}: Training Loss = {train_loss:.4f} - Train Acc = {train_acc:.4f}')

def validate_model(model, test_data, test_labels):
    model.eval()
    list_test_pred = []
    with torch.no_grad():
        for it, batch in enumerate(test_data):
            batch = torch.tensor(batch, dtype=torch.float32).cuda()
            test_logits = model(batch)[0]
            test_pred = test_logits.softmax(axis=0).argmax(axis=0).item()
            list_test_pred.append(test_pred)
    list_test_pred = np.stack(list_test_pred)
    conf_mx = sklearn.metrics.confusion_matrix(test_labels, list_test_pred)
    bal_acc = sklearn.metrics.balanced_accuracy_score(test_labels, list_test_pred)
    return conf_mx, bal_acc