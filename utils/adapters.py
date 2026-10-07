import numpy as np
import torch
import torch.nn as nn

from utils.MIL import ABMIL, TransMIL, WIKGMIL
from utils.MIL.ILRAMIL import ILRA
from utils.MIL.RRTMIL import RRTMIL

class ZSMIL(torch.nn.Module):
    def __init__(self, text_embeddings, init="ZS", aggregator="BGAP"):
        super(ZSMIL, self).__init__()
        self.text_embeddings = torch.tensor(text_embeddings).clone().cuda()
        self.init = init
        L, n_classes = text_embeddings.shape

        # Classifier initialization
        torch.manual_seed(42) # Fix seed for variability only in few-shot sample selection
        if self.init == "ZS":
            self.classifier = torch.nn.Parameter(self.text_embeddings.clone())
        elif self.init == "random":
            self.classifier = torch.nn.Linear(L,n_classes)

        # Aggregator initialization
        self.aggregator = aggregator
        self.ABMIL = ABMIL(L=L) if self.aggregator == "ABMIL" else None
        self.TransMIL = TransMIL(L=L) if self.aggregator == "TransMIL" else None
        self.WIKGMIL = WIKGMIL(in_dim=L, embed_dim=L) if self.aggregator == "WIKGMIL" else None
        self.ILRAMIL = ILRA(in_dim=L, embed_dim=L, num_attention_layers=1) if self.aggregator == "ILRAMIL" else None
        self.RRTMIL = RRTMIL(in_dim=L, embed_dim=L, trans_dim=L//8, mlp_dim=L, n_layers=2, n_heads=4) if self.aggregator == "RRTMIL" else None
        self.logit_scale = 56.347694396972656

    def forward(self, features):
        if self.aggregator == "BGAP":
            embedding = torch.mean(features, dim=0)
        elif self.aggregator == "BGMP":
            embedding = torch.max(features, dim=0)[0]
        elif self.aggregator == "ABMIL":
            embedding = self.ABMIL(features)[0]
        elif self.aggregator == "TransMIL":
            embedding = self.TransMIL(features)
        elif self.aggregator == "WIKGMIL":
            embedding = self.WIKGMIL(features)
        elif self.aggregator == "ILRAMIL":
            embedding = self.ILRAMIL(features)
        elif self.aggregator == "RRTMIL":
            embedding = self.RRTMIL(features)

        if self.init == "ZS":
            prototype = self.classifier
            prototype = prototype / prototype.norm(dim=0,keepdim=True)
            embedding = embedding / embedding.norm(dim=-1,keepdim=True)
            output = embedding @ prototype * self.logit_scale
        elif self.init == "random":
            output = self.classifier(embedding)
        return output, embedding

class TaskRes(torch.nn.Module):
    def __init__(self, text_embeddings, alpha=0.25, aggregator="BGAP"):
        super(TaskRes, self).__init__()
        self.alpha = alpha
        self.aggregator = aggregator
        self.ABMIL = ABMIL(L= text_embeddings.shape[0]) if self.aggregator == "ABMIL" else None
        self.TransMIL = TransMIL(L=text_embeddings.shape[0]) if self.aggregator == "TransMIL" else None
        self.text_embeddings = torch.tensor(text_embeddings).clone().cuda()
        self.classifier = torch.nn.Parameter(self.text_embeddings.clone())
        self.logit_scale = 56.347694396972656

    def forward(self, features):
        if self.aggregator == "BGAP":
            embedding = torch.mean(features, dim=0)
        elif self.aggregator == "BGMP":
            embedding = torch.max(features, dim=0)[0]
        elif self.aggregator == "ABMIL":
            embedding, w = self.ABMIL(features)
        elif self.aggregator == "TransMIL":
            embedding = self.TransMIL(features)
        prototype = self.classifier
        prototype = self.text_embeddings + self.alpha * prototype
        prototype_norm = prototype / prototype.norm(dim=0,keepdim=True)
        embedding_norm = embedding / embedding.norm(dim=-1,keepdim=True)
        output = embedding_norm @ prototype_norm * self.logit_scale
        return output, embedding_norm

class CLIPAdapter(torch.nn.Module):
    def __init__(self, text_embeddings, aggregator = "BGAP"):
        super(CLIPAdapter, self).__init__()
        self.c_in, self.reduction, self.ratio = 512, 4, 0.2
        self.aggregator = aggregator
        self.logit_scale = 56.347694396972656
        self.ABMIL = ABMIL(L= text_embeddings.shape[0]) if self.aggregator == "ABMIL" else None
        self.TransMIL = TransMIL(L=text_embeddings.shape[0]) if self.aggregator == "TransMIL" else None
        self.text_embeddings = torch.tensor(text_embeddings).clone().cuda()
        self.classifier = torch.nn.Parameter(self.text_embeddings.clone())
        self.classifier.requires_grad = False
        self.adapter = torch.nn.Sequential(torch.nn.Linear(self.c_in, self.c_in // self.reduction, bias=False),
                                           torch.nn.ReLU(inplace=True),
                                           torch.nn.Linear(self.c_in // self.reduction, self.c_in, bias=False),
                                           torch.nn.ReLU(inplace=True))

    def forward(self, features):
        if self.aggregator == "BGAP":
            embedding = torch.mean(features, dim=0)
        elif self.aggregator == "BGMP":
            embedding = torch.max(features, dim=0)[0]
        elif self.aggregator == "ABMIL":
            embedding, w = self.ABMIL(features)
        elif self.aggregator == "TransMIL":
            embedding = self.TransMIL(features)
        prototype = self.classifier
        embedding_res = self.adapter(embedding)
        embedding = self.ratio * embedding_res + (1-self.ratio)*embedding
        embedding_norm = embedding / embedding.norm(dim=-1, keepdim=True)
        prototype_norm = prototype / prototype.norm(dim=0,keepdim=True)
        output = embedding_norm @ prototype_norm * self.logit_scale
        return output, embedding_norm

class TIPAdapter(torch.nn.Module):
    def __init__(self, text_embeddings, train_data, train_labels, aggregator = "BGAP"):
        super(TIPAdapter, self).__init__()
        self.beta, self.alpha = 5, 1
        self.aggregator = aggregator
        self.logit_scale = 56.347694396972656
        self.ABMIL = ABMIL(L=text_embeddings.shape[0]) if self.aggregator == "ABMIL" else None
        self.TransMIL = TransMIL(L=text_embeddings.shape[0]) if self.aggregator == "TransMIL" else None
        self.text_embeddings = torch.tensor(text_embeddings).clone().cuda()
        self.classifier = torch.nn.Parameter(self.text_embeddings)
        self.classifier.requires_grad = False

        # Building cache
        self.cache_keys, self.cache_values = [], []
        train_data = [np.mean(train_data_idx,axis=0) for train_data_idx in train_data] # BGAP
        train_data = np.stack(train_data)
        self.cache_keys = torch.tensor(train_data).to(torch.float32).cuda()
        self.cache_keys = torch.nn.Parameter(self.cache_keys.clone())
        self.cache_keys.requires_grad = True
        self.cache_values = torch.nn.functional.one_hot(torch.tensor(train_labels)).to(torch.float32).cuda()
        self.cache_values = nn.Parameter(self.cache_values.clone())
        self.cache_values.requires_grad = False

    def forward(self, features):
        if self.aggregator == "BGAP":
            embedding = torch.mean(features, dim=0)
        elif self.aggregator == "BGMP":
            embedding = torch.max(features, dim=0)[0]
        elif self.aggregator == "ABMIL":
            embedding = self.ABMIL(features)
        elif self.aggregator == "TransMIL":
            embedding = self.TransMIL(features)
        prototype = self.classifier
        embedding_norm = embedding / embedding.norm(dim=-1, keepdim=True)
        prototype_norm = prototype / prototype.norm(dim=0,keepdim=True)
        clip_logits = embedding_norm @ prototype_norm * self.logit_scale

        cache_keys = self.cache_keys / self.cache_keys.norm(dim=-1, keepdim=True)
        affinity = embedding_norm @ cache_keys.t() # Afinitty to all cache samples
        affinity = torch.exp(((-1) * (self.beta - self.beta * affinity)))
        cache_logits = affinity @ self.cache_values # Sum of similarities from the query to all the support samples of each class
        output = clip_logits + cache_logits * self.alpha
        return output, embedding_norm