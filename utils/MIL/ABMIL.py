import torch


class ABMIL(torch.nn.Module):
    def __init__(self, L):
        super(ABMIL, self).__init__()

        # Attention MIL embedding from Ilse et al. (2018) for MIL.
        # Class based on Julio Silva's MILAggregation class in PyTorch
        self.L = L
        self.D = 384
        self.K = 1
        self.attention_V = torch.nn.Sequential(
            torch.nn.Linear(self.L, self.D),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=0.25)
        )
        self.attention_U = torch.nn.Sequential(
            torch.nn.Linear(self.L, self.D),
            torch.nn.Sigmoid(),
            torch.nn.Dropout(p=0.25)
        )

        self.attention_weights = torch.nn.Linear(self.D, self.K)

    def forward(self, features):
        A_V = self.attention_V(features)  # Attention
        A_U = self.attention_U(features)  # Gate
        w = torch.softmax(self.attention_weights(A_V * A_U), dim=0)
        features = torch.transpose(features, 1, 0)
        embedding = torch.squeeze(torch.mm(features, w))  # MIL Attention
        return embedding, w