import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader, TensorDataset


# # Custom Dataset
# class ASPADEDataset(Dataset):
#     def __init__(self, inputs, targets_estimates, targets_sparsity):
#         self.inputs = torch.tensor(inputs, dtype=torch.float32)
#         self.targets_estimates = torch.tensor(targets_estimates, dtype=torch.float32)
#         self.targets_sparsity = torch.tensor(targets_sparsity, dtype=torch.float32).squeeze()

#     def __len__(self):
#         return len(self.inputs)

#     def __getitem__(self, idx):
#         return self.inputs[idx], self.targets_estimates[idx], self.targets_sparsity[idx]

# class ASPADEModel(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(ASPADEModel, self).__init__()
        
#         # Common feature extractor
#         self.fc1 = nn.Linear(input_size, 512)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.fc2 = nn.Linear(512, 256)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.fc3 = nn.Linear(256, 128)

#         # Complex estimate output branch
#         self.fc_est1 = nn.Linear(128, 64)
#         self.fc_est2 = nn.Linear(64, output_size)

#         # Sparsity output branch
#         self.fc_spar1 = nn.Linear(128, 64)
#         self.fc_spar2 = nn.Linear(64, 1)

#         self.activation = nn.LeakyReLU(0.1)
#         self.dropout = nn.Dropout(0.2)

#     def forward(self, x):
#         # Shared layers
#         x = self.activation(self.bn1(self.fc1(x)))
#         x = self.dropout(x)
#         x = self.activation(self.bn2(self.fc2(x)))
#         x = self.dropout(x)
#         x = self.activation(self.fc3(x))

#         # Complex estimate output
#         estimate = self.activation(self.fc_est1(x))
#         estimate = self.fc_est2(estimate)

#         # Sparsity output
#         sparsity = self.activation(self.fc_spar1(x))
#         sparsity = self.fc_spar2(sparsity).squeeze()  # Scalar output

#         return estimate, sparsity
    


class ASPADEDataset(Dataset):
    def __init__(self, inputs, targets_estimates, targets_sparsity):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets_estimates = torch.tensor(targets_estimates, dtype=torch.float32)
        self.targets_sparsity = torch.tensor(targets_sparsity, dtype=torch.long).squeeze()  # Use long tensor for integers

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets_estimates[idx], self.targets_sparsity[idx]

class ASPADEModel(nn.Module):
    def __init__(self, input_size, output_size, max_sparsity=1000):
        super(ASPADEModel, self).__init__()

        # Shared feature extractor
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)

        # Estimate branch (complex values output)
        self.fc_est1 = nn.Linear(128, 64)
        self.fc_est2 = nn.Linear(64, output_size)

        # Sparsity classification branch
        self.fc_spar1 = nn.Linear(128, 64)
        self.fc_spar2 = nn.Linear(64, max_sparsity + 1)  # Classification over [0, max_sparsity]

        self.activation = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Shared layers
        x = self.activation(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.activation(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.activation(self.fc3(x))

        # Complex estimate output
        estimate = self.activation(self.fc_est1(x))
        estimate = self.fc_est2(estimate)

        # Sparsity classification output (logits)
        sparsity_logits = self.activation(self.fc_spar1(x))
        sparsity_logits = self.fc_spar2(sparsity_logits)

        return estimate, sparsity_logits