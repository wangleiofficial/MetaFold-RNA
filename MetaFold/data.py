import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class RNADataset(Dataset):
    def __init__(self, pickle_path):
        print(f"Loading data from {pickle_path}...")
        try:
            with open(pickle_path, 'rb') as f:
                self.data = pickle.load(f)
            print(f"Successfully loaded {len(self.data)} samples.")
        except FileNotFoundError:
            print(f"Error: Pickle file not found at {pickle_path}")
            self.data = []

        self.base2idx = {'A': 0, 'U': 1, 'G': 2, 'C': 3}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        features = entry["predict_contact"]
        labels = entry["contact"]
        sequence = entry["sequence"]

        seq_encoded = torch.tensor([self.base2idx.get(base, 0) for base in sequence], dtype=torch.long)

        features_tensor = torch.tensor(features, dtype=torch.float)
        labels_tensor = torch.tensor(labels, dtype=torch.float)

        return {
            "features": features_tensor,
            "labels": labels_tensor,
            "name": entry["name"],
            "sequence": sequence,
            "seq_encoded": seq_encoded
        }
    
# define dataset without labels, input is fasta file
class RNADatasetNoLabels(Dataset):
    def __init__(self, fasta_path):
        self.sequences = self.load_fasta(fasta_path)
        self.base2idx = {'A': 0, 'U': 1, 'G': 2, 'C': 3}

    def load_fasta(self, fasta_path):
        sequences = []
        with open(fasta_path, 'r') as f:
            for line in f:
                if line.startswith(">"):
                    continue
                sequences.append(line.strip())
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        seq_encoded = torch.tensor([self.base2idx.get(base, 0) for base in sequence], dtype=torch.long)
        return {
            "sequence": sequence,
            "seq_encoded": seq_encoded,
            "name": f"seq_{idx}"
        }