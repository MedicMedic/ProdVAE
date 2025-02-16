import torch
from torch.utils.data import Dataset

class MidiDataset(Dataset):
    def __init__(self, token_sequences):
        """
        Parameters
        ----------
        token_sequences : list
            A list of token sequences, where each token sequence is a list of integers.
        """
        self.token_sequences = token_sequences

    def __len__(self):
        return len(self.token_sequences)

    def __getitem__(self, idx):
        # Return the token sequence as a PyTorch tensor.
        # If your sequences have variable length, they will be padded later in a collate_fn.
        return torch.tensor(self.token_sequences[idx], dtype=torch.long)
    
def pad_collate(batch):
    # batch is a list of tensors (each one is a token sequence)
    lengths = [len(seq) for seq in batch]
    max_len = max(lengths)
    # Pad each sequence with zeros (or a designated PAD token, if available)
    padded_batch = [torch.cat([seq, torch.zeros(max_len - len(seq), dtype=torch.long)]) for seq in batch]
    return torch.stack(padded_batch), torch.tensor(lengths)

