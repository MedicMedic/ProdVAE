import torch
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
from dataset import MidiDataset, pad_collate
from midi_vae import MidiVAE, vae_loss

# Load your training data and convert it to a list (if necessary)
train_data = np.load("train_data.npy", allow_pickle=True).tolist()
train_dataset = MidiDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=pad_collate)

# Load validation data
valid_data = np.load("valid_data.npy", allow_pickle=True).tolist()
valid_dataset = MidiDataset(valid_data)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=pad_collate)

device = torch.device("cpu")
vocab_size = 256 + 100 + 32  # Based on your tokenization
model = MidiVAE(vocab_size=vocab_size, embed_size=128, hidden_size=256, latent_size=64).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 1000

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch, lengths in train_loader:
        batch, lengths = batch.to(device), lengths.to(device)
        optimizer.zero_grad()
        logits, mu, logvar = model(batch, lengths)
        loss = vae_loss(logits, batch, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}")
    
    # Optionally, evaluate on the validation set:
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch, lengths in valid_loader:
            batch, lengths = batch.to(device), lengths.to(device)
            logits, mu, logvar = model(batch, lengths)
            loss = vae_loss(logits, batch, mu, logvar)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(valid_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")

# Save model state dict as a .bin file.
torch.save(model.state_dict(), "music_vae.bin")
print("Model saved as music_vae.bin")
