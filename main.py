import glob
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from to_pianoroll import midi_to_piano_roll, piano_roll_to_pretty_midi

# Data augmentation functions (pitch and time shifts)
def random_pitch_shift(pr, max_transposition=5):
    shift = np.random.randint(-max_transposition, max_transposition + 1)
    if shift == 0:
        return pr
    pr_shifted = np.roll(pr, shift, axis=0)
    if shift > 0:
        pr_shifted[:shift, :] = 0
    else:
        pr_shifted[shift:, :] = 0
    return pr_shifted


def random_time_shift(pr, max_shift=10):
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift == 0:
        return pr
    pr_shifted = np.roll(pr, shift, axis=1)
    if shift > 0:
        pr_shifted[:, :shift] = 0
    else:
        pr_shifted[:, shift:] = 0
    return pr_shifted

# Load MIDI files and convert them to piano rolls
midi_files = glob.glob('./dataset/*.mid')
data = []
fixed_length = 500  # fixed number of time steps
fs =  20        # sampling frequency

for f in tqdm(midi_files):
    piano_roll = midi_to_piano_roll(f, fs=fs)
    if piano_roll is not None:
        # Data augmentation: apply pitch and time shifts with 50% probability
        if np.random.rand() < 0.5:
            piano_roll = random_pitch_shift(piano_roll, max_transposition=5)
        if np.random.rand() < 0.5:
            piano_roll = random_time_shift(piano_roll, max_shift=10)
        # Truncate or pad piano roll to fixed_length along time axis
        pr = piano_roll[:, :fixed_length]
        if pr.shape[1] < fixed_length:
            pr = np.pad(pr, ((0, 0), (0, fixed_length - pr.shape[1])), mode='constant')
        # Add channel dimension: shape becomes (1, 128, fixed_length)
        pr = np.expand_dims(pr, axis=0)
        data.append(pr)

data = np.array(data)
print("Data shape:", data.shape)  # Expected shape: (num_samples, 1, 128, 500)

# Create PyTorch dataset and dataloader
data_tensor = torch.tensor(data, dtype=torch.float32)
dataset = TensorDataset(data_tensor)
batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the convolutional VAE model
class ConvVAE(nn.Module):
    def __init__(self, latent_dim=50):
        super(ConvVAE, self).__init__()
        # Encoder: input shape (batch, 1, 128, 500)
        self.enc_conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1)   # -> (16, 64, 250)
        self.enc_conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)  # -> (32, 32, 125)
        self.enc_conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # -> (64, 16, 62)
        self.enc_conv4 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # -> (128, 8, 31)
        
        self.flatten_size = 128 * 8 * 31  # = 31,744
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        # Decoder: mirror the encoder architecture
        self.fc_decode = nn.Linear(latent_dim, self.flatten_size)
        self.dec_conv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # -> (64, 16, 62)
        # For symmetry, add output_padding for width to get 125
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=(0,1))  # -> (32, 32, 125)
        self.dec_conv2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)  # -> (16, 64, 250)
        self.dec_conv1 = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)   # -> (1, 128, 500)

    def encode(self, x):
        h = F.relu(self.enc_conv1(x))
        h = F.relu(self.enc_conv2(h))
        h = F.relu(self.enc_conv3(h))
        h = F.relu(self.enc_conv4(h))
        h = h.view(x.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc_decode(z))
        h = h.view(z.size(0), 128, 8, 31)
        h = F.relu(self.dec_conv4(h))
        h = F.relu(self.dec_conv3(h))
        h = F.relu(self.dec_conv2(h))
        h = torch.sigmoid(self.dec_conv1(h))
        return h

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# Loss function for the VAE: reconstruction + KL divergence
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Setup device, model, optimizer, and training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 50
model = ConvVAE(latent_dim=latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 50

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
    for batch in pbar:
        x = batch[0].to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(x)
        loss = loss_function(recon, x, mu, logvar)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
    avg_loss = train_loss / len(dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), "model_conv_vae.bin")
print("Model saved as model_conv_vae.bin")

# Generate a sample and convert to MIDI
model.eval()
with torch.no_grad():
    z = torch.randn(1, latent_dim).to(device)
    generated = model.decode(z).cpu().numpy()  # shape: (1, 1, 128, 500)
    generated = generated[0, 0]  # remove batch and channel dimensions, shape: (128, 500)
    # Binarize the piano roll using a threshold (0.5)
    generated_binary = (generated > 0.5).astype(np.float32)
    # Convert the binary piano roll to a PrettyMIDI object
    pm = piano_roll_to_pretty_midi(generated_binary, fs=fs, program=0)
    output_midi_file = "generated_conv.mid"
    pm.write(output_midi_file)
    print("Generated MIDI file saved as:", output_midi_file)
