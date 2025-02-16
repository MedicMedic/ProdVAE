import torch
import numpy as np
from midi_vae import VAE  # Ensure this is the same VAE class you used for training
from to_pianoroll import piano_roll_to_pretty_midi  # Convert piano roll to MIDI

# Define model parameters (must match training settings)
input_dim = 128 * 500  # Adjust based on your training
hidden_dim = 512
latent_dim = 50  # Same latent space size used during training

# Initialize the model
model = VAE(input_dim, hidden_dim, latent_dim)

# Load the trained weights
model.load_state_dict(torch.load("model.bin"))
model.eval()  # Set model to evaluation mode

# Generate a new random latent vector
z = torch.randn(1, latent_dim)  # Sample a random point in latent space

# Decode to get a piano roll
with torch.no_grad():
    generated_piano_roll = model.decode(z).view(128, -1).cpu().numpy()  # Reshape to (128, fixed_length)

# Convert the piano roll to MIDI
fs = 50  # Frames per second (same as training)
pm = piano_roll_to_pretty_midi(generated_piano_roll, fs=fs, program=0)

# Save the MIDI file
output_midi_file = "generated_new.mid"
pm.write(output_midi_file)
print("Generated MIDI file saved as:", output_midi_file)
