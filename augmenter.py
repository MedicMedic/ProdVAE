import numpy as np

# Load your preprocessed token sequences.
# Assume each entry in the .npy file is a list of integer tokens.
data = np.load("dataset_tokens.npy", allow_pickle=True).tolist()

# Define augmentation functions.
def augment_pitch_shift(tokens, semitone_shift):
    """Shift all note tokens by a fixed number of semitones.
    
    Note:
      - Note-on tokens are in [0, 127].
      - Note-off tokens are in [128, 255].
    """
    augmented = []
    for token in tokens:
        if 0 <= token < 128:  # note-on token
            new_token = token + semitone_shift
            augmented.append(max(0, min(127, new_token)))
        elif 128 <= token < 256:  # note-off token
            new_token = token + semitone_shift
            augmented.append(max(128, min(255, new_token)))
        else:
            # Other tokens (time-shift, velocity) remain unchanged.
            augmented.append(token)
    return augmented

def augment_mirror(tokens, center_pitch=60):
    """Mirror note tokens around a central pitch.
    
    Each note is reflected: new_pitch = 2*center - original_pitch.
    """
    augmented = []
    for token in tokens:
        if 0 <= token < 128:  # note-on token
            new_pitch = 2 * center_pitch - token
            augmented.append(max(0, min(127, new_pitch)))
        elif 128 <= token < 256:  # note-off token
            orig_pitch = token - 128
            new_pitch = 2 * center_pitch - orig_pitch
            augmented.append(128 + max(0, min(127, new_pitch)))
        else:
            augmented.append(token)
    return augmented

def augment_time_stretch(tokens, stretch_factor, tick_start=256, tick_dim=100):
    """Stretch or compress time-shift tokens by a given factor.
    
    Tokens in the range [tick_start, tick_start+tick_dim) represent time shifts.
    """
    augmented = []
    for token in tokens:
        if tick_start <= token < tick_start + tick_dim:
            tick_value = token - tick_start + 1  # quantized tick value (1 to tick_dim)
            new_tick = int(round(tick_value * stretch_factor))
            new_tick = max(1, min(tick_dim, new_tick))
            augmented.append(tick_start + new_tick - 1)
        else:
            augmented.append(token)
    return augmented

def augment_dynamics(tokens, velocity_factor=1.1, velocity_start=256+100, velocity_dim=32):
    """Scale velocity tokens by a factor."""
    augmented = []
    for token in tokens:
        if velocity_start <= token < velocity_start + velocity_dim:
            quant_val = token - velocity_start
            new_quant = int(round(quant_val * velocity_factor))
            new_quant = max(0, min(velocity_dim - 1, new_quant))
            augmented.append(velocity_start + new_quant)
        else:
            augmented.append(token)
    return augmented

# Create an augmented dataset by generating additional versions for each sequence.
augmented_data = []
for tokens in data:
    # Add the original token sequence.
    augmented_data.append(tokens)
    # Variant 1: Pitch shift by +2 semitones.
    variant1 = augment_pitch_shift(tokens, 2)
    augmented_data.append(variant1)
    # Variant 2: Mirror the melody around a center pitch (e.g., 60).
    variant2 = augment_mirror(tokens, center_pitch=60)
    augmented_data.append(variant2)
    # Variant 3: Time-stretch the sequence (e.g., slow down by 10%).
    variant3 = augment_time_stretch(tokens, stretch_factor=1.1)
    augmented_data.append(variant3)
    # Variant 4: Dynamics augmentation.
    variant4 = augment_dynamics(tokens, velocity_factor=1.1)
    augmented_data.append(variant4)

# Now, augmented_data should contain roughly 5 times as many sequences as your original dataset.
np.save("dataset_tokens_augmented.npy", np.array(augmented_data, dtype=object))
print("Augmented dataset saved with", len(augmented_data), "sequences.")
