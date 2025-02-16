import numpy as np
from sklearn.model_selection import train_test_split

# Load your augmented token sequences from your .npy file
data = np.load("dataset_tokens.npy", allow_pickle=True).tolist()

# First split off 20% for validation+test
train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)

# Then split the remaining 20% equally into validation and test sets
valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

print("Training samples:", len(train_data))
print("Validation samples:", len(valid_data))
print("Test samples:", len(test_data))

# Optionally, save these splits:
np.save("train_data.npy", np.array(train_data, dtype=object))
np.save("valid_data.npy", np.array(valid_data, dtype=object))
np.save("test_data.npy", np.array(test_data, dtype=object))
