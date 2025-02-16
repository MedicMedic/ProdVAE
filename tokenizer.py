import os
import numpy as np
import pretty_midi as pyd
from processor import MidiEventProcessor  # our custom processor

def preprocess_midi_file(file_path):
    """Process a single MIDI file into a token sequence using PrettyMIDI."""
    midi_data = pyd.PrettyMIDI(file_path)
    # For a simple melody, process only the first instrument.
    if not midi_data.instruments:
        return []
    notes = midi_data.instruments[0].notes
    # Round times and sort the notes
    for note in notes:
        note.start = round(note.start, 2)
        note.end = round(note.end, 2)
    notes.sort(key=lambda n: n.start)
    
    processor = MidiEventProcessor(min_step=1, tick_dim=100, velocity_dim=32)
    tokens = processor.encode(notes)
    return tokens

def process_dataset(midi_root, output_path):
    """
    Process all MIDI files in a single folder and save their token sequences in one .npy file.
    
    Parameters
    ----------
    midi_root : str
        Path to the folder containing MIDI files.
    output_path : str
        Filename for saving the processed tokens (e.g., "pop909_tokens.npy").
    """
    all_tokens = []
    # List only MIDI files (assuming .mid extension)
    midi_files = [f for f in os.listdir(midi_root) if f.lower().endswith('.mid')]
    
    for file in midi_files:
        file_path = os.path.join(midi_root, file)
        print(f"Processing {file}...", end=" ")
        try:
            tokens = preprocess_midi_file(file_path)
            all_tokens.append(tokens)
            print(f"done (length={len(tokens)})")
        except Exception as e:
            print(f"Error: {e}")
    
    # Save as a numpy object array (to support variable-length sequences)
    np.save(output_path, np.array(all_tokens, dtype=object))
    print(f"Saved token sequences for {len(all_tokens)} MIDI files to {output_path}")

if __name__ == "__main__":
    MIDI_FOLDER = "./dataset"         # Replace with your folder path
    OUTPUT_FILE = "dataset_tokens.npy"
    process_dataset(MIDI_FOLDER, OUTPUT_FILE)
