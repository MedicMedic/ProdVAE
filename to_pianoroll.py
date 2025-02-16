import pretty_midi
import numpy as np

def midi_to_piano_roll(midi_file, fs=100):
    """
    Converts a MIDI file to a piano roll.
    Args:
        midi_file (str): Path to the MIDI file.
        fs (int): Sampling frequency (frames per second).
    Returns:
        np.ndarray or None: Piano roll representation, or None if an error occurs.
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file)

        if not midi_data.instruments:
            print(f"Warning: {midi_file} has no instruments.")
            return None

        # Get the piano roll for all instruments combined
        piano_roll = midi_data.get_piano_roll(fs=fs)

        if piano_roll is None or piano_roll.shape[1] == 0:
            print(f"Warning: {midi_file} has no valid note data.")
            return None

        # Optional: Binarize the piano roll (0 if no note, 1 if note is played)
        piano_roll = (piano_roll > 0).astype(np.float32)

        return piano_roll

    except Exception as e:
        print(f"Error processing {midi_file}: {e}")
        return None


def piano_roll_to_pretty_midi(piano_roll, fs=50, program=0):
    """
    Convert a piano roll array into a PrettyMIDI object.
    
    Args:
        piano_roll (np.ndarray): 2D array with shape (128, num_frames)
        fs (int): Frames per second used for the piano roll.
        program (int): MIDI program number (0 for Acoustic Grand Piano).
    
    Returns:
        pretty_midi.PrettyMIDI: A PrettyMIDI object with one instrument.
    """
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)
    
    # Check for empty piano roll
    if piano_roll is None or piano_roll.shape[1] == 0:
        print("Warning: Empty piano roll, skipping conversion.")
        return pm

    # Ensure pitch values stay in the MIDI range
    num_pitches = piano_roll.shape[0]
    if num_pitches != 128:
        print(f"Warning: Unexpected piano roll size ({num_pitches} pitches), resizing to 128.")
        piano_roll = np.pad(piano_roll, ((0, max(0, 128 - num_pitches)), (0, 0)), mode='constant')[:128, :]

    # Iterate over all 128 MIDI pitches.
    for pitch in range(128):  # MIDI range is fixed (0-127)
        # Find all frames where the note is active.
        indices = np.where(piano_roll[pitch] > 0)[0]
        if len(indices) == 0:
            continue

        # Group contiguous indices (notes that are held over multiple frames)
        note_segments = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
        for segment in note_segments:
            start_time = segment[0] / fs
            end_time = (segment[-1] + 1) / fs

            if start_time >= end_time:
                continue  # Skip invalid notes

            note = pretty_midi.Note(velocity=100, pitch=pitch,
                                    start=start_time, end=end_time)
            instrument.notes.append(note)
    
    if not instrument.notes:
        print("Warning: No notes detected after conversion.")

    pm.instruments.append(instrument)
    return pm
