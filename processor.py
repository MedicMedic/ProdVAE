import pretty_midi as pyd
import numpy as np
from abc import ABC, abstractmethod

# An abstract representation processor: defines compress/expand helpers,
# and the abstract encode/decode methods.
class ReprProcessor(ABC):
    def __init__(self, min_step=1):
        self.min_step = min_step

    def _compress(self, note_seq):
        # If min_step > 1, reduce the time resolution by dividing start and end times.
        return [
            pyd.Note(
                pitch=note.pitch,
                velocity=note.velocity,
                start=int(note.start / self.min_step),
                end=int(note.end / self.min_step)
            )
            for note in note_seq
        ]

    def _expand(self, note_seq):
        # Inverse of compress: multiply the note times back by min_step.
        return [
            pyd.Note(
                pitch=note.pitch,
                velocity=note.velocity,
                start=int(note.start * self.min_step),
                end=int(note.end * self.min_step)
            )
            for note in note_seq
        ]

    @abstractmethod
    def encode(self, note_seq):
        """Convert a note sequence into a sequence of integer tokens."""
        pass

    @abstractmethod
    def decode(self, repr_seq):
        """Convert a sequence of integer tokens back into a note sequence."""
        pass

# MidiEventProcessor that implements a token-based representation.
class MidiEventProcessor(ReprProcessor):
    def __init__(self, min_step=1, tick_dim=100, velocity_dim=32):
        super(MidiEventProcessor, self).__init__(min_step)
        self.tick_dim = tick_dim
        self.velocity_dim = velocity_dim
        if self.velocity_dim > 128:
            raise ValueError("velocity_dim cannot be larger than 128")
        # Vocabulary is divided into four regions:
        # 0-127: note_on, 128-255: note_off,
        # 256 to 256+tick_dim-1: time-shift, and
        # 256+tick_dim to max_vocab-1: velocity events.
        self.max_vocab = 256 + self.tick_dim + self.velocity_dim
        self.start_index = {
            "note_on": 0,
            "note_off": 128,
            "time_shift": 256,
            "velocity": 256 + self.tick_dim,
        }
    def encode(self, note_seq):
        """
        Encode a list of PrettyMIDI Note objects into a token sequence.
        Token regions:
        - 0–127: note_on (pitch)
        - 128–255: note_off (pitch offset by 128)
        - 256–(256+tick_dim-1): time-shift events (quantized time differences)
        - (256+tick_dim)–max_vocab-1: velocity events (quantized velocities)
        """
        if note_seq is None:
            return []
        # Optionally compress note times
        if self.min_step > 1:
            note_seq = self._compress(note_seq)
        events = []
        meta_events = []
        # Create meta events: note-on and note-off for each note.
        for note in note_seq:
            meta_events.append({
                "name": "note_on",
                "time": note.start,
                "pitch": note.pitch,
                "vel": note.velocity
            })
            meta_events.append({
                "name": "note_off",
                "time": note.end,
                "pitch": note.pitch,
                "vel": None
            })
        # Sort meta events by pitch (if desired) and then by time.
        meta_events.sort(key=lambda x: x["pitch"])
        meta_events.sort(key=lambda x: x["time"])
        
        current_time = 0.0
        current_velocity = 0  # Initialize the current velocity variable.
        for me in meta_events:
            # Compute the time difference since the last event (quantized with 0.01s per tick)
            time_diff = me["time"] - current_time
            if time_diff < 0:
                time_diff = 0
            ticks = int(round(time_diff * 100))
            while ticks >= self.tick_dim:
                events.append(self.start_index["time_shift"] + self.tick_dim - 1)
                ticks -= self.tick_dim
            if ticks > 0:
                events.append(self.start_index["time_shift"] + ticks - 1)
            current_time = me["time"]
            # Add a velocity event if the note has a velocity and it differs from the current one.
            if me["vel"] is not None:
                if current_velocity != me["vel"]:
                    current_velocity = me["vel"]
                    vel_index = int(round(me["vel"] * self.velocity_dim / 128))
                    vel_index = min(vel_index, self.velocity_dim - 1)  # Clamp to valid range.
                    events.append(self.start_index["velocity"] + vel_index)
            # Append the note event token.
            events.append(self.start_index[me["name"]] + me["pitch"])
        return events

    def decode(self, repr_seq):
        """
        Decode a token sequence back into a list of PrettyMIDI Note objects.
        This method reverses the encoding process by keeping track of the
        current time and velocity while reading tokens.
        """
        if repr_seq is None:
            return []
        current_time = 0.0
        current_velocity = 0
        meta_events = []
        notes = []
        # Iterate over tokens and reconstruct meta events.
        for token in repr_seq:
            if self.start_index["note_on"] <= token < self.start_index["note_off"]:
                meta_events.append({
                    "name": "note_on",
                    "time": current_time,
                    "pitch": token - self.start_index["note_on"],
                    "vel": current_velocity
                })
            elif self.start_index["note_off"] <= token < self.start_index["time_shift"]:
                meta_events.append({
                    "name": "note_off",
                    "time": current_time,
                    "pitch": token - self.start_index["note_off"],
                    "vel": current_velocity
                })
            elif self.start_index["time_shift"] <= token < self.start_index["velocity"]:
                # Increase time by the quantized tick amount.
                tick_value = token - self.start_index["time_shift"] + 1
                current_time += tick_value * 0.01
            elif self.start_index["velocity"] <= token < self.max_vocab:
                # Update current velocity based on the token.
                quant_vel = token - self.start_index["velocity"]
                current_velocity = int(round(quant_vel * 128 / self.velocity_dim))
        # Pair note_on and note_off events to create complete notes.
        note_on_dict = {}
        for ev in meta_events:
            if ev["name"] == "note_on":
                note_on_dict[ev["pitch"]] = ev
            elif ev["name"] == "note_off":
                if ev["pitch"] in note_on_dict:
                    on_ev = note_on_dict[ev["pitch"]]
                    # Avoid zero-duration notes.
                    if on_ev["time"] != ev["time"]:
                        notes.append(pyd.Note(
                            pitch=on_ev["pitch"],
                            velocity=on_ev["vel"],
                            start=on_ev["time"],
                            end=ev["time"]
                        ))
        # Sort the resulting notes by start time.
        notes.sort(key=lambda n: n.start)
        if self.min_step > 1:
            notes = self._expand(notes)
        return notes

# Example usage:
if __name__ == "__main__":
    # Assume you have a MIDI file and want to process its first instrument track.
    midi_path = "example.mid"  # Replace with your MIDI file path.
    midi_data = pyd.PrettyMIDI(midi_path)
    # For demonstration, pick the first instrument's notes.
    if midi_data.instruments:
        melody_notes = midi_data.instruments[0].notes
        processor = MidiEventProcessor(min_step=1, tick_dim=100, velocity_dim=32)
        tokens = processor.encode(melody_notes)
        print("Encoded tokens:", tokens)
        # Optionally, decode tokens back to note objects.
        decoded_notes = processor.decode(tokens)
        print("Decoded notes:")
        for n in decoded_notes:
            print(f"Pitch: {n.pitch}, Velocity: {n.velocity}, Start: {n.start}, End: {n.end}")
    else:
        print("No instruments found in the MIDI file.")
