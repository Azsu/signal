from dataclasses import dataclass, field
from typing import Callable, List

import librosa
import numpy as np


@dataclass
class Signal:
    """A class for representing and manipulating audio signals."""

    frequency: float
    duration: float
    timestamps: List[float]
    signal_generator: Callable[[float, float, int], np.ndarray] = field(repr=False)
    sample_rate: int = 44100
    data: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.data = self.signal_generator(
            self.frequency, self.duration, self.sample_rate
        )

    def amplitude_envelope(
        self, envelope_function: Callable[[np.ndarray], np.ndarray]
    ) -> "Signal":
        """Apply an amplitude envelope to the signal.

        Args:
            envelope_function: A function that takes an array of time points and returns an array of amplitude values.

        Returns:
            The modified Signal instance.
        """
        self.data = self.data * envelope_function(
            np.linspace(0, len(self.data) / self.sample_rate, len(self.data), False)
        )
        return self

    def apply_filter(
        self, filter_function: Callable[[np.ndarray], np.ndarray]
    ) -> "Signal":
        """Apply a filter function to the signal.

        Args:
            filter_function: A function that takes an array of samples and returns an array of filtered samples.

        Returns:
            The modified Signal instance.
        """
        self.data = filter_function(self.data)
        return self

    def time_stretch(self, rate: float) -> "Signal":
        """Apply a time-stretch to the signal.

        Args:
            rate: The time-stretch factor. Values greater than 1 speed up the signal, values between 0 and 1 slow it down.

        Returns:
            The modified Signal instance.
        """
        self.data = librosa.effects.time_stretch(self.data, rate)
        return self

    @staticmethod
    def generate_timestamps(
        pattern: List[int],
        segment_duration: float,
        repetitions: int = 1,
        spacing_function: Callable[[int], float] = None,
    ) -> List[float]:
        """Generate timestamps based on a pattern, segment duration, and optional spacing function.

        Args:
            pattern: A list of relative note positions.
            segment_duration: The duration of each pattern segment in seconds.
            repetitions: The number of times the pattern should repeat.
            spacing_function: A function that takes the note index and returns a scaling factor for the note duration.

        Returns:
            A list of timestamps.
        """
        timestamps = []

        for i in range(repetitions):
            pattern_start = i * segment_duration
            for idx, note in enumerate(pattern):
                note_duration = segment_duration / len(pattern)
                if spacing_function:
                    note_duration *= spacing_function(idx)
                timestamps.append(pattern_start + note * note_duration)

        return timestamps


@dataclass
class Song:
    """A class for representing and manipulating songs composed of multiple signals."""

    signals: List[Signal]
    sample_rate: int = 44100

    def mix_signals(self) -> np.ndarray:
        """Mix the signals in the song based on their timestamps.

        Returns:
            An array of mixed audio samples.
        """
        # Calculate the total song length
        song_length = max(
            [signal.timestamps[-1] + signal.duration for signal in self.signals]
        )

        # Generate an empty data stream for the duration of the song
        song_data = np.zeros(int(song_length * self.sample_rate))

        for signal in self.signals:
            for timestamp in signal.timestamps:
                start_idx = int(timestamp * self.sample_rate)
                end_idx = start_idx + len(signal.data)

                if end_idx <= len(song_data):
                    song_data[start_idx:end_idx] += signal.data

        return song_data

import numpy as np

def sine_wave(frequency, duration, sample_rate):
    time_points = np.linspace(0, duration, int(duration * sample_rate), False)
    return np.sin(frequency * time_points * 2 * np.pi)

def kick_drum(frequency, duration, sample_rate):
    time_points = np.linspace(0, duration, int(duration * sample_rate), False)
    envelope = np.exp(-3 * time_points)
    return sine_wave(frequency, duration, sample_rate) * envelope

def snare_drum(frequency, duration, sample_rate):
    time_points = np.linspace(0, duration, int(duration * sample_rate), False)
    envelope = np.exp(-5 * time_points)
    white_noise = np.random.normal(0, 1, int(duration * sample_rate))
    return (sine_wave(frequency, duration, sample_rate) + white_noise) * envelope

# Define the drum sounds
kick = Signal(60, 0.5, [], kick_drum)
snare = Signal(200, 0.3, [], snare_drum)

# Define the tribal drum pattern
pattern = [0, 2, 3, 6]
segment_duration = 8
repetitions = 4

timestamps_kick = Signal.generate_timestamps(pattern, segment_duration, repetitions)
timestamps_snare = Signal.generate_timestamps(pattern[1:], segment_duration, repetitions)

kick.timestamps = timestamps_kick
snare.timestamps = timestamps_snare

# Create a song with the drum sounds
tribal_drum_song = Song([kick, snare])

# Mix the signals to create the tribal drum song
song_data = tribal_drum_song.mix_signals()

# Normalize the audio data
song_data = song_data / np.max(np.abs(song_data))

# Save the tribal drum song as a WAV file
from scipy.io.wavfile import write

write("tribal_drum_song.wav", 44100, (song_data * 32767).astype(np.int16))

@dataclass
class EnvelopeSignal(Signal):
    waveform: Callable[[float, float, int], np.ndarray]
    attack: float
    decay: float

    def __post_init__(self):
        time_points = np.linspace(0, self.duration, int(self.duration * self.sample_rate), False)

        # Create the waveform
        wave = self.waveform(self.frequency, self.duration, self.sample_rate)

        # Create the attack-decay envelope
        envelope = np.concatenate([
            np.linspace(0, 1, int(self.attack * self.sample_rate), False),
            np.exp(-self.decay * (time_points[self.attack * self.sample_rate:] - self.attack))
        ])

        # Combine the waveform and envelope
        self.data = wave * envelope

# Update Drum and Guitar classes to extend EnvelopeSignal
@dataclass
class Drum(EnvelopeSignal):
    noise_level: float

    def __post_init__(self):
        super().__post_init__()

        # Add the noise component
        noise = np.random.normal(0, self.noise_level, int(self.duration * self.sample_rate))
        self.data += noise

@dataclass
class Guitar(EnvelopeSignal):
    def __init__(self, freq=440):
        self.freq = freq

