import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
import os

def audio_reg(file_path: str = None, duration: int = 4, sr: int = 16000, n_mfcc: int = 13) -> np.ndarray:
    """
    Records or loads an audio sample and returns the MFCC-based feature vector.

    Parameters:
        file_path (str): Optional path to a .wav file. If None, records live audio.
        duration (int): Duration of recording in seconds (if recording).
        sr (int): Sample rate for audio.
        n_mfcc (int): Number of MFCCs to extract.

    Returns:
        np.ndarray: Feature vector (shape: [n_mfcc * 3,])
    """
    if file_path is None:
        print(f"🎙️ Recording for {duration} seconds...")
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        file_path = "temp_recording.wav"
        sf.write(file_path, audio, sr)
        print(f"✅ Audio recorded to {file_path}")

    try:
        y, sr = librosa.load(file_path, sr=sr, mono=True)
        y, _ = librosa.effects.trim(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        delta1 = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        combined = np.vstack([mfcc, delta1, delta2])
        feature_vector = np.mean(combined.T, axis=0)

        if file_path == "temp_recording.wav":
            os.remove(file_path)

        return feature_vector

    except Exception as e:
        print(f"❌ Failed to extract features: {e}")
        return np.array([])

# Example usage:
# features = audio_reg()  # For live mic input
# features = audio_reg("sample.wav")  # For saved audio file
