import pyaudio
import numpy as np
import torch
from scipy.signal import butter, sosfilt
import speechbrain as sb
from speechbrain.pretrained import EncoderClassifier

# Audio Parameters
samplerate = 44100  # Required for SpeechBrain
channels = 1
blocksize = 1024 * 16  # Buffer size
format = pyaudio.paInt16
trigger_threshold = 30.0  # Adjust based on mic sensitivity

# Load SpeechBrain Model for Speaker Embeddings
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp_spkrec")

# Design a Low-Pass Filter (3kHz cutoff for noise reduction)
cutoff_freq = 3000


# State Variables
triggered = False  # Start listening only when triggered
buffer = []  # Store samples for processing


    #%% Function for design of filter
def design_filter(lowfreq, highfreq, fs, order=3):
    nyq = 0.5*fs
    low = lowfreq/nyq
    high = highfreq/nyq
    sos = butter(order, [low,high], btype='band',output='sos')
    return sos

sos = design_filter(3000, 15000, 48000, 3) #change the lower and higher freqcies according to choice


def process_audio():
    """Convert buffered audio into a SpeechBrain embedding."""
    global buffer
    if not buffer:
        return

    # Convert to Torch Tensor (Shape: [1, Samples])
    audio_tensor = torch.tensor(np.concatenate(buffer), dtype=torch.int16).unsqueeze(0)

    # Get Speaker Embedding
    embedding = classifier.encode_batch(audio_tensor).squeeze().detach().numpy()

    print("?? Extracted SpeechBrain Embedding:", embedding[:5])  # Print first 5 values as a preview
    buffer = []  # Clear buffer


def audio_callback(in_data, frame_count, time_info, status):
    global triggered, buffer

    # Convert byte data to NumPy float32 array
    audio_data = np.frombuffer(in_data, dtype=np.int16)

    # Apply SOS filter to incoming audio
    filtered_audio = sosfilt(sos, audio_data)

    # Compute energy level to check for trigger activation
    energy = np.mean(np.abs(filtered_audio))

    

    print(energy)

    if not triggered and energy > trigger_threshold:
        print("?? Triggered! Start recording...")
        triggered = True
    elif triggered and energy < trigger_threshold:
        print("?? Done! Recording Stopped...")
        triggered = True

    if triggered:
        buffer.append(filtered_audio)  # Store audio chunks
        if len(buffer) * blocksize >= samplerate * 2:  # 2 seconds of speech
            triggered = False  # Reset trigger
            print("?? Processing recorded speech...")
            process_audio()

    return (in_data, pyaudio.paContinue)  # Continue streaming


# Initialize PyAudio
p = pyaudio.PyAudio()

# Open Audio Stream
stream = p.open(
    format=format,
    channels=channels,
    rate=samplerate,
    input=True,
    output=True,
    frames_per_buffer=blocksize,
    stream_callback=audio_callback
)

print("Listening for speech... (Waiting for trigger)")
#stream.start_stream()

try:
    while stream.is_active():
        pass  # Keep running
except KeyboardInterrupt:
    print("\n?? Stopped.")
    stream.stop_stream()
    stream.close()
    p.terminate()
