import pyaudio
import numpy as np
import torch
from scipy.signal import butter, sosfilt
import speechbrain as sb
from speechbrain.pretrained import MelSpectrogramEncoder
import wave
import time
from cameradriver import compare_embeddings
from pymongo import MongoClient
from videostream import VideoStream

# Audio Parameters
samplerate = 44100  # Required for SpeechBrain
channels = 1
blocksize = 1024 * 16  # Buffer size
format = pyaudio.paInt16
trigger_threshold = 50.0  # Adjust based on mic sensitivity
p = None       # PyAudio instance

# Load SpeechBrain Model for Speaker Embeddings
classifier = MelSpectrogramEncoder.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb-mel-spec", 
    savedir="./melspec"  # Optional, to save the model locally
)
stream_running = True
no_samples =1

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["project"]
users_collection = db["users"]  # Collection storing embeddings

# State Variables
triggered = False  # Start listening only when triggered
buffer = []  # Store samples for processing
stream = None  # Will be assigned later

last_embedding = None  # Store the latest embedding
output_wav = "recorded_audio.wav"



def save_audio(filename, audio_data):
    """Save recorded audio to a WAV file for debugging."""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())

# Design a Bandpass Filter (3kHz - 15kHz)
def design_filter(lowfreq, highfreq, fs, order=3):
    nyq = 0.5 * fs
    low = lowfreq / nyq
    high = highfreq / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos

sos = design_filter(2000, 15000, samplerate, 3)  # Adjust frequency range




def process_audio():
    """Convert buffered audio into a Mel-spectrogram embedding and return it."""
    global buffer, last_embedding

    if not buffer:
        return None  # No data, return None
        
    audio_array = np.concatenate(buffer).astype(np.int16)  # Keep int16 for WAV saving
    save_audio(output_wav, audio_array)  # Save to a .wav file for debugging

    # Convert to float32 and normalize int16 to [-1, 1]
    audio_data = np.concatenate(buffer).astype(np.float32) / 32768.0
    audio_tensor = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)

    
    spk_embedding = classifier.encode_waveform(audio_tensor)

    # Get speaker embedding (the output of the MelSpectrogramEncoder)
    last_embedding = spk_embedding.squeeze().detach().numpy()



    print("?? Extracted SpeechBrain Mel-Spectrogram Embedding:", last_embedding[:5])  # Print first 5 values as a preview

    buffer = []  # Clear buffer
    return last_embedding  # Return the embedding


def audio_callback(in_data, frame_count, time_info, status):
    """Audio callback function to process incoming audio."""
    global triggered, buffer, stream_running, no_samples

    try:
        if no_samples == 0:
            print("No more samples to process. Stopping stream.")
            stream_running = False  # Signal to stop the stream
            return (None, pyaudio.paComplete)  # Stop stream properly
        
        # Convert byte data to NumPy int16 array
        audio_data = np.frombuffer(in_data, dtype=np.int16)

        # Apply SOS filter to incoming audio
        filtered_audio = sosfilt(sos, audio_data)

        # Compute energy level to check for trigger activation
        energy = np.mean(np.abs(filtered_audio))

        if not triggered and energy > trigger_threshold:
            print("Triggered! Start recording...")
            triggered = True

        if triggered:
            buffer.append(filtered_audio)  # Store audio chunks
            if len(buffer) * blocksize >= samplerate * 2:  # 2 seconds of speech
                triggered = False  # Reset trigger
                print(f"Processing recorded speech... Audio saved as {output_wav}")
                process_audio()  # Get embedding and return
                no_samples -= 1  # Decrease remaining samples
        
        return (in_data, pyaudio.paContinue)
    except Exception as e:
        print(f"Error in audio callback: {e}")



def start_listening(samples=1):
    """Start PyAudio stream and listen for speech."""
    global p, stream , stream_running , no_samples
    stream_running = True
    no_samples=samples
    
    p = pyaudio.PyAudio() 
    
    # Audio Parameters
    samplerate = 44100  # Required for SpeechBrain
    channels = 1
    blocksize = 1024 * 16  # Buffer size
    format = pyaudio.paInt16
    trigger_threshold = 50.0  # Adjust based on mic sensitivity
    
    stream = p.open(
        format=format,
        channels=channels,
        rate=samplerate,
        input=True,
        output=False,
        frames_per_buffer=blocksize,
        stream_callback=audio_callback
    )

    print("?? Listening for speech... (Waiting for trigger)")
    stream.start_stream()

    try:
        while stream.is_active() and stream_running:
            pass
    except KeyboardInterrupt:
        print("\n?? Stopping manually...")
        stop_listening()


def stop_listening():
    """Stop the PyAudio stream."""
    global stream, p

    if stream:
        stream.stop_stream()
        stream.close()
    if p:
        p.terminate()
        p = None
    
    
def get_last_embedding():
    """Return the last extracted embedding."""
    return last_embedding
    
def embed_voice(samples=1):
    audio_embeddings =[]
    start_listening(samples)
    print(f"say your wakeword {samples} time(s)" )
    for i in range(1, samples+1):
        
        # Retrieve the last embedding after the stream stops
        audio_embeddings.append(get_last_embedding())
    
    
    return audio_embeddings
    
def recognise_voice(retries=1,threshold=0.9):
    global stream_running
    chances = retries
    result={}
    
    
    for _ in range(retries):
        start_listening(1)
          # Capture one sample per attempt
        recorded_embedding = get_last_embedding()

        if recorded_embedding is None:
            print("? No valid embedding captured.")
            continue  # Retry authentication

        result = verify_voice(recorded_embedding,threshold)
        print(result)
        if result["verified"]:
            print(f"? Success! User {result['match']} authenticated.")
            stream_running = False
            #stop_listening()
            return result  # Stop early if authentication is successful

        chances -= 1
        print(f"? Authentication failed. {chances} attempt(s) remaining.")

    print("? Voice authentication failed. User is not in the system.")
    #stream_running = False
    return result

    
def verify_voice(emb1, threshold=0.78):
    best_match = None
    best_score = float("-inf") 
    users = users_collection.find({})
    
    for user in users:
        username = user["username"]
        stored_embeddings = user["audio_embeddings"]  # List of stored embeddings per user
        #print(stored_embeddings)
        if stored_embeddings:
            for stored_emb in stored_embeddings:
                stored_emb = np.array(stored_emb)
                similarity_score = compare_embeddings(emb1, stored_emb)

                if similarity_score > best_score:
                    best_score = similarity_score
                    best_match = username

    is_match = best_score > threshold
    if is_match:
        return {"verified": is_match, "match": best_match, "similarity": best_score}
    else:
        return {"verified": is_match, "match": None, "similarity": best_score}
    

    
    
if __name__ == "__main__":
    video_stream=VideoStream()#idk why this makes it work
    #input("press_enter")
    
    recognise_voice(3)
    stop_listening()
    video_stream.stop()
    
