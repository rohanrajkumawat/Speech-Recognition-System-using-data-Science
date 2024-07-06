import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
import tarfile
import shutil

# Download the Speech Commands dataset
data_url = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
data_dir = tf.keras.utils.get_file('speech_commands', data_url, untar=True, cache_dir='.', cache_subdir='datasets')

# Define the function to load audio files and their labels
def load_audio_files(data_dir, keywords=['yes', 'no'], sample_rate=16000):
    audio_files = []
    labels = []
    for keyword in keywords:
        keyword_dir = os.path.join(data_dir, keyword)
        if os.path.exists(keyword_dir):
            for file_name in os.listdir(keyword_dir):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(keyword_dir, file_name)
                    audio_files.append(file_path)
                    labels.append(keyword)
        else:
            print(f"Directory {keyword_dir} does not exist.")
    return audio_files, labels

# Load a small subset of audio files
audio_files, labels = load_audio_files(data_dir, keywords=['yes', 'no'])

# Preprocess the audio files to extract features (e.g., spectrograms)
def preprocess_audio(file_path, sample_rate=16000):
    audio_binary = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(audio_binary, desired_channels=1, desired_samples=sample_rate)
    audio = tf.squeeze(audio, axis=-1)
    spectrogram = tf.signal.stft(audio, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    return spectrogram

# Preprocess a small subset of audio files
spectrograms = [preprocess_audio(file_path) for file_path in audio_files]
labels = [1 if label == 'yes' else 0 for label in labels]  # Convert labels to binary (1: 'yes', 0: 'no')

# Visualize a spectrogram
plt.figure(figsize=(10, 8))
plt.imshow(np.log(spectrograms[0] + 1e-10).numpy().T, aspect='auto', origin='lower')
plt.title("Spectrogram of 'yes'")
plt.show()

# Prepare the dataset for training
spectrograms = np.array([spec.numpy() for spec in spectrograms])
labels = np.array(labels)
spectrograms = np.expand_dims(spectrograms, -1)  # Add a channel dimension

# Split the data into training and validation sets
train_size = int(0.8 * len(spectrograms))
x_train, x_val = spectrograms[:train_size], spectrograms[train_size:]
y_train, y_val = labels[:train_size], labels[train_size:]

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(x_train.shape[1], x_train.shape[2], 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

# Evaluate the model
val_loss, val_acc = model.evaluate(x_val, y_val)
print(f"Validation Accuracy: {val_acc}")

# Create a small dataset folder
small_data_dir = './small_speech_commands'
if not os.path.exists(small_data_dir):
    os.makedirs(small_data_dir)

# Copy a few files from each keyword
for keyword in ['yes', 'no']:
    keyword_dir = os.path.join(data_dir, keyword)
    small_keyword_dir = os.path.join(small_data_dir, keyword)
    if not os.path.exists(small_keyword_dir):
        os.makedirs(small_keyword_dir)
    for file_name in os.listdir(keyword_dir)[:10]:  # Copy only 10 files for each keyword
        shutil.copy(os.path.join(keyword_dir, file_name), small_keyword_dir)

# Create a tar.gz file of the small dataset
with tarfile.open('./small_speech_commands.tar.gz', 'w:gz') as tar:
    tar.add(small_data_dir, arcname=os.path.basename(small_data_dir))

print("Small dataset created and saved as ./small_speech_commands.tar.gz")
