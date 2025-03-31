import os
import pathlib
import json
import logging

import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.layers import (
    Conv2D,
    Flatten,
    Dense,
    MaxPooling2D,
    Dropout,
    Resizing,
    Input,
    Normalization,
)
from keras import models
from IPython import display

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------- #
#  Configuration Management #
# ------------------------- #

# Path to the configuration file
CONFIG_FILE = "config.json"

# Default configuration parameters
default_config = {
    "DATASET_PATH": "data/",
    "BATCH_SIZE": 64,
    "VALIDATION_SPLIT": 0.2,
    "SEED": 0,
    "OUTPUT_SEQUENCE_LENGTH": 16000,
    "EPOCHS": 10,
    "EARLY_STOPPING_PATIENCE": 2,
}

# Load configuration from JSON file if available; otherwise use default settings.
try:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
        logger.info("Configuration loaded from %s", CONFIG_FILE)
    else:
        config = default_config
        logger.info("Configuration file not found. Using default configuration.")
except Exception as e:
    logger.error("Error loading configuration: %s", e)
    config = default_config

# ------------------------ #
#  Data Download & Loading #
# ------------------------ #

# Set dataset path from configuration
DATASET_PATH = config.get("DATASET_PATH", "data/")
data_dir = pathlib.Path(DATASET_PATH)

# Download and extract the dataset
try:
    tf.keras.utils.get_file(
        "voicedataset.zip",
        origin="http://aiolearn.com/dl/datasets/voicedata.zip",
        extract=True,
        cache_dir=".",
        cache_subdir="data",
    )
    logger.info("Dataset downloaded and extracted successfully.")
except Exception as e:
    logger.error("Error downloading dataset: %s", e)
    raise e

# List available files in the dataset directory and filter unwanted ones
try:
    files = tf.io.gfile.listdir(str(data_dir))
    commands = np.array(files)
    commands = commands[
        (commands != "README.md")
        & (commands != ".DS_Store")
        & (commands != "voicedataset.zip")
    ]
    logger.info("Commands Available: %s", commands)
except Exception as e:
    logger.error("Error listing dataset directory: %s", e)
    raise e

# Create training and testing datasets using audio_dataset_from_directory
try:
    X_train, X_test = tf.keras.utils.audio_dataset_from_directory(
        directory=data_dir,
        batch_size=config.get("BATCH_SIZE", 64),
        validation_split=config.get("VALIDATION_SPLIT", 0.2),
        seed=config.get("SEED", 0),
        output_sequence_length=config.get("OUTPUT_SEQUENCE_LENGTH", 16000),
        subset="both",
    )
except Exception as e:
    logger.error("Error creating audio dataset: %s", e)
    raise e

# Retrieve label names from the dataset
label_names = np.array(X_train.class_names)
logger.info("Labels: %s", label_names)

# ------------------------ #
#   Data Preprocessing     #
# ------------------------ #

def squeeze(audio, labels):
    """Remove the last dimension from the audio tensor."""
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels

# Apply the squeeze function to training and testing datasets
X_train = X_train.map(squeeze, tf.data.AUTOTUNE)
X_test = X_test.map(squeeze, tf.data.AUTOTUNE)

# Shard the test dataset into two parts for evaluation
val = X_test
X_test = X_test.shard(num_shards=2, index=0)
val = X_test.shard(num_shards=2, index=1)

# Display sample audio and labels from the training set
for audio_sample, label in X_train.take(1):
    logger.info("Sample audio tensor: %s", audio_sample)
    logger.info("Corresponding label: %s", label_names[label])

for example_audio, example_labels in X_train.take(1):
    logger.info("Example audio shape: %s", example_audio.shape)
    logger.info("Example labels shape: %s", example_labels.shape)

# Plot sample audio signals
plt.figure(figsize=(10, 10))
rows = 3
cols = 3
n = rows * cols
for i in range(n):
    plt.subplot(rows, cols, i + 1)
    audio_signal = example_audio[i]
    plt.plot(audio_signal)
    plt.title(label_names[example_labels[i]])
    plt.yticks(np.arange(-1.2, 1.2, 0.2))
    plt.ylim([-1.1, 1.1])
plt.tight_layout()
plt.show()

def get_spectrogram(waveform):
    """Convert a waveform to a spectrogram using Short-Time Fourier Transform (STFT)."""
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]  # Add channels dimension
    return spectrogram

# Display spectrograms and audio playback for sample audio
for i in range(5):
    label = label_names[example_labels[i]]
    waveform = example_audio[i]
    spectrogram = get_spectrogram(waveform)
    logger.info("Label: %s", label)
    logger.info("Waveform shape: %s", waveform.shape)
    logger.info("Spectrogram shape: %s", spectrogram.shape)
    display.display(display.Audio(waveform, rate=16000))

def plot_spectrogram(spectrogram, ax):
    """Plot a spectrogram on the provided matplotlib axis."""
    if len(spectrogram.shape) > 2:
        spectrogram = np.squeeze(spectrogram, axis=-1)
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)

plt.close()
fig, axes = plt.subplots(2, figsize=(10, 8))
timescale = np.arange(waveform.shape[0])
axes[0].plot(timescale, waveform.numpy())
axes[0].set_title("Waveform")
axes[0].set_xlim([0, 16000])
plot_spectrogram(spectrogram.numpy(), axes[1])
axes[1].set_title("Spectrogram")
plt.suptitle(label.title())
plt.show()
display.display(display.Audio(waveform, rate=16000))

def make_spec_ds(ds):
    """Convert a dataset of audio waveforms into spectrograms."""
    return ds.map(
        lambda audio, label: (get_spectrogram(audio), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

# Create spectrogram datasets for training, validation, and testing
train_spectrogram_ds = make_spec_ds(X_train)
val_spectrogram_ds = make_spec_ds(val)
test_spectrogram_ds = make_spec_ds(X_test)

# Cache and prefetch datasets for performance
train_spectrogram_ds = train_spectrogram_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
val_spectrogram_ds = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
test_spectrogram_ds = test_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)

# Determine the input shape for the model
for example_spectrograms, _ in train_spectrogram_ds.take(1):
    input_shape = example_spectrograms.shape[1:]
    break
logger.info("Input shape: %s", input_shape)
num_labels = len(label_names)

# ------------------------ #
#      Model Building      #
# ------------------------ #

# Create a normalization layer and adapt it to the training spectrograms
norm_layer = Normalization()
norm_layer.adapt(data=train_spectrogram_ds.map(lambda spec, label: spec))

# Build the Sequential model
model = models.Sequential([
    Input(shape=input_shape),
    Resizing(32, 32),   # Downsample the input to a fixed size
    norm_layer,         # Normalize the input data
    Conv2D(32, 3, activation="relu"),
    Conv2D(64, 3, activation="relu"),
    MaxPooling2D(),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(num_labels),  # Output layer with units equal to the number of labels
])
model.summary()

# Compile the model with appropriate loss function and optimizer
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# ------------------------ #
#       Model Training     #
# ------------------------ #

# Train the model with early stopping callback
try:
    history = model.fit(
        train_spectrogram_ds,
        validation_data=val_spectrogram_ds,
        epochs=config.get("EPOCHS", 10),
        callbacks=tf.keras.callbacks.EarlyStopping(
            verbose=1, patience=config.get("EARLY_STOPPING_PATIENCE", 2)
        ),
    )
except Exception as e:
    logger.error("Error during model training: %s", e)
    raise e

# Plot training and validation loss/accuracy
metrics = history.history
plt.close()
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(history.epoch, metrics["loss"], label="loss")
plt.plot(history.epoch, metrics["val_loss"], label="val_loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss [CrossEntropy]")

plt.subplot(1, 2, 2)
plt.plot(history.epoch, 100 * np.array(metrics["accuracy"]), label="accuracy")
plt.plot(history.epoch, 100 * np.array(metrics["val_accuracy"]), label="val_accuracy")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy [%]")
plt.show()

# ------------------------ #
#     Model Evaluation     #
# ------------------------ #

try:
    eval_results = model.evaluate(test_spectrogram_ds, return_dict=True)
    logger.info("Evaluation results: %s", eval_results)
except Exception as e:
    logger.error("Error during model evaluation: %s", e)
    raise e

# Generate confusion matrix for test set
y_pred = model.predict(test_spectrogram_ds)
y_pred = tf.argmax(y_pred, axis=1)
y_true = tf.concat(list(test_spectrogram_ds.map(lambda s, lab: lab)), axis=0)
confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, xticklabels=label_names, yticklabels=label_names, annot=True, fmt="g")
plt.xlabel("Prediction")
plt.ylabel("Label")
plt.show()

# ------------------------ #
#    Single Audio Prediction
# ------------------------ #

def predict_audio_file(file_path):
    """
    Load an audio file, convert it to a spectrogram, and predict its label using the trained model.
    Returns the softmax probabilities for each label.
    """
    try:
        file_contents = tf.io.read_file(file_path)
        audio, sample_rate = tf.audio.decode_wav(
            file_contents, desired_channels=1, desired_samples=16000
        )
        audio = tf.squeeze(audio, axis=-1)
        spectrogram = get_spectrogram(audio)
        spectrogram = spectrogram[tf.newaxis, ...]
        prediction = model(spectrogram)
        return tf.nn.softmax(prediction[0])
    except Exception as e:
        logger.error("Error processing file %s: %s", file_path, e)
        return None

# Predict on a sample audio file
sample_file = "data/down/0a9f9af7_nohash_0.wav"
predictions = predict_audio_file(sample_file)
if predictions is not None:
    plt.figure()
    plt.bar(label_names, predictions)
    plt.title("Prediction for sample audio")
    plt.show()
    display.display(display.Audio('audio', rate=16000))
else:
    logger.error("Prediction failed for sample file.")

# ------------------------ #
#    Audio Recording Demo  #
# ------------------------ #

try:
    import pyaudio

    p = pyaudio.PyAudio()
    RECORD_SECONDS = 1     # Recording duration in seconds
    CHUNK = 16000          # Frames per buffer
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=CHUNK
    )
    logger.info("Recording audio...")
    audio_data = stream.read(CHUNK)
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded audio to a WAV file
    import wave

    with wave.open("recorded_audio.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(audio_data)
    logger.info("Recorded audio saved as recorded_audio.wav")
except Exception as e:
    logger.error("Error during audio recording: %s", e)

# ------------------------ #
#      Save the Model      #
# ------------------------ #

try:
    model.save("model.h5")
    logger.info("Model saved to model.h5")
except Exception as e:
    logger.error("Error saving model: %s", e)
    raise e
