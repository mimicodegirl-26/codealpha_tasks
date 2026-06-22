import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Bidirectional, LSTM,
    Dropout, Dense, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ========== CONFIG ==========
DATASET_PATH = 'data'          # Actor folders inside data/
SAMPLE_RATE = 22050
N_MFCC = 60                    # more coefficients
MAX_PAD_LEN = 200              # enough for augmented audio

EMOTION_MAP = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

# ========== FEATURE EXTRACTION ==========
def extract_features(file_path, augment=False):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        # Data augmentation (only for training)
        if augment:
            # Random noise
            noise = 0.005 * np.random.randn(len(audio))
            audio = audio + noise
            # Time stretching (90–110%)
            rate = np.random.uniform(0.9, 1.1)
            audio = librosa.effects.time_stretch(y=audio, rate=rate)
            # Pitch shifting (±2 semitones)
            n_steps = np.random.randint(-2, 3)
            audio = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)

        # Extract MFCCs + delta + delta-delta
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        features = np.vstack([mfcc, delta, delta2])   # (180, time)

        # Pad/truncate to fixed length
        if features.shape[1] < MAX_PAD_LEN:
            pad_width = MAX_PAD_LEN - features.shape[1]
            features = np.pad(features, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            features = features[:, :MAX_PAD_LEN]

        return features
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# ========== LOAD ORIGINAL DATA ==========
features, labels = [], []
print("Extracting features (no augmentation)...")
for actor_folder in os.listdir(DATASET_PATH):
    actor_path = os.path.join(DATASET_PATH, actor_folder)
    if not os.path.isdir(actor_path):
        continue
    for file in os.listdir(actor_path):
        if not file.endswith('.wav'):
            continue
        parts = file.split('-')
        if len(parts) < 3:
            continue
        emotion_code = parts[2]
        emotion = EMOTION_MAP.get(emotion_code)
        if not emotion:
            continue
        file_path = os.path.join(actor_path, file)
        feat = extract_features(file_path, augment=False)
        if feat is not None:
            features.append(feat)
            labels.append(emotion)

X = np.array(features)
y = np.array(labels)
print(f"Loaded {len(X)} samples. Shape: {X.shape}")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)
joblib.dump(le, 'label_encoder.pkl')
np.save('classes.npy', le.classes_)

# Split (stratified, 80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y_encoded)

# ========== DATA AUGMENTATION (only on training set) ==========
print("Augmenting training set...")
X_train_aug, y_train_aug = [], []
for i in range(len(X_train)):
    # Original
    X_train_aug.append(X_train[i])
    y_train_aug.append(y_train[i])
    # Augmented version (since we stored features, we add noise to features)
    noise = 0.01 * np.random.randn(*X_train[i].shape)
    X_train_aug.append(X_train[i] + noise)
    y_train_aug.append(y_train[i])

X_train = np.array(X_train_aug)
y_train = np.array(y_train_aug)
print(f"Training set size after augmentation: {len(X_train)}")

# Transpose for Conv1D: (samples, time_steps, features)
X_train = X_train.transpose(0, 2, 1)   # (samples, MAX_PAD_LEN, 180)
X_test  = X_test.transpose(0, 2, 1)

# ========== MODEL ==========
model = Sequential([
    Conv1D(128, kernel_size=5, activation='relu', input_shape=(MAX_PAD_LEN, N_MFCC*3)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Conv1D(256, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Bidirectional(LSTM(256, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(128, return_sequences=False)),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dense(y_train.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# ========== TRAINING ==========
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100, batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ========== EVALUATION ==========
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=le.classes_, digits=3))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_.tolist(), yticklabels=le.classes_.tolist())
plt.title('Confusion Matrix (Custom CNN‑BiLSTM)')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# Training curves
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# Save model
model.save('emotion_model.h5')
print("Model saved as emotion_model.h5")