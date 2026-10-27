
import tensorflow as tf

# Image settings
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
SEED = 42

# Load 80% of the training folder for training
train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="binary"
)

# Load the remaining 20% for validation
val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="binary"
)

print("Classes:", train_ds.class_names)
print("Training batches:", tf.data.experimental.cardinality(train_ds).numpy())
print("Validation batches:", tf.data.experimental.cardinality(val_ds).numpy())

for images, labels in train_ds.take(1):
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)

# Normalize pixel values from 0–255 to 0–1
normalization = tf.keras.layers.Rescaling(1.0 / 255)

# Build the CNN
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(150, 150, 1)),
    normalization,

    # First convolution block
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Second convolution block
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Convert feature maps into a vector
    tf.keras.layers.Flatten(),

    # Classification layer
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.5),

    # Binary output: NORMAL or PNEUMONIA
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# Configure training
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Display the architecture
model.summary()

# Train the CNN
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

# Load the untouched test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="binary",
    shuffle=False
)

print("Test classes:", test_ds.class_names)

# Evaluate on the untouched test dataset
test_loss, test_accuracy = model.evaluate(test_ds)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
from sklearn.metrics import confusion_matrix, classification_report

# Get actual labels
y_true = []
for images, labels in test_ds:
    y_true.extend(labels.numpy().ravel().astype(int))

# Predict probabilities
y_prob = model.predict(test_ds, verbose=0).ravel()

# Convert probabilities into class labels
y_pred = (y_prob >= 0.5).astype(int)

# Display results
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=["NORMAL", "PNEUMONIA"],
    digits=4
))