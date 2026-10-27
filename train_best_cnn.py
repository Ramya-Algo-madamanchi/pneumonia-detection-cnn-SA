
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

# 1. Configuration
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
SEED = 42

tf.keras.utils.set_random_seed(SEED)

# 2. Load best hyperparameters
with open("best_hyperparameters.json", "r") as file:
    params = json.load(file)

print("Selected hyperparameters:", params)

# 3. Load training and validation datasets
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

# 4. Build the final CNN
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(150, 150, 1)),
    tf.keras.layers.Rescaling(1.0 / 255),

    tf.keras.layers.Conv2D(
        params["filters"], 3, activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        params["filters"] * 2, 3, activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        params["filters"] * 4, 3, activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(params["dropout_rate"]),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=params["learning_rate"]
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 5. Train using training data and monitor validation data
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=[early_stopping],
    verbose=1
)

# 6. Save the trained model
model.save("best_pneumonia_cnn.keras")
print("\nModel saved as best_pneumonia_cnn.keras")

# 7. Load the separate test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="binary",
    shuffle=False
)

# 8. Evaluate on unseen test data
test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# 9. Generate predictions and classification metrics
y_true = np.concatenate([
    labels.numpy().ravel() for _, labels in test_ds
]).astype(int)

y_prob = model.predict(test_ds, verbose=1).ravel()
y_pred = (y_prob >= 0.5).astype(int)

class_names = [
    test_ds.class_names[0],
    test_ds.class_names[1]
]

print("\nClass names:", class_names)

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=class_names,
    digits=4,
    zero_division=0
))

print(f"Accuracy from predictions: {accuracy_score(y_true, y_pred) * 100:.2f}%")
