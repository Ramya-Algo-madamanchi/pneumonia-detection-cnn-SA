
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

IMG_SIZE = (150, 150)
BATCH_SIZE = 32

# Load the saved model; no retraining required
model = tf.keras.models.load_model("best_pneumonia_cnn.keras")

# Load the test data in a fixed order
test_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="binary",
    shuffle=False
)

# Collect actual labels and predicted probabilities
y_true = np.concatenate([
    labels.numpy().ravel().astype(int)
    for _, labels in test_ds
])

y_prob = model.predict(test_ds, verbose=0).ravel()

print("\nPrediction probability analysis")
print("NORMAL = 0, PNEUMONIA = 1")
print(f"Mean probability: {np.mean(y_prob):.4f}")
print(f"Minimum probability: {np.min(y_prob):.4f}")
print(f"Maximum probability: {np.max(y_prob):.4f}")

# Show probability distributions for each actual class
for class_id, class_name in [(0, "NORMAL"), (1, "PNEUMONIA")]:
    probabilities = y_prob[y_true == class_id]

    print(f"\nActual class: {class_name}")
    print(f"Mean pneumonia probability: {np.mean(probabilities):.4f}")
    print(f"Median pneumonia probability: {np.median(probabilities):.4f}")
    print(f"Below 0.5: {np.sum(probabilities < 0.5)}")
    print(f"At least 0.5: {np.sum(probabilities >= 0.5)}")

# Confirm the current 0.5 threshold results
y_pred = (y_prob >= 0.5).astype(int)

print("\nConfusion matrix at threshold 0.5:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=["NORMAL", "PNEUMONIA"],
    digits=4,
    zero_division=0
))

