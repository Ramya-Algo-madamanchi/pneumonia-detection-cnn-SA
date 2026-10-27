
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

IMG_SIZE = (150, 150)
BATCH_SIZE = 32

model = tf.keras.models.load_model("best_pneumonia_cnn.keras")

test_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="binary",
    shuffle=False
)

y_true = np.concatenate([
    labels.numpy().ravel().astype(int)
    for _, labels in test_ds
])

y_prob = model.predict(test_ds, verbose=0).ravel()

print("\nThreshold comparison")
print("-" * 75)
print(
    f"{'Threshold':<12}{'Accuracy':<12}"
    f"{'Normal recall':<16}{'Pneumonia recall':<18}"
)
print("-" * 75)

for threshold in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]:
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    accuracy = (tn + tp) / len(y_true)
    normal_recall = tn / (tn + fp) if (tn + fp) else 0
    pneumonia_recall = tp / (tp + fn) if (tp + fn) else 0

    print(
        f"{threshold:<12.2f}"
        f"{accuracy:<12.4f}"
        f"{normal_recall:<16.4f}"
        f"{pneumonia_recall:<18.4f}"
    )
