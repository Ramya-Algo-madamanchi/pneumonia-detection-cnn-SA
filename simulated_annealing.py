
import random
import math
import json
import tensorflow as tf

# 1. Configuration
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
SEED = 42

random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# 2. Load training and validation datasets
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

# Improve input pipeline performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# 3. Hyperparameter search space
SEARCH_SPACE = {
    "learning_rate": [0.0001, 0.0005, 0.001],
    "dropout_rate": [0.3, 0.5],
    "filters": [16, 32]
}

INITIAL_TEMPERATURE = 0.1
COOLING_RATE = 0.7
MAX_ITERATIONS = 5
EPOCHS_PER_CANDIDATE = 5


# 4. Generate a neighboring solution
def get_neighbor(current):
    neighbor = current.copy()

    parameter = random.choice(list(SEARCH_SPACE.keys()))
    choices = [
        value
        for value in SEARCH_SPACE[parameter]
        if value != current[parameter]
    ]

    neighbor[parameter] = random.choice(choices)
    return neighbor


# 5. Build the CNN model
def build_cnn(params):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(150, 150, 1)),
        tf.keras.layers.Rescaling(1.0 / 255),

        tf.keras.layers.Conv2D(
            params["filters"], 3, activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(pool_size=2),

        tf.keras.layers.Conv2D(
            params["filters"] * 2, 3, activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(pool_size=2),

        tf.keras.layers.Conv2D(
            params["filters"] * 4, 3, activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(pool_size=2),

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

    return model


# 6. Evaluate one hyperparameter configuration
def evaluate_cnn(params):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    model = build_cnn(params)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_PER_CANDIDATE,
        callbacks=[early_stopping],
        verbose=1
    )

    best_epoch = min(
        range(len(history.history["val_loss"])),
        key=lambda i: history.history["val_loss"][i]
    )

    best_loss = history.history["val_loss"][best_epoch]
    best_accuracy = history.history["val_accuracy"][best_epoch]

    print("\nHyperparameters:", params)
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Validation accuracy at best-loss epoch: {best_accuracy:.4f}")

    del model
    tf.keras.backend.clear_session()

    return best_loss


# 7. Simulated Annealing acceptance probability
def acceptance_probability(current_loss, new_loss, temperature):
    if new_loss < current_loss:
        return 1.0

    temperature = max(temperature, 1e-10)

    return math.exp(
        -(new_loss - current_loss) / temperature
    )


# 8. Run Simulated Annealing
def simulated_annealing():
    current = {
        key: random.choice(values)
        for key, values in SEARCH_SPACE.items()
    }

    print("\nEvaluating initial configuration:", current)
    current_loss = evaluate_cnn(current)

    best = current.copy()
    best_loss = current_loss

    temperature = INITIAL_TEMPERATURE

    for iteration in range(MAX_ITERATIONS):
        print(f"\n{'=' * 40}")
        print(f"SA Iteration {iteration + 1}/{MAX_ITERATIONS}")
        print(f"{'=' * 40}")

        candidate = get_neighbor(current)
        print("Candidate configuration:", candidate)

        candidate_loss = evaluate_cnn(candidate)

        probability = acceptance_probability(
            current_loss,
            candidate_loss,
            temperature
        )

        if (
            candidate_loss < current_loss
            or random.random() < probability
        ):
            current = candidate.copy()
            current_loss = candidate_loss
            print("Candidate accepted.")
        else:
            print("Candidate rejected.")

        # Track the best candidate evaluated, not only the current one
        if candidate_loss < best_loss:
            best = candidate.copy()
            best_loss = candidate_loss

        print(f"Acceptance probability: {probability:.4f}")
        print(f"Temperature: {temperature:.5f}")
        print(f"Best validation loss so far: {best_loss:.4f}")

        temperature *= COOLING_RATE

    print("\n" + "=" * 40)
    print("SIMULATED ANNEALING RESULTS")
    print("=" * 40)
    print("Best hyperparameters:", best)
    print(f"Best validation loss: {best_loss:.4f}")

    with open("best_hyperparameters.json", "w") as file:
        json.dump(best, file, indent=4)

    print("Best hyperparameters saved to best_hyperparameters.json")


if __name__ == "__main__":
    simulated_annealing()
