# train_fruit_cnn.py
# Train a CNN on fruit sketches in data_fruits/ and save fruit_cnn.h5

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

DATA_DIR = "data_quickdraw"
IMG_SIZE = 64
BATCH_SIZE = 16  # Increased for more stable training
EPOCHS = 50

# Training generator WITH augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    shear_range=0.1,
    horizontal_flip=True,
    validation_split=0.2,
)

# Validation generator WITHOUT augmentation (critical fix!)
val_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2,
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
)

val_gen = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
)

num_classes = train_gen.num_classes
print(f"\nFound {num_classes} classes")
print("Classes and indices:", train_gen.class_indices)
print(f"Training samples: {train_gen.samples}")
print(f"Validation samples: {val_gen.samples}\n")

# Improved CNN model with batch normalization
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1), padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.2),

    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.2),

    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.3),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# Callbacks for better training
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=8,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=4,
    min_lr=0.00001,
    verbose=1
)

# Train
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
)

# Save model
model.save("fruit_cnn.h5")
print("\n✓ Saved model as fruit_cnn.h5")

# Print class order for GUI
class_indices = train_gen.class_indices
sorted_classes = [c for c, _ in sorted(class_indices.items(), key=lambda x: x[1])]
print("\n" + "="*50)
print("COPY THIS LINE TO fruit_gui.py:")
print(f"CLASS_NAMES = {sorted_classes}")
print("="*50)

# Print final metrics
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
print(f"\nFinal Training Accuracy: {final_train_acc:.2%}")
print(f"Final Validation Accuracy: {final_val_acc:.2%}")

