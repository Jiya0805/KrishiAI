import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, RandomFlip, RandomRotation, RandomZoom
import os

# --- 1. Load and Prepare the Data ---
# The path now points directly to the 'PlantVillage' folder.
DATA_DIR = 'backend/cnn_data/PlantVillage'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Check if the data directory exists before proceeding
if not os.path.exists(DATA_DIR):
    print(f"Error: The data directory '{DATA_DIR}' does not exist.")
    print("Please ensure your data is located in the 'cnn_data/PlantVillage' folder.")
else:
    # `image_dataset_from_directory` will infer classes from the subdirectories
    train_ds = image_dataset_from_directory(
        DATA_DIR,
        labels='inferred',
        label_mode='categorical',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset='training',
        seed=42
    )

    val_ds = image_dataset_from_directory(
        DATA_DIR,
        labels='inferred',
        label_mode='categorical',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset='validation',
        seed=42
    )

    num_classes = len(train_ds.class_names)
    print(f"Found {num_classes} classes.")

    # --- 2. Build the Model with Transfer Learning and Data Augmentation ---
    # Use data augmentation to increase the training data size and reduce overfitting.
    data_augmentation = Sequential([
        RandomFlip("horizontal_and_vertical"),
        RandomRotation(0.2),
        RandomZoom(0.2)
    ])

    # MobileNetV2 is an efficient base model for image classification
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )

    # Freeze the base model for the first training phase
    base_model.trainable = False

    # Construct the full model
    model = Sequential([
        data_augmentation,
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        Dense(num_classes, activation='softmax')
    ])

    # --- 3. Compile and Train the Model (Phase 1) ---
    print("Starting initial training (frozen base model)...")
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    epochs_initial = 5
    model.fit(train_ds, epochs=epochs_initial, validation_data=val_ds)

    # --- 4. Fine-Tuning (Phase 2) ---
    print("\nStarting fine-tuning...")
    # Unfreeze the base model to allow for fine-tuning
    base_model.trainable = True

    # Recompile the model with a lower learning rate for fine-tuning
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Continue training for a few more epochs
    epochs_fine_tune = 5
    model.fit(train_ds, epochs=epochs_fine_tune, validation_data=val_ds)

    # --- 5. Save the Final, Trained Model ---
    # Determine the path for saving the model
    project_root = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(project_root, 'backend', 'ml_models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'cnn_model.h5')

    model.save(model_path)
    print(f"\nOptimized CNN model successfully trained and saved to: {model_path}")
