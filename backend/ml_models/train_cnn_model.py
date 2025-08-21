import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, RandomFlip, RandomRotation, RandomZoom
import os

# --- 1. Load and Prepare the Data ---
# The path now points directly to the 'PlantVillage' folder.
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cnn_data', 'PlantVillage')
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Check if the data directory exists before proceeding
if not os.path.exists(DATA_DIR):
    print(f"Error: The data directory '{DATA_DIR}' does not exist.")
    print("Please ensure your data is located in the 'cnn_data/PlantVillage' folder.")
else:
    # Debug: Print class names detected
    print("Class names detected:")
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
    print("Class names:", train_ds.class_names)

    # Debug: Count images per class in train_ds
    class_counts = {name: 0 for name in train_ds.class_names}
    for batch_images, batch_labels in train_ds:
        labels = batch_labels.numpy()
        for label in labels:
            class_idx = label.argmax()
            class_name = train_ds.class_names[class_idx]
            class_counts[class_name] += 1
    print("Number of images per class in training set:")
    for name, count in class_counts.items():
        print(f"  {name}: {count}")

    # --- 2. Build the Model with Transfer Learning and Data Augmentation ---
    # Use data augmentation to increase the training data size and reduce overfitting.
    data_augmentation = Sequential([
        RandomFlip("horizontal_and_vertical"),
        RandomRotation(0.3),
        RandomZoom(0.3),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.2)
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
        Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    # --- 3. Compile and Train the Model (Phase 1) ---
    print("Starting initial training (frozen base model)...")
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Compute class weights for imbalance
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np
    labels_list = []
    for batch_images, batch_labels in train_ds:
        labels_list.extend(np.argmax(batch_labels.numpy(), axis=1))
    class_weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=labels_list)
    class_weights_dict = {i: w for i, w in enumerate(class_weights)}
    print("Class weights:", class_weights_dict)

    epochs_initial = 15
    model.fit(train_ds, epochs=epochs_initial, validation_data=val_ds, class_weight=class_weights_dict)

    # --- 4. Fine-Tuning (Phase 2) ---
    print("\nStarting fine-tuning...")
    # Unfreeze last 30 layers of base model for fine-tuning
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    for layer in base_model.layers[-30:]:
        layer.trainable = True

    # Recompile the model with a lower learning rate for fine-tuning
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Continue training for more epochs
    epochs_fine_tune = 15
    model.fit(train_ds, epochs=epochs_fine_tune, validation_data=val_ds, class_weight=class_weights_dict)

    # --- 5. Save the Final, Trained Model ---
    # Determine the path for saving the model
    project_root = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(project_root, 'backend', 'ml_models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'cnn_model.h5')

    model.save(model_path)
    print(f"\nOptimized CNN model successfully trained and saved to: {model_path}")
