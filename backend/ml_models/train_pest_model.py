import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomBrightness
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# --- 1. Load and Prepare the Data ---
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cnn_data', 'agricultural-pests-image-dataset')
IMG_SIZE = (224, 224) 
BATCH_SIZE = 32

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

# Debug: Print class names detected
print("Class names detected:")
print(train_ds.class_names)

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
print(f"Found {num_classes} classes (pests).")

# --- 2. Build the Model with Transfer Learning and Enhanced Data Augmentation ---
data_augmentation = Sequential([
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.2),
    RandomZoom(0.2),
    RandomBrightness(0.2),
    RandomContrast(0.2)
])

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

model = Sequential([
    data_augmentation,
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    Dense(num_classes, activation='softmax')
])

# --- 3. Compile and Train the Model (Phase 1) with Callbacks ---
print("Starting initial training (frozen base model)...")

# Define callbacks for training
callbacks_list = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(filepath='best_model_phase1.h5', monitor='val_loss', save_best_only=True)
]

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

epochs_initial = 10 
model.fit(train_ds, epochs=epochs_initial, validation_data=val_ds, callbacks=callbacks_list)

# --- 4. Fine-Tuning (Phase 2) with a very low learning rate ---
print("\nStarting fine-tuning...")
base_model.trainable = True

# Define callbacks for the fine-tuning phase
callbacks_list_fine_tune = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint(filepath='best_pest_model.h5', monitor='val_loss', save_best_only=True)
]

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

epochs_fine_tune = 20
model.fit(train_ds, epochs=epochs_fine_tune, validation_data=val_ds, callbacks=callbacks_list_fine_tune)

# --- 5. Save the Final, Trained Model ---
# The best model has already been saved by ModelCheckpoint.
project_root = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(project_root, 'backend', 'ml_models')
os.makedirs(model_dir, exist_ok=True)
final_model_path = os.path.join(model_dir, 'pest_model.h5')

# Load the best model from the fine-tuning phase and save it as the final model
best_model = tf.keras.models.load_model('best_pest_model.h5')
best_model.save(final_model_path)
print(f"\nOptimized pest classification model saved to: {final_model_path}")