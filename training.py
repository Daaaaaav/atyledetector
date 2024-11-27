import os
import shutil
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

TRAIN_CSV = 'styles/train/train_split.csv'
VALID_CSV = 'styles/valid/valid_split.csv'
TRAIN_IMAGES_DIR = 'styles/train/images'
VALID_IMAGES_DIR = 'styles/valid/images'
SORTED_TRAIN_DIR = 'sorted_data/train'
SORTED_VALID_DIR = 'sorted_data/validation'

def sort_images(csv_file, source_dir, dest_dir):
    data = pd.read_csv(csv_file)
    for _, row in data.iterrows():
        filename = row['filename']
        label = row['class']
        source_path = os.path.join(source_dir, filename)
        dest_folder = os.path.join(dest_dir, label)
        dest_path = os.path.join(dest_folder, filename)
        os.makedirs(dest_folder, exist_ok=True)
        if os.path.exists(source_path):
            shutil.copy(source_path, dest_path)
        else:
            print(f"Warning: File {source_path} not found. Skipping. (Class: {label})")
    print(f"Images sorted into {dest_dir}")

sort_images(TRAIN_CSV, TRAIN_IMAGES_DIR, SORTED_TRAIN_DIR)
sort_images(VALID_CSV, VALID_IMAGES_DIR, SORTED_VALID_DIR)

train_df = pd.read_csv(TRAIN_CSV)
valid_df = pd.read_csv(VALID_CSV)

train_df.columns = train_df.columns.str.strip()
valid_df.columns = valid_df.columns.str.strip()

train_df['class'] = train_df['class'].apply(str)
valid_df['class'] = valid_df['class'].apply(str)

assert train_df['class'].dtype == object, "train_df labels are not strings"
assert valid_df['class'].dtype == object, "valid_df labels are not strings"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory=SORTED_TRAIN_DIR,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

valid_generator = valid_datagen.flow_from_directory(
    directory=SORTED_VALID_DIR,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

if valid_generator.samples == 0:
    raise ValueError("Validation generator contains no images. Please check the dataset and sorting process.")

model = Sequential([
    Input(shape=(150, 150, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // valid_generator.batch_size
)

model.save('clothing_style_model.h5')

print("Model training complete and saved as 'clothing_style_model.h5'.")
