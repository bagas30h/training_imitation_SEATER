import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, Dropout, Flatten, Dense, BatchNormalization, InputLayer, Attention, Reshape, Multiply, Input, Concatenate, Reshape, Multiply
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import cv2
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load data only with existing columns
columns = ['image_filename', 'linear_velocity', 'angular_velocity']
data = pd.read_csv('data_filtered.csv', names=columns)

# Clean angular velocity values
data['angular_velocity'] = pd.to_numeric(data['angular_velocity'], errors='coerce')
data.dropna(subset=['angular_velocity'], inplace=True)

# Filter data for each camera
camera_right_data = data[data['image_filename'].str.contains("camera_right")]
camera_left_data = data[data['image_filename'].str.contains("camera_left")]
camera_belok_data = data[data['image_filename'].str.contains("camera_belok")]

print(f"Total data: {len(data)}")
print(f"Data camera_belok: {len(camera_belok_data)}")
print(f"Sample camera_belok data:\n{camera_belok_data.head()}")

# Menyeimbangkan data antara kamera kanan, kiri, dan belok
min_length = min(len(camera_right_data), len(camera_left_data), len(camera_belok_data))
camera_right_data = camera_right_data.sample(n=min_length, random_state=42)
camera_left_data = camera_left_data.sample(n=min_length, random_state=42)
camera_belok_data = camera_belok_data.sample(n=min_length, random_state=42)

# Menggabungkan data menjadi satu dataset yang seimbang
data_balanced = pd.concat([camera_right_data, camera_left_data, camera_belok_data])

# Menghapus baris yang memiliki nilai NaN pada 'angular_velocity' atau 'linear_velocity'
data_balanced = data_balanced.dropna(subset=['angular_velocity', 'linear_velocity'])

# Mengonversi kolom 'angular_velocity' dan 'linear_velocity' menjadi tipe numerik
data_balanced['angular_velocity'] = pd.to_numeric(data_balanced['angular_velocity'], errors='coerce')
data_balanced['linear_velocity'] = pd.to_numeric(data_balanced['linear_velocity'], errors='coerce')

# Menghapus kembali baris yang memiliki nilai NaN setelah konversi
data_balanced.dropna(subset=['angular_velocity', 'linear_velocity'], inplace=True)

# Menyimpan nilai 'linear_velocity' ke dalam variabel
filtered_linear_velocity = data_balanced['linear_velocity'].values

# Function to load images and angular velocity with corrections for center, left, and right cameras
def load_img_angular_velocity(datadir, df):
    image_path = []
    angular_velocity = []
    original_angular_velocity = []
    linear_velocity = []
    
    for i in range(len(df)):
        indexed_data = df.iloc[i]
        image_file = indexed_data['image_filename']
        img_path = os.path.join(datadir, image_file.strip())
        
        # Load the original angle from the dataset
        angle = float(indexed_data['angular_velocity'])
        original_angular_velocity.append(angle)

        if "camera_belok_cepat" in image_file:
            if angle <= -0.3:
                angle += -0.05
            elif angle >= 0.3:
                angle += 0.05
        elif "camera_belok" in image_file:
            if angle <= -0.05:
                angle += -0.1
            elif angle >= 0.05:
                angle += 0.1
        
        # Apply corrections for each camera type
        if "camera_right" in image_file:
            angle += 0.3 
        elif "camera_left" in image_file:
            angle -= 0.3
        
        img = mpimg.imread(img_path)

        image_path.append(img_path)
        angular_velocity.append(angle)
        
    return np.asarray(image_path), np.asarray(angular_velocity), np.asarray(original_angular_velocity)

# Run load_img_angular_velocity
image_path, angular_velocities, original_angular_velocities = load_img_angular_velocity(
    'E:\data', data_balanced
)

# Split dataset
x_train, x_valid, y_train, y_valid, linear_train, linear_valid = train_test_split(
    image_path, angular_velocities, filtered_linear_velocity, test_size=0.2, random_state=6
)

# Visualize the split of the dataset into training and validation
train_data = pd.DataFrame({'image_filename': x_train, 'angular_velocity': y_train})
valid_data = pd.DataFrame({'image_filename': x_valid, 'angular_velocity': y_valid})

# Plot the distribution of angular velocities for training dataset
plt.figure(figsize=(10, 6))
plt.hist(train_data['angular_velocity'], bins=50, alpha=0.5, label='Training Set', color='blue')
plt.xlabel('Angular Velocity', fontsize=18)
plt.ylabel('Frequency', fontsize=18)
plt.title('Distribution of Angular Velocities in Training Set', fontsize=20)
plt.legend(loc='upper right', fontsize=16)
# plt.show()

# Plot the distribution of angular velocities for validation dataset
plt.figure(figsize=(10, 6))
plt.hist(valid_data['angular_velocity'], bins=50, alpha=0.5, label='Validation Set', color='orange')
plt.xlabel('Angular Velocity', fontsize=18)
plt.ylabel('Frequency', fontsize=18)
plt.title('Distribution of Angular Velocities in Validation Set', fontsize=20)
plt.legend(loc='upper right', fontsize=16)
# plt.show()

# Plot a pie chart for the split ratio
split_labels = ['Training Set', 'Validation Set']
split_sizes = [len(x_train), len(x_valid)]

plt.figure(figsize=(7, 7))
plt.pie(split_sizes, labels=split_labels, autopct='%1.1f%%', colors=['blue', 'orange'])
plt.title('Dataset Split (Training vs Validation)', fontsize=20)
# plt.show()

# Functions for preprocessing with augmentation
def random_brightness(image):
    # Pastikan gambar dalam format uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + (0.5 - np.random.rand()) * 0.5
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * ratio, 0, 255)  # Pastikan nilai tetap dalam rentang [0, 255]
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def darken_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + (0.2 - np.random.rand()) * 0.3  # Menggelapkan hingga 20%
    hsv[:,:,2] = hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def add_random_shadow(image):
    top_y = 320 * np.random.rand()
    top_x = 0
    bot_x = 160
    bot_y = 320 * np.random.rand()
    shadow_mask = 0 * image[:,:,1]
    X_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]
    shadow_mask[((Y_m - top_y)*(bot_x - top_x) - (bot_y - top_y)*(X_m - top_x) >= 0)] = 1
    if np.random.rand() > 0.5:
        random_bright = .5
        cond1 = shadow_mask == 1
        image[:,:,0][cond1] = image[:,:,0][cond1] * random_bright
        image[:,:,1][cond1] = image[:,:,1][cond1] * random_bright
        image[:,:,2][cond1] = image[:,:,2][cond1] * random_bright
    return image

def img_preprocess(img_path, angle):
    img = mpimg.imread(img_path)
    height = img.shape[0]
    img = img[height // 2 + 200:, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = darken_image(img)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255.0
    return img, angle

# Generator function for batching images and angles
def img_generator(image_paths, angular_velocities, linear_velocities, batch_size):
    while True:
        # Shuffle the data
        image_paths, angular_velocities, linear_velocities = shuffle(image_paths, angular_velocities, linear_velocities)

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_angles = angular_velocities[i:i + batch_size]
            batch_velocities = linear_velocities[i:i + batch_size]
            images = []
            velocities = []

            for img_path, velocity in zip(batch_paths, batch_velocities):
                # Read and preprocess image
                img, _ = img_preprocess(img_path, 0)  # You can replace '0' with appropriate angle if needed
                images.append(img)
                velocities.append(velocity)

            # Yield both the processed images and angular velocities (targets)
            yield [np.array(images), np.array(velocities)], np.array(batch_angles)

# Define CIL model with attention mechanism
def behavior_cloning():
    # Input untuk gambar
    image_input = Input(shape=(66, 200, 3), name='image_input')
    x = Conv2D(32, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu')(image_input)
    x = BatchNormalization()(x)

    for _ in range(5):
        kernel_size = (3, 3)
        stride = (2, 2) if (_ % 2 == 0) else (1, 1)
        x = Conv2D(64, kernel_size=kernel_size, strides=stride, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)

    # Mekanisme Attention
    attention = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    attention = Flatten()(attention)
    attention = Dense(128, activation='relu')(attention)
    attention = Dense(64, activation='relu')(attention)
    attention = Dense(1, activation='sigmoid')(attention)
    attention = Reshape((1, 1, 1))(attention)

    # Terapkan attention
    x = Multiply()([x, attention])
    x = Flatten()(x)

    # Input untuk kecepatan linear
    velocity_input = Input(shape=(1,), name='velocity_input')
    concatenated = Concatenate()([x, velocity_input])

    x = Dense(128, activation='relu')(concatenated)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    output = Dense(1, name='angular_velocity_output')(x)

    # Gabungkan model
    model = keras.Model(inputs=[image_input, velocity_input], outputs=output)

    # Kompilasi model
    optimizer = Adam(learning_rate=1e-4)
    model.compile(loss='mse', optimizer=optimizer)

    return model

model = behavior_cloning()
print(model.summary())

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train model with generator
history = model.fit(
    img_generator(x_train, y_train, linear_train, batch_size=32),
    validation_data=img_generator(x_valid, y_valid, linear_valid, batch_size=32),
    steps_per_epoch=len(x_train) // 32,
    validation_steps=len(x_valid) // 32,
    epochs=30,
    callbacks=[early_stopping]
)

# Save the model after training
model.save('model.h5')

# Adjust steps to include all data
steps_train = (len(x_train) + 64 - 1) // 64
steps_valid = (len(x_valid) + 64 - 1) // 64

# Predictions
train_preds = model.predict(img_generator(x_train, y_train, linear_train, batch_size=64), steps=steps_train)
valid_preds = model.predict(img_generator(x_valid, y_valid, linear_valid, batch_size=64), steps=steps_valid)

# Debugging lengths
print(f"y_train length: {len(y_train)}, train_preds length: {len(train_preds)}")
print(f"y_valid length: {len(y_valid)}, valid_preds length: {len(valid_preds)}")

# Adjust predictions to match the length of true labels
train_preds = train_preds[:len(y_train)]
valid_preds = valid_preds[:len(y_valid)]

# Calculate metrics
train_mse = mean_squared_error(y_train, train_preds)
valid_mse = mean_squared_error(y_valid, valid_preds)

train_mae = mean_absolute_error(y_train, train_preds)
valid_mae = mean_absolute_error(y_valid, valid_preds)

print(f"Train MSE: {train_mse}, Train MAE: {train_mae}")
print(f"Validation MSE: {valid_mse}, Validation MAE: {valid_mae}")

# Plotting loss curve
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss Curve')
plt.show()

# Plotting predictions vs actual values for validation
plt.scatter(y_valid, valid_preds, alpha=0.5)
plt.xlabel('True Angular Velocity')
plt.ylabel('Predicted Angular Velocity')
plt.title('Predicted vs Actual for Validation Data')
plt.show()
