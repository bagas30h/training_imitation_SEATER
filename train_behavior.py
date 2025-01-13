import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, Dropout, Flatten, Dense, BatchNormalization, InputLayer, Attention, Reshape, Multiply
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

# No balancing between camera_right_data and camera_left_data
center_camera_data = data[~data['image_filename'].str.contains("camera_right|camera_left|camera_belok")]

# Gabungkan data yang sudah diseimbangkan
data_balanced = pd.concat([camera_right_data, camera_left_data, camera_belok_data, center_camera_data]) 


# Function to load images and angular velocity with corrections for center, left, and right cameras
def load_img_angular_velocity(datadir, df):
    image_path = []
    angular_velocity = []
    original_angular_velocity = []
    
    for i in range(len(df)):
        indexed_data = df.iloc[i]
        image_file = indexed_data['image_filename']
        img_path = os.path.join(datadir, image_file.strip())
        
        # Load the original angle from the dataset
        angle = float(indexed_data['angular_velocity'])
        original_angular_velocity.append(angle)

        if "camera_belok" in image_file:
            if angle <= -0.05:
                angle += -0.1
            elif angle >= 0.05:
                angle += 0.1
        
        # Apply corrections for each camera type
        if "camera_right" in image_file:
            angle += 0.3
        elif "camera_left" in image_file:
            angle -= 0.3
        
        # If the image was flipped, read and flip the image
        img = mpimg.imread(img_path)
        
        image_path.append(img_path)
        angular_velocity.append(angle)
        
    return np.asarray(image_path), np.asarray(angular_velocity), np.asarray(original_angular_velocity)

# Run load_img_angular_velocity
image_path, angular_velocities, original_angular_velocities = load_img_angular_velocity(
    'C:\\Users\\bagas30h\\Documents\\GitHub\\behavior_cloning_real_world', data_balanced
)

# Split dataset
x_train, x_valid, y_train, y_valid, orig_train, orig_valid = train_test_split(
    image_path, angular_velocities, original_angular_velocities, test_size=0.2, random_state=6
)

# Functions for preprocessing with augmentation
def random_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + (0.5 - np.random.rand()) * 0.5
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
    img = img[height // 2 + 200:, :, :]  # Crop bagian bawah gambar
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)  # Ubah ke ruang warna YUV
    
    # Simulasi ND Filter dengan mengurangi kecerahan seluruh gambar secara proporsional
    factor = 0.1  # Misalnya 0.5 untuk efek ND filter (0 untuk hitam, 1 untuk aslinya)
    img[:, :, 0] = img[:, :, 0] * factor  # Kanal Y (luminance)
    
    # Terapkan CLAHE untuk meningkatkan kontras setelah penggelapan
    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
    img[:, :, 0] = clahe.apply(img[:, :, 0])  # Terapkan CLAHE ke kanal Y
    
    img = cv2.cvtColor(img, cv2.COLOR_YUV2RGB)  # Kembalikan ke RGB
    img = cv2.GaussianBlur(img, (3, 3), 0)  # Gaussian blur untuk smoothing
    img = cv2.resize(img, (200, 66))  # Ubah ukuran gambar ke 200x66

    # img = (img * 255).astype(np.uint8)  # Ubah kembali ke tipe uint8
    cv2.imwrite('a.jpg', img)
    return img, angle

# Generator function for batching images and angles
def img_generator(image_paths, angular_velocities, batch_size):
    while True:
        image_paths, angular_velocities = shuffle(image_paths, angular_velocities)
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_angles = angular_velocities[i:i + batch_size]
            images = []
            angles = []
            for img_path, angle in zip(batch_paths, batch_angles):
                img, corrected_angle = img_preprocess(img_path, angle)
                images.append(img)
                angles.append(corrected_angle)
            yield np.array(images), np.array(angles)

# Define CIL model with attention mechanism
def cil_model_improved_with_attention():
    model = Sequential()
    model.add(InputLayer(input_shape=(66, 200, 3)))
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'))
    model.add(BatchNormalization())
    
    for _ in range(5):
        kernel_size = (3, 3)
        stride = (2, 2) if (_ % 2 == 0) else (1, 1)
        model.add(Conv2D(64, kernel_size=kernel_size, strides=stride, padding='same', activation='relu'))
        model.add(BatchNormalization())
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(1))

    optimizer = Adam(learning_rate=1e-4)
    model.compile(loss='mse', optimizer=optimizer)
    return model

model = cil_model_improved_with_attention()
print(model.summary())

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train model with generator
history = model.fit(
    img_generator(x_train, y_train, batch_size=64), 
    steps_per_epoch=len(x_train)//64,
    epochs=30,
    validation_data=img_generator(x_valid, y_valid, batch_size=64),
    validation_steps=len(x_valid)//64,
    callbacks=[early_stopping]
)

# Save the model after training
model.save('model.h5')

# Adjust steps to include all data
steps_train = (len(x_train) + 64 - 1) // 64
steps_valid = (len(x_valid) + 64 - 1) // 64

# Predictions
train_preds = model.predict(img_generator(x_train, y_train, batch_size=64), steps=steps_train)
valid_preds = model.predict(img_generator(x_valid, y_valid, batch_size=64), steps=steps_valid)

# Debugging lengths
print(f"y_train length: {len(y_train)}, train_preds length: {len(train_preds)}")
print(f"y_valid length: {len(y_valid)}, valid_preds length: {len(valid_preds)}")

# Adjust y_train and y_valid if necessary
y_train = y_train[:len(train_preds)]
y_valid = y_valid[:len(valid_preds)]

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
