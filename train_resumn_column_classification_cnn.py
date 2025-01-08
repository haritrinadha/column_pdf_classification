import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pdf2image import convert_from_path
import os,shutil

img_height = 150
img_width = 150
# small bath_size will help model do well with unseen data
batch_size = 8

''' all the train and test data should be in below Directory structure only!!. and they should contain .jpg format images only in them(1st page of pdf should be converted into image and saved in this folder before training start)
--/train
------/column
------/noncolumn
--/test
------/column
------/noncolumn
'''

train_dir = './train'
test_dir = './test'

# Load and preprocess the images using ImageDataGenerator
# Create an ImageDataGenerator for the training and test sets
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Flow the images from the directory with augmentation for the training data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'  # Use 'categorical' for categorical classification and 'binary' for binary classification
)

# Flow the images from the directory for testing
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'  # Use 'categorical' for categorical classification and 'binary' for binary classification
)

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(train_generator.num_classes, activation='softmax')  # Output layer for multi-class classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Use 'categorical_crossentropy' for category classification
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size
)

# the folder names in train directory are taken as labels to classify.
# check below print statement for numeric value of categorical label
print(train_generator.class_indices)

model.save("3layer_model_column_classification_10epoch.h5", include_optimizer=True)

# #use below code to plot accuracy
# Plot training history (optional)
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.legend()
# plt.show()

# #use below code to convert pdf files to training required format
# def img_convert():
#     files = os.listdir('/home/Desktop/issue_resumes')
#     for root,subdir,files in os.walk('/home/Desktop/issue_resumes'):
#         for each_file in files:
#             src = root+'/'+each_file
#             dst = '/home/Desktop/issue_resumes_image/'+(each_file.split('.')[0])+'.jpg'
#             print(dst)
#             images = convert_from_path(src)
#             images[0].save(dst, 'JPEG')

