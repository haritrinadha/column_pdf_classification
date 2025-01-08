# This below code gives you out of n column format resumes, how many are predicted correctly and how many predicted wrong.
'''Changes to do before executing code
 In line 25,22 give the path to column format resumes
 In line 28 check if predicted_class[0] value equals 0 if you are testing for column format resumes
'''

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np,os

loaded_model = tf.keras.models.load_model('3layer_model_column_classification_10epoch.h5')
img_height, img_width=150, 150

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(img_height, img_width))  # Resize image
    img_array = image.img_to_array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (model expects 4D input)
    img_array /= 255.0  # Normalize the image (same as in ImageDataGenerator, which used for training)
    return img_array

# # labels are {'images_noncolumn': 0, 'images_noncolumn': 1}
files = os.listdir('test/images_column')
correct=wrong=0
for each_img in files:
    img_path = 'test/images_column/{}'.format(each_img)
    prepared_image = prepare_image(img_path)
    predictions = loaded_model.predict(prepared_image)
    predicted_class = np.argmax(predictions, axis=1)
    # if predicted_class[0]==1 for noncolumn resumes testing
    if predicted_class[0]==0:
        correct+=1
    else:
        print(img_path)
        wrong+=1
print('correct=',correct)
print('wrong=',wrong)