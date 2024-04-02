import tensorflow as tf
from tensorflow.keras.preprocessing import image
from keras import layers
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
from tqdm import tqdm  #we'll get to know the progrsss as it will take the procedure slow

filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load VGGFace model
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

def extractor(img_path,model):
    img = image.load_img(img_path,target_size=(224,244))
    img_array = image.img_to_array(img)
    img1 = np.expand_dims(img_array,axis=0)   # if img_array is a 3-dimensional array representing a single image with shape (height, width, channels), np.expand_dims(img_array, axis=0) will add an extra dimension at the beginning, resulting in a 4-dimensional array with shape (1, height, width, channels)
    img_processed = preprocess_input(img1)    # in the context of image classification models, preprocessing might involve mean subtraction to center the pixel values around zero and scaling to normalize them to a certain range

    result = model.predict(img_processed).flatten()

feature = []

for file in tqdm(filenames):
    feature.append(extractor(file,model))

pickle.dump(feature,open('featuresextracted.pkl','wb'))