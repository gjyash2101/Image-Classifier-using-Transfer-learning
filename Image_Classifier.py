import streamlit as st
from PIL import Image
from skimage.io import imread
from skimage.transform import resize
import tensorflow as tf
import numpy as np
st.title("Chest X_ray Classifier")
st.write("Upload a chest x_ray image and the model will predict the disease that person can have!")
upload = st.file_uploader("Choose an Image")
model = tf.keras.models.load_model('chestx_ray.h5')
target = ['Atelectasis','Consolidation','Infiltration','Pneumothorax','Edema',"Emphysema",'Fibrosis',
          'Effusion',"Pneumonia","Pleural_thickening",'Cardiomegaly','Nodule', 'Mass','Hernia','No Finding']
if upload is not None:
  img = Image.open(upload)
  st.image(img,caption = 'UPLOADED IMAGE',use_column_width=True)

  if st.button('DETECT'):
    li=[]
    img = np.array(img)
    ar1 = resize(img,(224,224,3))
    li.append(ar1)
    li = np.array(li)
    y1 = model.predict(li)
    st.write("Disease is:")
    st.title(target[y1.argmax()])