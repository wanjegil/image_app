from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import streamlit as st

st.title('이미지 분류기')
st.write('고양이 또는 강아지 사진을 업로드하세요.')

np.set_printoptions(suppress=True)
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

###################################
# 사용자로부터 이미지 업로드 받기
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    st.image(image) # 업로드한 이미지를 출력

    st.write("Class:", class_name[2:])
    st.write("Confidence Score:", confidence_score)
