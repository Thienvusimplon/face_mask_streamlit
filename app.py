import os
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp

model = tf.keras.models.load_model('model_transfer_crop.h5')
st.title("Face mask detection")

menu = ["Neural network"]
choice = st.sidebar.selectbox("Menu",menu)

if choice == "Neural network":
	st.subheader("Neural network")

elif choice == "Dataset":
	st.subheader("Dataset")

elif choice == "DocumentFiles":
	st.subheader("DocumentFiles")

def load_image(image_file):
	img = Image.open(image_file)
	
	return img


if choice == "Neural network":
	image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
	
	if image_file is not None:

		# To View Uploaded Image
		image_load = load_image(image_file)
		st.image(image_load)
		image = np.array(image_load)
		st.write(type(image))
		#image = cv2.imread('4.jpg')
		h, w, c = image.shape
		image_input = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		mp_face_detection = mp.solutions.face_detection
		face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
		mp_drawing = mp.solutions.drawing_utils
		face_detection_results = face_detection.process(image_input)
		#st.write(face_detection_results.detections)

		if face_detection_results.detections:
			for face_no, face in enumerate(face_detection_results.detections):

				st.write(f'FACE NUMBER: {face_no+1}')
				print('==============================')

				st.write(f'FACE CONFIDENCE: {round(face.score[0], 2)}')

				face_data = face.location_data

				print(f'nFACE BOUNDING BOX:n{face_data.relative_bounding_box}')
				
				box = face_data.relative_bounding_box
				
				xleft = int(box.xmin*w)
				xtop = int(box.ymin*h)
				xright = int(box.width*w + xleft)
				xbottom = int(box.height*h + xtop)
				
				detected_faces = [(xleft, xtop, xright, xbottom)]
				
				for n, face_rect in enumerate(detected_faces):
					face = Image.fromarray(image).crop(face_rect)
					face_np = np.asarray(face)
					
					img_size = 224

					new_array = cv2.resize(face_np, (img_size, img_size))
					#convert_image = cv2.cvtColor(new_array, cv2.COLOR_BGR2RGB)
					image_converti = np.expand_dims(new_array, axis=0)
					final_image = image_converti/255.0
					
					im = Image.fromarray(new_array)
					st.image(im)
					
					prediction = model.predict(final_image)
					print(prediction)
					if prediction > 0.000001:
						st.write("No mask")
					else:
						st.write("Mask")
		else:
			st.write("No face detected")
				
