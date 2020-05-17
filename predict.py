# -*- coding: utf-8 -*-
"""
Created on Fri May 15 23:08:32 2020

@author: Edgar
"""
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from PIL import Image
import tensorflow as tf
import tkinter as tk

#%%
datagen = tf.keras.preprocessing.image.ImageDataGenerator()
train_it = datagen.flow_from_directory('chest_xray/train/', class_mode='binary', batch_size=163, target_size = (64,64))
val_it = datagen.flow_from_directory('chest_xray/val/', class_mode='binary', batch_size=4, target_size = (64,64))
test_it = datagen.flow_from_directory('chest_xray/test/', class_mode='binary', batch_size=26, target_size = (64,64))



# load json and create model
json_file = open('model_cache/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_cache/model.h5")
print("Loaded model from disk")

#%%
# evaluate loaded model on test data
loaded_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
score = loaded_model.evaluate_generator(test_it, steps=24)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

#%%
#Predicting
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory("chest_xray/predict/",target_size=(64, 64),class_mode='binary',batch_size=1)


filenames = test_generator.filenames
nb_samples = len(filenames)

predict = loaded_model.predict_generator(test_generator,steps = nb_samples)

classified_as = predict[0][0] > 0.5 and "PNEUMONIA" or "NORMAL"
print(classified_as)


#%%
#Front-End Gui
from PIL import Image, ImageTk

window = tk.Tk()
image = Image.open("chest_xray/predict/pneumonia/sample.jpeg")
photo = ImageTk.PhotoImage(image)
label = tk.Label(image=photo)
label.image = photo # keep a reference!
label.pack()
image_label = tk.Label(text="Model %s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
image_label.pack()
image_label = tk.Label(text="Classified as %s with sigmoid value %.2f%%" % (classified_as, predict[0][0]))
image_label.pack()
window.mainloop()