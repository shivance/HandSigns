import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import h5py
from colab import *

model = tf.keras.models.load_model('model.h5')

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} wr itten!".format(img_name))
        


        img = plt.imread(img_name)

        imgp = cv2.resize(img,(64,64))
        imgp = imgp.reshape(1,64,64,3)
        imgp1 = np.array(imgp,dtype='float64')
        print(model.predict(imgp1))
        print("\n\n\n"+str(np.argmax(model.predict(imgp1)))+"\n\n\n")
        #print("\n\nTHE DIGIT IS "+str(np.argmax(model.predict(imgp1))))    

        img_counter += 1

cam.release()

cv2.destroyAllWindows()
