import keras
import numpy as np
import cv2
from PIL import Image
import glob
from keras.layers import Dense,Dropout,Flatten,ConvLSTM2D,AveragePooling3D,Reshape,TimeDistributed,LSTM,Conv2D, MaxPooling2D,BatchNormalization, Lambda,GlobalAveragePooling2D
from keras.optimizers import SGD
import tensorflow as tf
from keras import layers, models, applications
from keras.models import Sequential,load_model
from keras import backend as K
from keras.utils import plot_model
from keras.applications.vgg16 import VGG16
from keras.applications import InceptionV3, VGG19
from sklearn.model_selection import train_test_split,KFold
import time
import os
from SMRS import SMRS
import os, random , sys
from playsound import playsound
import webbrowser
import PIL.ImageGrab

#loading existing model to test
def loading_model():
    model = load_model('model//model_for_5.h5')
    model.load_weights('model//model_weights_for_5.h5')
    return model

#remove previous saved files
def remove_exfiles():    
    for data_file in os.listdir('testing_video_frames//'):                
        os.remove('testing_video_frames//'+data_file)
    print("existing files removed!")

#capture video for testing
def capture_video():
    #capture video for 5 seconds
    capture_duration = 5

    cap = cv2.VideoCapture(0)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

    start_time = time.time()
    x, y, w, h = 300, 100, 300, 300
    while( int(time.time() - start_time) < capture_duration ):
        #print("started capturing")
        ret, frame = cap.read()        
        if ret==True:
            #frame = cv2.flip(frame,0)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            out.write(frame)
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()    

#converting video in to frames
def convert_to_frames():
    #reading testing video
    vidcap = cv2.VideoCapture('output.avi')
    success,image = vidcap.read()
    count = 0

    #save frames from video
    while success:    
        cv2.imwrite("testing_video_frames//%d.jpg" % count, image)     # save frame as JPEG file      
        success,image = vidcap.read()        
        count += 1


# taking frames and extracting representative frames using SMRF
def extract_frames():
    #take images one by one and sort it
    vid = []
    original = []
    count = 0     
    file_name_list = []
    for data_file in sorted(os.listdir('testing_video_frames//')):                
        file_name_list.append(int(os.path.splitext(data_file)[0]))

    file_name_list.sort()

    #take images in sorted order and preprocess it
    for filename in file_name_list:            
        image = cv2.imread('testing_video_frames//'+str(filename)+'.jpg')        
        if filename>35:            
            crop_image = image[100:400,300:600]
            ori = image[100:400,300:600]    
            gray_image = cv2.cvtColor(crop_image, cv2.COLOR_RGB2GRAY)    
            image = cv2.resize(gray_image,(320,240))        
            ori = cv2.resize(ori,(320,240))        
            vid.append(image.flatten())
            original.append(ori.flatten())
            
    # setting frames on the column and pixels on the rows    
    vid = np.transpose(np.asarray(vid))
    original = np.transpose(np.asarray(original))
    print(np.array(vid).shape,np.array(original).shape)

    print ('Problem size: [%d,%d]' % (vid.shape[0],vid.shape[1]))
    print ('Extracting the representatives frames from the video...It may takes a while...')
    #start_time = time.time()

    #call SMRF
    smrs = SMRS(data=vid,
                alpha=5,
                norm_type=1,
                verbose=True, thr=[10**-8],
                thrS=0.99,
                thrP=0.50,
                max_iter=5000,
                affine=True,
                normalize=False,
                step=1,
                PCA=False,
                GPU=False)

    sInd, repInd, C = smrs.smrs()
    repInd.sort()
    print(repInd)

    #take all representative frames
    vid = []
    for i in repInd:                        
        image = np.reshape(original[:,i],(240,320,3))                                   
        vid.append(image)

    #take random image frames and predict gesture
    vid = np.array(vid)
    x = np.random.choice(len(vid),5,replace= False)
    x = np.sort(x)
    print("samples extracted:",x)
    vid = vid[x]        
    vid = vid.reshape(1,5,240,320,3)
    print(vid.shape)
    vid = vid/255
    return vid

# predicting labels
def predict(model,vid):
    pred = model.predict(vid)
    print(pred)
    arg = np.argmax(pred)        
    return pred,arg

def rndmp3 ():
    randomfile = random.choice(os.listdir("C:/HARDDISK/songs/english"))
    file = "C:/HARDDISK/songs/english/" + randomfile
    playsound(file)


def task(arg):
    if arg == 0:
        url = 'https://web.whatsapp.com/'
        chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'

        whatsapp = webbrowser.get(chrome_path).open(url)
        
    elif arg == 1:
        url = 'https://www.google.com/maps/'
        chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'

        g_maps = webbrowser.get(chrome_path).open(url)

    elif arg == 2:
        im = PIL.ImageGrab.grab()     
        im.show()  

    elif arg == 3:
        url = 'https://www.facebook.com/'
        chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'

        fb = webbrowser.get(chrome_path).open(url)
        
    elif arg == 4:
        url = 'https://mail.google.com/'
        chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'

        g_mail = webbrowser.get(chrome_path).open(url)
        
    elif arg == 5:
        url = 'https://www.youtube.com/'
        chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'

        y_tube = webbrowser.get(chrome_path).open(url)
        
    elif arg == 6:
        os.system('calc.exe')
        
    elif arg == 7:
        os.system('notepad.exe')
        
    elif arg == 8:
        rndmp3 ()
    sys.exit()

def main():
    print("loading model")
    model  = loading_model()
    remove_exfiles()
    capture_video()
    convert_to_frames()
    vid = extract_frames()
    pred,arg = predict(model,vid)
    
    if arg == 0:
        print("Predicted Label:      Flat/Leftward    Class: 0")        
    elif arg == 1:
        print("Predicted Label:      Flat/Rightward    Class: 1")        
    elif arg == 2:
        print("Predicted Label:      Flat/Contract    Class: 2")        
    elif arg == 3:
        print("Predicted Label:      Spread/Leftward    Class: 3")        
    elif arg == 4:
        print("Predicted Label:      Spread/Rightward    Class: 4")        
    elif arg == 5:
        print("Predicted Label:      Spread/Contract    Class: 5")        
    elif arg == 6:
        print("Predicted Label:      V-shape/Leftward    Class: 6")        
    elif arg == 7:
        print("Predicted Label:      V-shape/Rightward    Class: 7")        
    elif arg == 8:
        print("Predicted Label:      V-shape/Contract    Class: 8")  
    task(arg)      
    sys.exit()


if __name__ == "__main__":
    main()