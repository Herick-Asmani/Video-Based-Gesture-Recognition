import keras
import numpy as np
import cv2
#from PIL import Image
import glob
from keras.layers import Dense,Dropout,Flatten,ConvLSTM2D,AveragePooling3D,Reshape,TimeDistributed,LSTM,Conv2D, MaxPooling2D,BatchNormalization, Lambda,GlobalAveragePooling2D , Activation
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
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint 
from sklearn.metrics import precision_recall_fscore_support as score

def collect_data():

	#gathering data from folders
	set1 = []
	vid=[]
	all_vids = []
	X = []
	y=[]
	labs = []
	
	print("***************************************Collecting all videos to train and test. This may takes a while*************************************************")
	#for each class labels
	for label in range(9):	    
	    label_str = "000"+str(label)
	    print("For Label:",label)
	    #for each set of perticular label in the given data
	    for Set in range(1,6):	        
	        set_str= "Set"+str(Set)
	        
	        #for each video of perticular set and label
	        for video in range(20):
	            
	            if(video<10):
	                video_str = "000"+str(video)
	            else:
	                video_str = "00"+str(video)
	            
	            #for each images for the video of perticular set and label
	            original_video = []
	            for filename in glob.glob('data/'+set_str+'/'+label_str+'/'+video_str+'/*.jpg'):    
	                image = cv2.imread(filename)                
	                original_video.append(image.flatten())
	                gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	                vid.append(gray_image.flatten())

	            original_video = np.transpose(np.asarray(original_video))
	            vid = np.transpose(np.asarray(vid))

	            print ('Extracting the representatives frames from the video using SMRF technique...')
	            start_time = time.time()
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
	            
	            vid = []
	            for i in repInd:                        
	                image = np.reshape(original_video[:,i],(240,320,3))                                
	                vid.append(image)
	            
	            vid = np.array(vid)	            
	            x = np.random.choice(len(vid),5,replace= False)
	            x = np.sort(x)
	            print("Representative frames extracted from the video:",x)
	            vid = vid[x]                        
	            #labs.append([str(label)]*len(vid))
	            
	            all_vids.append(vid)    
	            vid = []        
	    X.append(all_vids)
	    #y.append(labs)
	    y.append([str(label)]*len(all_vids))	    
	    print(y)    
	    #labs = []
	    all_vids = []
	    	    
	#converting list in to numpy arrays    
	X = np.array(X)
	y = np.array(y)
	#reshaping X and Y 
	X = X.reshape(900,5, 240, 320, 3)
	y = y.reshape(900)

	#saving dataset in numpy files
	np.save("X_data5.npy", X)
	np.save("y_data5.npy",y)
	print("X shape:",X.shape)
	print("y shape:",y.shape)	
	return X,y

def prepare_data():
	X,y = collect_data()
	print("****************************************************** Data Collected ****************************************************************")
	print()
	print()
	print("******************************************************* Pre Processing Data ***********************************************************")
	y = onehot(y)
	X,y = shuffle_data(X,y)
	X = normalize(X)
	X_train,X_val,X_test,y_train,y_val,y_test = split_data(X,y)	
	return X_train,X_val,X_test,y_train,y_val,y_test


def normalize(X):
	# Normalizing the RGB codes by dividing it to the max RGB value.
	X = X.astype('float32')
	X /= np.std(X, axis = 0)	
	return X


def split_data(X,y):
	
	print("splitting data into 700 training samples, 100 validation and 100 testing samples")	
	#splitting datasets in to train and test
	X_train, X_val, X_test  = X[:700, ...],X[700:800,...], X[800:, ...]
	y_train, y_val, y_test = y[:700, ...], y[700:800,...], y[800:, ...]
	#X_train,X_test  = X[:700, ...], X[700:, ...]
	#y_train,y_test = y[:700, ...], y[700:, ...]
	return X_train,X_val,X_test,y_train,y_val,y_test

def shuffle_data(X,y):
	#Shuffling datasets
	s = np.arange(X.shape[0])
	np.random.shuffle(s)	
	X = X[s]
	y = y[s]	
	return X,y


def onehot(y):
	#one hot encoding of class labels
	y = keras.utils.to_categorical(y,9)
	return y


def lrcn(X_train,X_val,X_test,y_train,y_val,y_test):

	print("**************************************************** Training Model. This may take some time!!*****************************************")		
	# LRCN model
	model = Sequential()

	model.add(TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2),
	    padding='same'), input_shape=(5,240,320,3)))
	model.add(TimeDistributed(Activation('relu')))
	model.add(TimeDistributed(Conv2D(32, (3,3),
	     padding='same')))
	model.add(TimeDistributed(Activation('relu')))
	model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

	model.add(TimeDistributed(Conv2D(64, (3,3),
	    padding='same')))
	model.add(TimeDistributed(Activation('relu')))
	model.add(TimeDistributed(Conv2D(64, (3,3),
	    padding='same')))
	model.add(TimeDistributed(Activation('relu')))
	model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

	model.add(TimeDistributed(Conv2D(128, (3,3),
	    padding='same')))
	model.add(TimeDistributed(Activation('relu')))
	model.add(TimeDistributed(Conv2D(128, (3,3),
	    padding='same')))
	model.add(TimeDistributed(Activation('relu')))
	model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

	model.add(TimeDistributed(Conv2D(256, (3,3),
	    padding='same')))
	model.add(TimeDistributed(Activation('relu')))
	model.add(TimeDistributed(Conv2D(256, (3,3),
	    padding='same')))
	model.add(TimeDistributed(Activation('relu')))
	model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

	model.add(TimeDistributed(Conv2D(512, (3,3),
	    padding='same')))
	model.add(TimeDistributed(Activation('relu')))
	model.add(TimeDistributed(Conv2D(512, (3,3),
	    padding='same',)))
	model.add(TimeDistributed(Activation('relu')))
	model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

	model.add(TimeDistributed(Flatten()))

	model.add(Dropout(0.5))
	model.add(LSTM(256, return_sequences=False, dropout=0.5))
	model.add(Dense(9, activation='softmax'))

	model.compile(
	        loss='categorical_crossentropy',
	        optimizer='adam',
	    metrics=['accuracy']
	)
	print(model.summary())
	checkpointer = ModelCheckpoint(filepath = 'model_weights_for_5.h5', verbose = 1, save_best_only = True)

	history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val,y_val), callbacks = [checkpointer], verbose = 1)	
	save_model(model)
	
	return history,model

def save_model(model):		
	#save model and weights
	model.save('model_for_5.h5')
	#model.save_weights('model//model_weights_for_5.h5')
	print("model saved successfully")

def evaluate_model(X_test,y_test,model):
	print("Evaluating model testing data:")
	history = model.evaluate(X_test,y_test)	
	print(" model Evaluated")
	print(history)

	z = X_test.reshape(100,5,240,320,3)
	classes = model.predict(z)
	ytest = []
	classe = []
	for i in range(len(X_test)):
	    ytest.append(np.argmax(y_test[i]))
	    classe.append(np.argmax(classes[i]))	

	print("************************ precision , recall, f1score, support on unseen test dataset *******************************")
	precision, recall, fscore, support = score(ytest, classe)

	print('precision: {}'.format(precision))
	print('recall: {}'.format(recall))
	print('fscore: {}'.format(fscore))
	print('support: {}'.format(support))

	
def plot_graphs(history):
	# Plot graphs for training loss and accuracies
	import matplotlib.pyplot as plt

	# summarize history for accuracy
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'val'], loc='upper left')
	plt.show()
	# summarize history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'val'], loc='upper left')
	plt.show()
	

def main():
	X_train,X_val,X_test,y_train,y_val,y_test = prepare_data()
	print("************************************************************ Data Prepared *****************************************************************")
	print()
	history_train,model = lrcn(X_train,X_val,X_test,y_train,y_val,y_test)
	print(history_train)
	print("Plotting loss and accuracy graphs: ")
	plot_graphs(history_train)
	evaluate_model(X_test,y_test,model)
	
	print("************************************* program ended. You can run test.py file for gesture recognition ******************************************")

if __name__ == "__main__":
	main()

