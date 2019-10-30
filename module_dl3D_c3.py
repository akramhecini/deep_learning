import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import os
from keras.models import Sequential
from keras.layers import Convolution3D, MaxPooling3D, Dense, Activation, Dropout, Flatten, LeakyReLU
import keras.utils.np_utils
import random
import glob
from pathlib import Path
from keras.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot
import pandas as pd
import scikitplot as skplt



def make_shape(n_length,pathy):

    """A function that takes data size and a path to data used in the model to create a shape for npy arrays

	- n_length : Data Size the user is willing to use for training, validation and test combined. int value. 
	- pathy : Path to the data.  
    """

    os.chdir(pathy)

    sample_complex = random.choice([f for f in os.listdir(pathy) if not f.startswith('.')])

    print(sample_complex)

    new_shape = (n_length,)+ np.load(sample_complex).shape

    return(new_shape)


def find_class(complex,heme,nucleotide,control, steroid):


    """
A function that takes a proteine-ligand complex (one .npy file) and tells the user its class

- complexe : npy file of the protein-ligand complex 

- heme,nucleotide,control, steroid : 4 lists that contain the names of all ligand-protein complexes used. 

    """

    if complex in heme:
        return 0
    elif complex in nucleotide:
        return 1
    elif complex in control :
        return 2
    elif steroid in control :
        return 3


def data_sample(complexe_list, taille):

    """
A function that takes a proteine-ligand complex list of names and a number and returns a smaller list of protein-ligand complex. 

- complexe_list : a list of names of all protein-complexes of a certain class

- taille : the size of the smaller list of names. 

    """

    indices = random.sample(range(len(complexe_list)), taille)

    complex_file_names = [complexe_list[i] for i in indices]

    return(complex_file_names)



def Resize_data(data_size):


    """
A Function that takes the data size then split it into 3 parts of the three different classes. 

- data_Size : is the data size used for training, validation and test combined. 

    """

    data_size = data_size

    str_taille = 0

    data_size_n = data_size - str_taille

    taille = data_size_n//3

    reste = data_size_n - (taille*3)

    taille_sup = taille+reste

    return (taille_sup,taille)




def read_lists(path_to_lists, taille,taille_sup):


    """
A Function that takes a path to files names, and the size of the 3 lists that will be generated. 

a list for heme, a list for nucleotide and a list of control, each list has a size that's been decided by calling the function Resize_data

- Path_to lists : a path to the lists that contains the file names

- taille = the size of control and nucletide list

- taille_supp : the size of heme list. 

Usually taille and taille_sup are different when the data size % 3 is 1 

    """

    os.chdir(path_to_lists)

    heme = [i.strip() for i in open("heme.list").readlines()]
    heme_sample = data_sample(heme,taille_sup)

    steroid = [i.strip() for i in open("steroid.list").readlines()]
    steroid_sample = data_sample(steroid,len(steroid))

    nucleotide = [i.strip() for i in open("nucleotide.list").readlines()]
    nucleotide_sample = data_sample(nucleotide,taille)

    control = [i.strip() for i in open("control.list").readlines()]
    control_sample = data_sample(control,taille)

    #data_total = heme_sample+nucleotide_sample+control_sample+steroid_sample

    data_total = heme_sample+nucleotide_sample+control_sample
    return data_total,heme,steroid,nucleotide,control






def create_data(data_size,heme, nucleotide, control, steroid,data_total,path_to_data):

    """
A funtion that create the data used in the training, validation and test

- data_size : is the data size used. 
- heme, nucleotide, control, steroid : lists that contains the names of group of file that are included in a ceratin class.
- data_total: a list that contains the names of all files used. 
- path_to_data: the path leading to the data used. 

    """

    os.chdir(path_to_data)

    x_array = np.zeros(shape = (data_size,14,32,32,32))

    y_array = np.zeros(shape = data_size)

    print("data size = ", data_size)

    #training set :

    file_count = 0

    for file in data_total:

        y_array[file_count]= find_class(str(file), heme, nucleotide, control, steroid)

        x_array[file_count] = np.load(str(file+".npy"))

        file_count+=1


    return (x_array, y_array)



#splitting data :


def split_data(x_array,y_array):

    """
A function that splits data into training, validation and test
    """
    x_train, x_test, y_train, y_test = train_test_split(x_array, y_array, test_size=0.2 , shuffle=True)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, shuffle = True)

    np.save('x_train',x_train)
    np.save('y_train',y_train)
    np.save('x_test',x_test)
    np.save('y_test',y_test)
    np.save('x_val',x_val)
    np.save('y_val',y_val)

    #vald_array = keras.utils.np_utils.to_categorical(y_val, 4)

    #n_y_array = keras.utils.np_utils.to_categorical(y_train, 4)

    return(x_train, x_test, y_train, y_test, x_val, y_val)



###############



def create_model(new_shape):

    """
A function that creates the model, it takes the input shape and returns a model
    """
    model = Sequential()
    # Conv layer 1
    model.add(Convolution3D(
        input_shape = new_shape[1:],
        filters=64,
        kernel_size=6,
        data_format='channels_first',
    ))

    #model.add(LeakyReLU(alpha = 0.1))

    # Dropout 1
    model.add(Dropout(0.2))
    # Conv layer 2
    model.add(Convolution3D(
        filters=64,
        kernel_size=3,
        padding='valid',     
        data_format='channels_first',
    ))

    #model.add(LeakyReLU(alpha = 0.1))

    # Maxpooling 1
    model.add(MaxPooling3D(
        pool_size=(2,2,2),
        strides=None,
        padding='valid',    
        data_format='channels_first'
    ))
    # Dropout 2
    model.add(Dropout(0.4))
    # FC 1
    model.add(Flatten())
    model.add(Dense(128))

    #model.add(LeakyReLU(alpha = 0.1))
    # Dropout 3
    model.add(Dropout(0.4))
    # Fully connected layer 2 to shape (2) for 2 classes

    model.add(Dense(3))
    model.add(Activation('softmax'))



    return model

# Set callback functions to early stop training and save the best model so far

def train_model(model,x_train, n_y_array,x_val, vald_array, Epochs_size, Batch_size):

    """
A function that trains the model
    """

    cb = EarlyStopping(monitor='val_loss', mode = "min", patience = 5 , verbose=1)

    mc = ModelCheckpoint(filepath='model_C3_15b.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)

    print("training has started")

    history = model.fit(x_train, n_y_array, callbacks = [cb,mc], validation_data=(x_val, vald_array),

              batch_size=Batch_size, nb_epoch=Epochs_size)

    return(history)


# evaluate the model

def model_evaluate(model,x_train,n_y_array,x_val, vald_array):

    """
A function that evaluates the model
    """

    scores = model.evaluate(x_train, n_y_array, verbose=1)

    scores2 = model.evaluate(x_val, vald_array, verbose=1)


    print("for traininf set")

    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    print("%s: %.2f%%" % (model.metrics_names[0], scores[0]))



    print("for validation set : ") 

    print("%s: %.2f%%" % (model.metrics_names[1], scores2[1]*100))

    print("%s: %.2f%%" % (model.metrics_names[0], scores2[0]))


def plot_metrics(history):

    """
A function that plots the loss and val_loss functions
    """

    pyplot.plot(history.history['loss'], label='loss')

    pyplot.plot(history.history['val_loss'], label='val_loss')

    pyplot.legend()

    pyplot.show()


def model_predict(model,x_test,y_test):

    """
A function that predicts the class of protein-ligand complexes
    """


    y_pred = model.predict(x_test)

    predict_class = np.argmax(y_pred, axis=1)

    predict_class = predict_class.tolist()

    return(y_pred,predict_class)
