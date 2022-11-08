#LSTM MANY TO MANY
#%%
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from keras.layers import BatchNormalization as BatchNorm
from keras.layers.core import *
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from numpy import array, concatenate
from keras.callbacks import ModelCheckpoint
from math import ceil, log10
from keras.utils import np_utils
from matplotlib import pyplot

#%%

#Read the csv file
file_in = "datasets/movil_18102022.csv"

def parse_database():
    trainset = []
    with open(file_in) as archivo:
        lineas = archivo.read().splitlines()
    for l in lineas:
      linea = l.split(' ')
      floats = [float(i) for i in linea]
      trainset.append(floats)
    return trainset

data = parse_database()
min_max_scaler = MinMaxScaler()
data_scaled = min_max_scaler.fit_transform(data)
print(data_scaled[0:3])
data = np.round(data_scaled, 1)
print(data[0:3])
#print(df_for_training_scaled)

#%%
#for testing

file_test_in = 'datasets/movil_II_09102022_violin_solo.csv'

def parse_database():
    trainset = []
    with open(file_test_in) as archivo:
        lineas = archivo.read().splitlines()
    for l in lineas:
      linea = l.split(' ')
      floats = [float(i) for i in linea]
      trainset.append(floats)
    return trainset

testset = parse_database()
min_max_scaler = MinMaxScaler()
testset_scaled = min_max_scaler.fit_transform(testset)
print(testset_scaled[0:3])
testset = np.round(testset_scaled, 1)
print(testset[0:3])
#print(df_for_training_scaled)


#%%
n_steps = 100

def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        #print("lenseq", len(sequences))
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

X, y = split_sequences(data, n_steps=n_steps)

print(X[0:1])
print(y[0:10])
#print(X.shape, y.shape)
#print(X[0:5])
#print(y[0:5])

#convert to one hot encode
print("yshape", y.shape)

y = (y*10) #move decimal for categorical transform
#y = y.astype(int)
print("Y", (y[0:5]))
y_categorical = np_utils.to_categorical(y)
print(y_categorical[1])
print(y_categorical.shape)

#%%

#for testset

n_steps = 100

X_test, y_test = split_sequences(testset, n_steps=n_steps)

print(X_test[0:1])
print(y_test[0:10])
#print(X.shape, y.shape)
#print(X[0:5])
#print(y[0:5])

#convert to one hot encode
print("yshape", y_test.shape)

y_test = (y*10) #move decimal for categorical transform
#y = y.astype(int)
print("Y", (y[0:5]))
y_test_categorical = np_utils.to_categorical(y)
print(y_test_categorical[1])
print(y_test_categorical.shape)

#%%
def create_network(network_input, neurons):
    """ create the structure of the neural network """
    model = Sequential()
    neurons = neurons
    model.add(LSTM(
        neurons,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        #recurrent_dropout=0.5,
        return_sequences=False
    ))
    model.add(RepeatVector(network_input.shape[2]))
    model.add(LSTM(neurons, return_sequences=True))
    #model.add(LSTM(50, return_sequences=True))
    model.add(TimeDistributed(Dense(y_categorical.shape[2], activation='softmax')))
    #model.add(Dense(y_categorical.shape[2]))
    model.compile(loss='categorical_crossentropy' , optimizer='adam', metrics=[ 'accuracy' ])
    print("Debuging", network_input.shape)
    print("Debuging", y_categorical.shape)
    return model

def train(model, network_input, network_output, epochs, batch_size):
    model.summary()
    history = model.fit(network_input, network_output, epochs=epochs, batch_size=batch_size)

def train_and_save(model, network_input, network_output, epochs, batch_size):
    """ train the neural network """
    filepath = "saved_models/weights/semanadbv_{epoch:02d}_{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]
    model.summary()
    #history = model.fit(network_input, network_output, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list) #, steps_per_epoch=100, validation_steps=20)
    #history = model.fit(network_input, network_output, epochs=epochs, batch_size=batch_size, validation_split=0.33)
    history = model.fit(network_input, network_output, epochs=epochs, batch_size=batch_size,validation_data=(network_input_test, network_output_test))
   # print(history.history['loss'])
   # print(history.history['accuracy'])
   # print(history.history['val_loss'])
   # print(history.history['val_accuracy'])
    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('model train vs validation loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.show()
    loss, acc = model.evaluate(network_input, network_output, verbose=0)
    print( "Loss: %f, Accuracy: %f" % (loss, acc))

#print(X.shape, y.shape)
model = create_network(X, neurons=128)
train_and_save(model, X, y_categorical, epochs=5,  batch_size=128)

#%%

X, y = generate_data(100, n_terms, largest, alphabet)
loss, acc = model.evaluate(X, y, verbose=0)
print( Loss: %f, Accuracy: %f % (loss, acc*100))




#%%
#Predicting...
#inputs = X[0:800]
inputs = X[0:100]
#print(inputs)
prediction = model.predict(inputs, verbose=0) #shape = (n, 1) where n is the n_days_for_prediction

#print(prediction)
#prediction = np.max(prediction)
#print (prediction)

def set_zero(sample, d, val):
    """Set all max value along dimension d in matrix sample to value val."""
    argmax_idxs = sample.argmax(d)
    #idxs = [np.indices(argmax_idxs.shape)[j].flatten() for j in range(len(argmax_idxs.shape))]
    #idxs.insert(d, argmax_idxs.flatten())
    #sample[idxs] = val
    return argmax_idxs*0.1 #this can vary depends on previos decimal transform

prediction = set_zero(prediction, d=2, val=0)

print (prediction)
#%%

#print(data[8:16])
#print(data[0:20])
#plt.plot(data[100:200])
plt.plot(prediction)
plt.ylabel('Clases')
plt.xlabel('Iteraciones')
plt.savefig('movilTDdata.png')
plt.show()

#train_network()
#%%
