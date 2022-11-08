#%%
import numpy as np
import matplotlib.pyplot as plt
import soundcard as sc
from struct import unpack
#from IPython import display
from essentia.streaming import *
from essentia import Pool, run, array, reset
from scipy.special import softmax
from essentia import INFO
import toolz as tz
from functools import reduce

from pydub import AudioSegment
from pydub.playback import play

#OSC libs
import argparse
import math
import requests # importing the requests library
from pythonosc import dispatcher
from pythonosc import osc_server
from pythonosc import udp_client
import json
from sklearn.preprocessing import MinMaxScaler
from essentia.streaming import FlatnessSFX as sFlatnessSFX

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout

client = udp_client.SimpleUDPClient('127.0.0.1', 5008) #this client sends to SC

#%%
min_max_scaler = MinMaxScaler()
sampleRate = 44100
frameSize = 2048
hopSize = 2048
numberBands = 13
onsets = 1
loudness_bands = 1
# analysis parameters
patchSize = 1  #control the velocity of the extractor 20 is approximately one second of audio
bufferSize = patchSize * hopSize
buffer = np.zeros(bufferSize, dtype='float32')
vectorInput = VectorInput(buffer)
frameCutter = FrameCutter(frameSize=frameSize, hopSize=hopSize)
w = Windowing(type = 'hann')
spec = Spectrum()
mfcc = MFCC(numberCoefficients=numberBands)
fft = FFT() # this gives us a complex FFT
c2p = CartesianToPolar()
onset = OnsetDetection()
eqloud = EqualLoudness()
flatness = Flatness()
envelope = Envelope()
accu = RealAccumulator()
loudness = Loudness()
complexity = SpectralComplexity()
centroid = Centroid()
square = UnaryOperator(type='square')

#load_model = "saved_models/weights/movil_FLC_100TS-64N_322_0.0284.hdf5"
#load_model = "Models/derek_bailey_electric_FLCMFCC_100TS-64N_242_0.0006.hdf5"
#load_model = "Models/Bailey_FLCMFCC_20TS-256N_242_0.0002.hdf5"
#load_model = "Models/Bailey_FLCMFCC_500TS-256N_358_0.0000.hdf5"
#json_in = "datasets/derek_bailey_electric-FCLMFCCs.json"

#load_model = "Models/senadbv_50TS-128N_1038_0.1012.hdf5"
#json_in = "datasets/semanadbv.json"

load_model = "Models/movil_18102022_708_0.0108.hdf5"
json_in = "datasets/movil_18102022.json"

def multifeaturesExtractor():
    pool = Pool()

    vectorInput.data  >> frameCutter.signal # al centro puede ir un >> eqloud.signal
    frameCutter.frame >> w.frame >> spec.frame
    spec.spectrum     >> flatness.array
    frameCutter.frame >> loudness.signal
    spec.spectrum     >> centroid.array #al centro puede ir un >> square.array que cambios genera?
    spec.spectrum     >> mfcc.spectrum
    #spec.spectrum     >> complexity.spectrum

    flatness.flatness >> (pool, 'flatness')
    loudness.loudness >> (pool, 'loudness')
    centroid.centroid >> (pool, 'centroid')
    mfcc.mfcc         >> (pool, 'mfcc')
    mfcc.bands        >> None

    #complexity.complexity >> (pool, "spectral complexity")

    #w.frame           >> fft.frame
    #fft.fft           >> c2p.complex
    #c2p.magnitude     >> onset.spectrum
    #c2p.phase         >> onset.phase
    #onset.onsetDetection >> (pool, 'onset')

    return pool

pool = multifeaturesExtractor()

def callback(data):
    # update audio buffer
    buffer[:] = array(unpack('f' * bufferSize, data))
    #print ("this is the buffer", buffer[:])
    flatnessBuffer = np.zeros([1])
    loudnessBuffer = np.zeros([1])
    centroidBuffer = np.zeros([1])
    mfccBuffer = np.zeros([numberBands])
    #onsetBuffer = np.zeros([onsets])

    reset(vectorInput)
    run(vectorInput)

    flatnessBuffer = np.roll(flatnessBuffer, -patchSize)
    loudnessBuffer = np.roll(loudnessBuffer, -patchSize)
    centroidBuffer = np.roll(centroidBuffer, -patchSize)
    mfccBuffer = np.roll(mfccBuffer, -patchSize)
    #onsetBuffer = np.roll(onsetBuffer, -patchSize)

    flatnessBuffer = pool['flatness'][-patchSize]
    loudnessBuffer = pool['loudness'][-patchSize]
    centroidBuffer = pool['centroid'][-patchSize]
    mfccBuffer = pool['mfcc'][-patchSize]
    #onsetBuffer = pool['onset'][-patchSize]

    #print ("MFCCs:", '\n', (mfccBuffer))
    #print ("loudnessBuffer:", loudnessBuffer)
    features = np.concatenate((flatnessBuffer, loudnessBuffer, centroidBuffer, mfccBuffer), axis=None)
    #min_max_scaler = MinMaxScaler()
    #data_scaled = min_max_scaler.fit_transform(features.reshape(1, -1))
    #roundNum = 2
    #features_round = np.round(data_scaled, roundNum)
    features = features.tolist()
    return features
    return client.send_message("/genLoud", loudnessBuffer) #have to be a list

#%%
def create_network():
    """ create the structure of the neural network """
    model = Sequential()
    neurons = 128
    model.add(LSTM(
        neurons,
        input_shape=(100, 16), #n_steps, n_features
        return_sequences=False
    ))
    model.add(RepeatVector(16)) #n_features
    model.add(LSTM(neurons, return_sequences=True))
    model.add(TimeDistributed(Dense(11, activation='softmax')))# eleven is the possible numbers descriptions from 0 to 10
    model.compile(loss='categorical_crossentropy' , optimizer='adam', metrics=[ 'accuracy' ])
    model.load_weights(load_model)
    return model

min_max_scaler = MinMaxScaler()
all_features = []
model = create_network()
#prediction = []

def reducer( features, acc, fileData):
  fileData_ =  np.array(fileData['allFeatures'])
  features_ = np.array(features)
  #print("fileData:", fileData_)
  #print("features", features)
  #print("len filedata", len(file))
  diff = np.linalg.norm(fileData_ - features_) #total diff of array
  #print("dif", diff)
  if acc==None:
    #print("tzassoc:", tz.assoc(fileData, 'diff', diff))
    return tz.assoc(fileData, 'diff', diff)
  else:
    if acc['diff'] <= diff:
        #print('acc:', acc)
        return acc
    else:
        #print('finalres:', tz.assoc(fileData, 'diff', diff))
        return tz.assoc(fileData, 'diff', diff)

def getClosestCandidate(features):
  with open(json_in) as f:
      jsonData = json.load(f)
  return reduce(lambda acc, fileData: reducer(features, acc, fileData), jsonData, None)

#%%
def set_zero(sample, d):
    """Set all max value along dimension d in matrix sample."""
    argmax_idxs = sample.argmax(d)
    return argmax_idxs*0.1 #this can vary depends on decimal transform of np.round

file_selected = '000000'

def accFeatures(data):
    global model
    global all_features
    global file_selected
    global features_out
    global allData

    i = len(all_features)
    if i == 100: #n_steps requiered by the model
        all_features = np.array(all_features)
        features_scaled = min_max_scaler.fit_transform(all_features)
        features_scaled = np.round(features_scaled, 1)
        features_reshaped = np.reshape(features_scaled, (1, features_scaled.shape[0], features_scaled.shape[1]))
        prediction = model.predict(features_reshaped, verbose=0) #shape = (n, 1) where n is the n_days_for_prediction
        prediction = set_zero(prediction, d=2)
        result = list(map(getClosestCandidate, prediction)) #[0:60]#se esta leyendo n veces el archivo?
        #print(result)
        all_features = []
        file_selected = list(map(lambda x: x['file'], result))
        file_selected = str(file_selected)[1:-1]
        file_selected = file_selected.replace("'", "")
        features_out = list(map(lambda x: x['allFeatures'], result))

        file_selected = file_selected.replace("[", "")
        file_selected = file_selected.replace("]", "")
        features_out[0].append(file_selected)
        print('Predicción LSTM')
        print(features_out)
        print(file_selected)
        return client.send_message("/allData", features_out) #have to be a list
    else:
        allData = []
        all_features.append(data)

# capture and process the speakers loopback
# the 2 selects the external interface Zoom h5 #3 o -1 for jack
with sc.all_microphones(include_loopback=True)[-1].recorder(samplerate=sampleRate) as mic:
  while True:
      accFeatures(callback(mic.record(numframes=bufferSize).mean(axis=1)))
# %%
