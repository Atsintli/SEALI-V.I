import numpy as np
import soundcard as sc
from struct import unpack
from essentia.streaming import *
from essentia import Pool, run, array, reset
import argparse
import requests
from pythonosc import dispatcher
from pythonosc import osc_server
from pythonosc import udp_client
import json

client = udp_client.SimpleUDPClient('127.0.0.4', 5012) #this client sends to SC

sampleRate = 44100
frameSize = 2048
hopSize = 2048
numberBands = 3
patchSize = 60  #control the velocity of the extractor 20 is approximately one second of audio
displaySize = 10

bufferSize = patchSize * hopSize
buffer = np.zeros(bufferSize, dtype='float32')
vectorInput = VectorInput(buffer)
frameCutter = FrameCutter(frameSize=frameSize, hopSize=hopSize)
w = Windowing(type = 'hann')
spec = Spectrum()
mfcc = MFCC(numberCoefficients=13)
pool = Pool()

vectorInput.data  >> frameCutter.signal
frameCutter.frame >> w.frame >> spec.frame
spec.spectrum     >> mfcc.spectrum
mfcc.bands        >> None
mfcc.mfcc         >> (pool, 'mfcc')

def callback(data):
    buffer[:] = array(unpack('f' * bufferSize, data))
    mfccBuffer = np.zeros([numberBands])
    reset(vectorInput)
    run(vectorInput)
    mfccBuffer = np.roll(mfccBuffer, -patchSize)
    mfccBuffer = pool['mfcc'][-patchSize]
    features = mfccBuffer
    features = features.tolist()
    return features

def tf_handler(args):
  headers = {"content-type": "application/json"}
  data = {"instances": [args]}
  r = requests.post(url = "http://localhost:8531/v1/models/improv_class:predict", data=json.dumps(data), headers=headers)
  response = r.json()
  data = response["predictions"]
  client.send_message("/clase", *data)

  clases=data[0]
  event = max(clases)
  index = clases.index(event)
  print("Clasificación DNN")
  print ("Clase Predominante", index)
  print(data)

# capture and process the speakers loopback
# the 2 selects the external interface Zoom h5 #3 for jack
with sc.all_microphones(include_loopback=True)[-1].recorder(samplerate=sampleRate) as mic:
  while True:
    tf_handler(callback(mic.record(numframes=bufferSize).mean(axis=1)) )
    #print ('\n', prediction)
