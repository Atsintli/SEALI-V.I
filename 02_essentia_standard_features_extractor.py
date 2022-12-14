#%%
import json
import essentia
import essentia.standard as ess
from essentia.standard import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import savetxt
import glob
import csv
import os
from utils import *
import toolz as tz


def extract_mfccs(audio_file):
    loader = essentia.standard.MonoLoader(filename=audio_file)
    print("Analyzing:" + audio_file)
    audio = loader()
    #spectrum = Spectrum()
    w = Windowing(type='hann')
    fft = FFT()

    name = audio_file.split('/')[1].split('.')[-1]
    print(name)

    pool = essentia.Pool()
    for frame in ess.FrameGenerator(audio, frameSize=2048, hopSize=2048, startFromZero=True): #for chroma frameSize=8192*2, hopSize=8192, #fz=88200, hs=44100
        mag, phase, = CartesianToPolar()(fft(w(frame)))
        mfcc_bands, mfcc_coeffs = MFCC(numberCoefficients=13)(mag)
        #mel_bands = MelBands()(spectrum(w(frame)))
        #contrast, spectralValley = SpectralContrast()(mag)
        flatness = Flatness()(mag)
        #dens = Welch()(spectrum(w(frame)))
        #onset = OnsetDetection()(mag,phase)
        #dynamic_complexity, loudness = DynamicComplexity()(mag)
        #spectral_complex = SpectralComplexity()(mag)
        centroid = Centroid()(mag)
        #croma = Chromagram(sampleRate=2048*5)(mag[1:],)
        loudness = Loudness()(mag)

        #probar utilizando un for para extraer todos los descriptores con alguna función de essentia


        pool.add('lowlevel.mfcc', mfcc_coeffs)
        pool.add('lowlevel.loudness', [loudness])
        #pool.add('lowlevel.melbands', mel_bands)
        #pool.add('lowlevel.spectralcontrast', contrast)
        pool.add('lowlevel.flatness', [flatness])
        #pool.add('lowlevel.onsets', [onset])
        #pool.add('lowlevel.dyncomplex', [dynamic_complexity])
        #pool.add('lowlevel.spectral_complexity', [spectral_complex])
        #pool.add('lowlevel.chroma', croma)
        #pool.add('lowlevel.dens', dens)
        pool.add('lowlevel.centroid', [centroid])

    pool.add('audio_file', (name))
    aggrPool = PoolAggregator(defaultStats=['mean','var'])(pool)

    YamlOutput(filename='features.json', format='json',
               writeVersion=False)(aggrPool)

    json_data = get_json("features.json")
    #dyncomp = json_data['lowlevel']['dynamic_complexity']['mean']

    #SCMIR Audio Features
    #[[MFCC],[Chromagram],[SpecPcile, 0.95],[SpecPcile, 0.80],[SpecFlatness]];

    #os.remove("mfccmean.json")
    return {"file": json_data['audio_file'],
            "mfccMean": json_data['lowlevel']['mfcc']['mean'],
            #"mfccVar": json_data['lowlevel']['mfcc']['var'],
            #"mel": json_data['lowlevel']['melbands']['mean'],
            "loudness": json_data['lowlevel']['loudness']['mean'],
            #"spectralContrast": json_data['lowlevel']['spectralcontrast']['mean'],
            # "chroma": json_data['lowlevel']['chroma']['mean'],
            "flatness": json_data['lowlevel']['flatness']['mean'],
            #"onsets": json_data['lowlevel']['onsets']['mean'],
            #"dyncomplexity": json_data['lowlevel']['dyncomplex']['mean'],
            #"complexity": json_data['lowlevel']['spectral_complexity']['mean'],
            #"dens": json_data['lowlevel']['dens']['mean'],
            #"densVar": json_data['lowlevel']['dens']['var'],
            "centroid": json_data['lowlevel']['centroid']['mean']
            }

def extract_all_mfccs(audio_files):
    return list(map(extract_mfccs, audio_files))

def getProps(props, dict):
    return map(lambda prop: dict[prop], props)

#%%

def concat_features(input_data):
    features = list(map(lambda data:
               list(tz.concat(getProps(
                   #['loudness'],
                   #['mel'],
                   #['mfccMean', 'mfccVar'],
                   #['mfccMean','flatness', 'complexity', 'onsets'],
                   #['flatness', 'mfccVar','complexity','mfccMean','loudness','centroid','spectralContrast'],
                   ['flatness', 'loudness', 'centroid', 'mfccMean'],
                   data))),
    input_data))
    print(features)
    return features

#%%
def save_as_matrix(features):
    save_descriptors_as_matrix('datasets/movil_II_09102022_violin_solo.csv', features) #this can be as floats or as string

#test
input_data = extract_all_mfccs(sorted(glob.glob('segments/movil_II_09102022_violin_solo/' + "*.wav")[0:]))
#print(input_data)
save_as_matrix(concat_features(input_data))
# %%
