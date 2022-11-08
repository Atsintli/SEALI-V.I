import json
import numpy as np
import glob
import csv
from numpy import loadtxt
import os
from utils import get_json, save_as_json, save_matrix_array
from utils import save_descriptors_as_matrix
import toolz as tz
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import sys
import os

#file_in = input("Enter the path and name of the csv file you want to work with: ")
#json_file = input("Enter the path and name of the json file you want to generate: ")


#assert os.path.exists(user_input), "I did not find the file at, "+str(file_in)
#f = open(user_input,'r+')
#print("Hooray we found your file!")
#stuff you do with the file goes here
#f.close()

#file_in = "features_music18_out.csv"
#json_file = "pca_music18.json"
file_in = "datasets/improv_elecrtoacoustic_dataset.csv"
json_file = "datasets/improv_elecrtoacoustic_dataset.json"


features = loadtxt(file_in)

def round(features):
    roundNum = 1
    #Standardize the feature matrix
    #std_features = StandardScaler().fit_transform(features.data)
    min_max_scaler = MinMaxScaler()
    data_scaled = min_max_scaler.fit_transform(features)
    features_round = np.round(data_scaled, roundNum)
    #Create a PCA that will retain 99% of variace
    #pca = PCA(n_components=16, whiten=True) #svd_solver="randomized"
    #features_pca = pca.fit_transform(std_features)
    print(features_round.shape[1])
    print(len(features_round))
    print (features_round.shape[1])
    print(features_round[1])
    return features_round

# Function to convert a CSV to JSON
def convert_write_json(data, json_file):
    with open(json_file, "w") as f:
        f.write(json.dumps(data, sort_keys=False, indent=1, separators=(',', ': ')))

counter = 0

def makePCAByFileName(pcaOutput):
    myarr=[]
    global counter
    for i in range(len(pcaOutput)):
        filename = "{:06d}".format(counter)
        myarr.append({'file': filename, 'allFeatures': pcaOutput[i]})
        counter = counter + 1
    return myarr

data = round(features)
new_data = makePCAByFileName(data.tolist())
convert_write_json(new_data, json_file)
