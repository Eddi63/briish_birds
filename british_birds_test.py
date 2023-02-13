# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 11:58:14 2023

@author: Eden Akiva
"""

import librosa
#import librosa.display
import numpy as np
import pandas as pd
from gen_pairs import generate_pairs
from model import get_model
from sklearn.model_selection import train_test_split

from tqdm import tqdm
import pickle


# Read metadata file
metadata = pd.read_csv("birdsong_metadata.csv")
header = list(metadata.head())


# Get bird names
#bird_names = metadata['english_cname'].values
#u_birdnames, counts = np.unique(bird_names, return_counts=True)
our_species = ['Canada Goose', 'Carrion Crow', 'Coal Tit', 'Common Blackbird', 
               'Common Chaffinch', 'Common Chiffchaff', 'Common Linnet', 
               'Common Moorhen', 'Common Nightingale']

ourbirds_df = metadata[ metadata['english_cname'].isin(our_species) ]
our_u_birdnames, counts = np.unique(ourbirds_df['english_cname'], return_counts=True)

# data_train = []
# data_test = []
# y_train = []
# y_test = []
bird_name_dict = {i:j for i,j in enumerate(our_u_birdnames)}

# Get file_id corresponding to bird names
# for i in range(len(our_u_birdnames)) :
#     df = metadata[metadata['english_cname'] == our_u_birdnames[i]]
#     df = df['file_id'].values
#     df = df.tolist()
#     data_train.append(df[0])
#     y_train.append(i)
#     #bird_name_dict[i] = our_u_birdnames[i]
#     data_test += df[1:]
#     y_test += [i] * (len(df) - 1)
    
X_frames = []

frame_len = 22050*2 # equivalent of 2 seconds ###fs 44100
y_frames = []

#frames_test = []
#y_frames_test = []


for i in tqdm(range(len(ourbirds_df))) :  # for each recording

    # Read audio
    curr_df = ourbirds_df.iloc[i] #, [0,3]] # to get only filename and commonname columns
    #curr = data_train[i]
    #curr = os.getcwd() + "/BritishBirdSongDataset/songs/songs/xc" + str(curr) + ".flac"
    curr_file = curr_df['file_id']
    curr_species = curr_df['english_cname']
    species_by_num = list(bird_name_dict.keys())[list(bird_name_dict.values()).index(curr_species)]
    curr = "songs/songs/xc" + str(curr_file) + ".flac"
    y, sr = librosa.load(curr) #, sr=None) ###fs ###
    # if sr != 44100: print("warning: sr of file", str(curr_file), 'is', str(sr)) ###fs

    # Normalize time series 
    y = ((y-np.amin(y))*2)/(np.amax(y) - np.amin(y)) - 1     #why?

    # Remove silence from the audio
    orig_len = len(y) # orig_len_sec = len(y)/sr
    intervals = librosa.effects.split(y, top_db= 15, ref= np.max)
    intervals = intervals.tolist()
    y = (y.flatten()).tolist()
    nonsilent_y = []

    for p,q in intervals :
        nonsilent_y = nonsilent_y + y[p:q+1] #???so does librosa..split return including the last index?

    y = np.array(nonsilent_y)
    final_len = len(y) #665764, when sr = 44100 i=0 ; 463456 when sr = 22050
    sil = orig_len - final_len  #how does above make sense???


    # Divide audio into frames
    start = 0 #???what for??
    end = frame_len #???what for??
    for j in range(0, len(y)-sr, int(frame_len*0.5)) : # 50% overlap #should we try going to len(y)-(2*sr) aka -frame_len? at leaaaast - sr
#****
        frame = y[j:j+frame_len]
        # if len(frame) < frame_len: # for debugging
        #     xx = 1 # do nothing
        if len(frame) < sr : continue #does this work??
        if len(frame) < frame_len :
            frame = frame.tolist() + [0]* (frame_len-len(frame)) #maybe duplicating what exists instead?
        frame = np.array(frame)
        
        # Extract spectrogram
        S = np.abs(librosa.stft(frame, n_fft=512))   # 4096, hop_length=1024))  # winlen=nfft???
        
        # Get frequencies associated with STFT
        freqs = librosa.fft_frequencies(sr=sr, n_fft=512)  # 4096, hop_length=1024))  #
        
        # Apply filter on frequency range
        upper = ([x for x in range(len(freqs)) if freqs[x] >= 8000])[0] # technically freqs[upper] = 8010.3515625
        lower = ([x for x in range(len(freqs)) if freqs[x] <= 1000])[-1] # and freqs[lower] = 990.52734375
        
        freqs = freqs[lower:upper] #upper gets cut so till 7967. but lower still 990.
        S = S[lower:upper,:]


        if S.shape != (163, 345) : #for sr=22050 #  (82, 690) : ###fs  
            print(S.shape)
        assert S.shape == (163, 345) #for sr=22050 #  (82, 690) ###fs  #345 is time samples(44100) divided by hop size ( n_fft/4 aka 512/4= 128)

        X_frames.append(S) 
        y_frames.append(species_by_num)

num_uniq, counts_num = np.unique(y_frames, return_counts=True)


#how do i use this to eliminate rerunning above code?
# Write X_frames and y_frames into a pickle file
f = open(r'data_frames1.pkl', 'wb')
pickle.dump([X_frames, y_frames], f) #list of 808 then 781 then samples and labels
f.close()

# Read training and testing data from the pickle file
f = open( "data_frames1.pkl", 'rb')
X_frames, y_frames = pickle.load(f) #normalized, nonsilent_smushed, 2sec frames, and labels
f.close()

y_frames = np.array(y_frames) #  why not in order? 0 1 4 7 6 8 2 5 3
r,c = X_frames[0].shape #unecessary??
X_frames = np.array(X_frames)
X_frames = X_frames.reshape((len(X_frames), r, c)) #  does this do anything?? i dont think it does

X_frames_train = []
X_frames_test = []
y_frames_train = []
y_frames_test = []

X_frames_train, X_frames_test, y_frames_train, y_frames_test = train_test_split(X_frames, y_frames, test_size=0.4)
#returns unordered species, 468 train, 313 test, from 781 total
#returns unordered species, 484 train, 324 test, from 808 total

#??? do we want to standardize the data? do we lose anything? what do we gain?
# Standardize the data
# mu = X_frames_train.mean() # so mu & std calculated based on train but applied to both train&test ?
# std = X_frames_train.std()
# X_frames_train = (X_frames_train-mu)/std
# X_frames_test = (X_frames_test-mu)/std


# There are imbalanced classes. Repeat the data so that all classes have same number of samples?
#??? should we do this? will have more samples, but menupach data.

anchor, pos, neg = generate_pairs(X_frames_train, y_frames_train, rand_samples= -1, pos_pair_size=1200)
#anchor.shape (1872, 163, 345)

#what will we lose from this
# change from numpy.float64 to np.float16
# anchor = anchor.astype(np.float16) #this is a hard assignment
# pos = pos.astype(np.float16)
# neg = neg.astype(np.float16)



# Train the model
r,c = anchor[0].shape
encoder, model = get_model(c,r) # why??

anchor = anchor.transpose(0, 2, 1) #??? pos.transpose????
pos = pos.transpose(0, 2, 1)
neg = neg.transpose(0, 2, 1)

y = np.ones((len(anchor), 32*3))  # Y is the dummy output



# import winsound
# import copy
# xx = copy.deepcopy(y)
# from scipy.io.wavfile import write
# INT16_FAC = (2 ** 15) - 1
# xx*= INT16_FAC
# xx = np.int16(xx)
# write("isthisathing", sr, xx)
# winsound.PlaySound("isthisathing", winsound.SND_FILENAME)




# pathAudio = "songs/songs"
# files = librosa.util.find_files(pathAudio) #, ext=['FLAC']) 
# A = [] 
# files = np.asarray(files)

# for y in files: 
#     data, sr = librosa.load(y, sr=None)
    
#     # Normalize time series between -1 and 1
#     data = ((data-np.amin(data))*2)/(np.amax(data) - np.amin(data)) - 1

#     ps = librosa.feature.melspectrogram(y= data, sr=sr)   
#     A.append((ps, y))
    
#     # fig, ax = plt.subplots()
#     # D = librosa.amplitude_to_db(np.abs(ps), ref=np.max)
#     # img = librosa.display.specshow(D, y_axis='linear', x_axis='time', sr=sr)

#     # Remove silence from the audio

#     org_len = len(data)
#     intervals = librosa.effects.split(data, top_db= 15, ref= np.max)
#     intervals = intervals.tolist()
#     data = (data.flatten()).tolist()
#     nonsilent_data = []
    
#     for p,q in intervals :
#         nonsilent_data = nonsilent_data + data[p:q+1] 
    
#     data = np.array(nonsilent_data)
    
    
#     for nonsilent_interval in data:
#         num_sections = int(np.ceil(len(nonsilent_interval) / sr*2))
#         split = []
#         for i in range(num_sections):
#             t = nonsilent_interval[i * sr : i * sr + sr*2]
#             split.append(t)
    
