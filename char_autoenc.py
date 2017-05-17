import re
import sys

import keras

import numpy as np

# hyperparameters
batch_size = 10
layer_size = 512
dropout_rate = 0.5

# load raw text
filename = "shakespeare.txt"
with open(filename, 'r') as f:
    lines = f.readlines()

# find list of characters
BLANK = '*'
# START = 'START'
# END = 'END'
raw_text = "".join(lines)
chars = sorted(set(raw_text)) + [BLANK]
num_chars = len(chars)
max_seq_len = max([len(s) for s in lines])

# map characters to vectors
char_to_ind = dict((c,i) for i,c in enumerate(chars))
def char_to_vec(c):
    vec = np.zeros((num_chars))
    vec[char_to_ind[c]] = 1
    return vec

# map vectors to characters
def vec_to_char(vec):
    ind = np.argmax(vec)
    return chars[ind]

# convert data tensor to string
def tensor_to_string(tensor):
    s = ""
    for i in range(len(tensor)):
        for j in range(len(tensor[i])):
            c = vec_to_char(tensor[i,j])
            if len(c) == 1:
                s += c
            # elif c == BLANK:
            #     s += '*'
        s += "\n"
    return s

# convert string into a matrix of one-hots
# stretch length to be seq_len and add START, END, and BLANKs
# looks like this: <string>, BLANK, BLANK, ...
# will return size seq_len x num_chars
def string_to_matrix(s, seq_len=None):
    if seq_len is None: seq_len = len(s)
    vecs = [char_to_vec(c) for c in s] \
        + [char_to_vec(BLANK)] * (seq_len - len(s))
    return np.array(vecs)

# convert a list of strings to a tensor
def strlst_to_tensor(strs, seq_len=None):
    if seq_len is None: seq_len = max([len(s) for s in strs])
    return np.array([string_to_matrix(s, seq_len) for s in strs])

# format data into vectors
def generateData(lines, batch_size):
    textInd = 0
    while True:
        batch_strs = lines[textInd : textInd + batch_size]
        X = strlst_to_tensor(batch_strs, max_seq_len)
        yield X, X
        textInd = (textInd + batch_size) % (len(lines) - batch_size)

'''i = 0
for x, y in generateData(lines, batch_size):
    print(tensor_to_string(x))
    i += 1
    if i >= 2: break'''

# build encoder model
enc_input = keras.layers.Input(shape=[max_seq_len, num_chars])
H = keras.layers.LSTM(layer_size, return_sequences=True)(enc_input)
H = keras.layers.Dropout(dropout_rate)(H)
H = keras.layers.LSTM(layer_size, return_sequences=False)(H)
enc_output = keras.layers.Dropout(dropout_rate)(H)

# build decoder model
H = keras.layers.RepeatVector(max_seq_len)(enc_output)
H = keras.layers.LSTM(layer_size, return_sequences=True)(H)
H = keras.layers.Dropout(dropout_rate)(H)
H = keras.layers.LSTM(layer_size, return_sequences=True)(H)
H = keras.layers.Dropout(dropout_rate)(H)
H = keras.layers.Dense(num_chars, activation='sigmoid')(H)
dec_output = keras.layers.Activation('softmax')(H)

# finalize autoencoder
autoencoder = keras.models.Model(enc_input, dec_output)
autoencoder.compile(loss="categorical_crossentropy", optimizer="adam")

# saving checkpoint
# create checkpoint
weightspath = "weights/weights-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(weightspath, monitor="loss",
    verbose=1, save_best_only=False, mode="min")

# train autoencoder
print("Training")
autoencoder.fit_generator(generateData(lines, batch_size),
    len(lines) / batch_size, epochs=1, callbacks=[checkpoint])
    #100, epochs=1, callbacks=[checkpoint])

# test autoencoder
inp = strlst_to_tensor(lines[10:15], max_seq_len)
pred = autoencoder.predict(inp)
print(tensor_to_string(pred))
