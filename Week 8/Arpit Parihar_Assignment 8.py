# %% [markdown]
# # Machine Learning and Predictive Modeling - Assignment 8
# ### Arpit Parihar
# ### 05/26/2021
# ****
# %% [markdown]
# **Importing modules**

# %%
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import SimpleRNN, GRU, LSTM, Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from collections import OrderedDict
import random
import os
import json

import warnings
warnings.filterwarnings('ignore')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# %% [markdown]
# **Setting seed for reproducibility**

# %%
seed_value = 2
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# %% [markdown]
# ### 1\. Data Processing

# %%
dataframe = pd.read_csv('dev-access.csv', engine='python',
                        quotechar='|', header=None)
dataset = dataframe.values
dataset.shape


# %%
X = dataset[:, 0]
y = dataset[:, 1]


# %%
for index, item in enumerate(X):
    # Quick hack to space out json elements
    reqJson = json.loads(item, object_pairs_hook=OrderedDict)
    del reqJson['timestamp']
    del reqJson['headers']
    del reqJson['source']
    del reqJson['route']
    del reqJson['responsePayload']
    X[index] = json.dumps(reqJson, separators=(',', ':'))


# %%
tokenizer = Tokenizer(filters='\t\n', char_level=True)
tokenizer.fit_on_texts(X)

num_words = len(tokenizer.word_index) + 1
X = tokenizer.texts_to_sequences(X)


# %%
max_log_length = 1024
X_processed = sequence.pad_sequences(X, maxlen=max_log_length)


# %%
X_train, X_test, y_train, y_test = train_test_split(
    np.asarray(X_processed).astype(np.float32), np.asarray(y).astype(np.float32), test_size=0.25)

# %% [markdown]
# ## 2\. Model 1 - RNN:
# 
# The first model will be a pretty minimal RNN with only an embedding layer, simple RNN and Dense layer. The next model we will add a few more layers.
# 
# a) Start by creating an instance of a Sequential model: https://keras.io/getting-started/sequential-model-guide/
# 
# b) From there, add an Embedding layer: https://keras.io/layers/embeddings/
# 
# Params:
# 
# - input_dim = num_words (the variable we created above)
# - output_dim = 32
# - input_length = max_log_length (we also created this above)
# - Keep all other variables as the defaults (shown below)
# 
# c) Add a SimpleRNN layer: https://keras.io/layers/recurrent/
# 
# Params:
# 
# - units = 32
# - activation = 'relu'
# 
# d) Finally, we will add a Dense layer: https://keras.io/layers/core/#dense
# 
# Params:
# 
# - units = 1 (this will be our output)
# - activation --> you can choose to use either relu or sigmoid.
# 
# e) Compile model using the .compile() method: https://keras.io/models/model/
# 
# Params:
# 
# - loss = binary_crossentropy
# - optimizer = adam
# - metrics = accuracy
# 
# f) Print the model summary
# 
# g) Use the .fit() method to fit the model on the train data. Use a validation split of 0.25, epochs=3 and batch size = 128.

# %%
try:
    model_RNN = load_model('model_RNN.h5')
except:
    model_RNN = Sequential()
    model_RNN.add(Embedding(input_dim=num_words,
                        output_dim=32, input_length=max_log_length))
    model_RNN.add(SimpleRNN(units = 32,activation = 'relu'))
    model_RNN.add(Dense(units=1, activation='sigmoid'))
    model_RNN.compile(loss='binary_crossentropy',
                    optimizer='adam', metrics='accuracy')
    model_RNN.fit(X_train, y_train, validation_split=0.25, epochs=3, batch_size=128)
    model_RNN.save('model_RNN.h5')

# %% [markdown]
# h) Use the .evaluate() method to get the loss value & the accuracy value on the test data. Use a batch size of 128 again.

# %%
model_RNN.summary()
results_RNN = model_RNN.evaluate(X_test, y_test, batch_size=128, verbose=0)
print(f'\nRNN Loss = {round(results_RNN[0], 4)}')
print(f'\nRNN Accuracy = {round(results_RNN[1], 4)}')

# %% [markdown]
# ## 3\) Model 2 - LSTM + Dropout Layers:
# 
# Now we will add a few new layers to our RNN and incorporate the more powerful LSTM. You will be creating a new model here, so make sure to call it something different than the model from Part 2.
# 
# a) This RNN needs to have the following layers (add in this order):
# - Embedding Layer (use same params as before)
# - LSTM Layer (units = 64, recurrent_dropout = 0.5)
# - Dropout Layer - use a value of 0.5
# - Dense Layer - (use same params as before)
# b) Compile model using the .compile() method:
# 
# Params:
# 
# - loss = binary_crossentropy
# - optimizer = adam
# - metrics = accuracy
# 
# c) Print the model summary
# 
# d) Use the .fit() method to fit the model on the train data. Use a validation split of 0.25, epochs=3 and batch size = 128.

# %%
try:
    model_LSTM = load_model('model_LSTM.h5')
except:
    model_LSTM = Sequential()
    model_LSTM.add(Embedding(input_dim=num_words,
                        output_dim=32, input_length=max_log_length))
    model_LSTM.add(LSTM(units=64, recurrent_dropout=0.5))
    model_LSTM.add(Dropout(0.5))
    model_LSTM.add(Dense(units=1, activation='sigmoid'))
    model_LSTM.compile(loss='binary_crossentropy',
                    optimizer='adam', metrics='accuracy')
    model_LSTM.fit(X_train, y_train, validation_split=0.25, epochs=3, batch_size=128)
    model_LSTM.save('model_LSTM.h5')

# %% [markdown]
#  e) Use the .evaluate() method to get the loss value & the accuracy value on the test data. Use a batch size of 128 again.

# %%
model_LSTM.summary()
results_LSTM = model_LSTM.evaluate(X_test, y_test, batch_size=128, verbose=0)
print(f'\nLSTM Loss = {round(results_LSTM[0], 4)}')
print(f'\nLSTM Accuracy = {round(results_LSTM[1], 4)}')

# %% [markdown]
#  ### 4\) Recurrent Neural Net Model 3: Build Your Own
# 
#  You wil now create your RNN based on what you have learned from Model 1 & Model 2:
# 
# a) RNN Requirements:
# - Use 5 or more layers
# - Add a layer that was not utilized in Model 1 or Model 2 (Note: This could be a new Dense layer or an additional LSTM)
# 
# b) Compiler Requirements:
# - Try a new optimizer for the compile step
# - Keep accuracy as a metric (feel free to add more metrics if desired)
# 
# c) Print the model summary
# 
# d) Use the .fit() method to fit the model on the train data. Use a validation split of 0.25, epochs=3 and batch size = 128.

# %%
try:
    model_custom = load_model('model_custom.h5')
except:
    model_custom = Sequential()
    model_custom.add(Embedding(input_dim=num_words,
                        output_dim=64, input_length=max_log_length))
    model_custom.add(GRU(units=64, return_sequences=True, recurrent_dropout=0.5))
    model_custom.add(GRU(units=64, return_sequences=False, recurrent_dropout=0.5))
    model_custom.add(Dropout(0.5))
    model_custom.add(Dense(units=128, activation='relu'))
    model_custom.add(Dense(units=1, activation='sigmoid'))
    model_custom.compile(loss='binary_crossentropy',
                    optimizer='RMSProp', metrics='accuracy')
    model_custom.fit(X_train, y_train, validation_split=0.25, epochs=3, batch_size=128)
    model_custom.save('model_custom.h5')

# %% [markdown]
# e) Use the .evaluate() method to get the loss value & the accuracy value on the test data. Use a batch size of 128 again.

# %%
model_custom.summary()
results_custom = model_custom.evaluate(X_test, y_test, batch_size=128, verbose=0)
print(f'\nCustom Model Loss = {round(results_custom[0], 4)}')
print(f'\nCustom Model Accuracy = {round(results_custom[1], 4)}')

# %% [markdown]
# ### Conceptual Questions:
# **5) Explain the difference between the relu activation function and the sigmoid activation function.**
#
# The ReLU activation function aims to lower the computational overhead in deep neural networks, and also alleviates vanishing gradient. It outputs 0 for negative inputs, and the input itself for positive ones. The function is non differentiable at x = 0, but the probability of the input being exactly 0 in a 16 or 32 bit float variable is almost non-existent, and the derivative can be hardcoded to be 0 or 1 at x = 0.
#
# Sigmoid function squishes the input between 0 and 1, and so has a vanishing gradient problem. It's computationaly expensive as compared to ReLU, and should not be used for deep neural networks, except for gates in LSTM and GRU.
#
# **6) Describe what one epoch actually is (epoch was a parameter used in the .fit() method).**
#
# One epoch is passing the training data through the model once. For small learning rates, one pass through the data might not be enough for the optimizer to find the minima, which is why multiple epochs are needed to train the neural network.
#
# **7) Explain how dropout works (you can look at the keras code and/or documentation) for (a) training, and (b) test data sets.**
#
# For training data, dropout sets randomly chosen activations to 0 with a probability $p$. The activations not set to 0 are scaled up by a factor of $1/(1-p)$ to keep the sum over all inputs unchanged.
#
# For test data, the dropout layer is inactive.
#
# **8) Explain why problems such as this homework assignment are better modeled with RNNs than CNNs. What type of problem will CNNs outperform RNNs on?**
#
# RNNs are better suited for sequences, as they remember dependencies in data. CNNs on the other hand work well for images, as the filters act over regions of data and extract information to create feature maps.
#
# In this problem, each record is dependent on previous ones, making RNN the better choice. Although CNNs are specialized for image data, they can be used for sequences as well, and should be picked when optimizing for performance on high dimensional data, as CNNs are faster.
#
# **9) Explain what RNN problem is solved using LSTM and briefly describe how.**
#
# RNNs struggle with vanishing gradients when dealing with long term dependencies as the derivatives are multiplied several times over for longer sequences. LSTMs address with by maintaining a cell state alongside the hidden state, which acts as "memory" to remember long term interactions.
