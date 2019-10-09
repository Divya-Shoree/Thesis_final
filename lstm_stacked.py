import numpy as np
import os
import tensorflow as tf
import keras.backend as K
import scikitplot as skplt
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Flatten, Reshape
from keras.layers import LSTM, Embedding, TimeDistributed
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from livelossplot import PlotLossesKeras

'''from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM, Flatten
from tensorflow.keras.callbacks import EarlyStopping
'''
# Import the data 
path = "/home/divya/deepLearning/dataset"
file_name = os.path.join(path, "simulation_40_static.npy")

# Load the imported data as an array
data = np.load(file_name)
print(data.shape)

# Initialize parameters
num_nodes = data.shape[1] 
print(num_nodes)
lstm_units = 256

'''------------------------- Prepare data for model training ------------------------'''
# Setting the parameters for preparing our test and train data sets.
# For the network structure from time t0-t9 predict the structure at time t10.
num_graph_input = 10

total_data = data.shape[0]
num_training_samples = int(0.7 * total_data)
num_testing_samples = int(0.2 * total_data)

x_train = np.array([data[l:num_graph_input+l] for l in range(num_training_samples)],dtype=np.float32)
y_train = np.array(data[num_graph_input:num_training_samples+num_graph_input],dtype=np.float32)
x_test = np.array([data[num_training_samples+l : num_training_samples+num_graph_input+l] for l in range(num_testing_samples)],dtype=np.float32)
y_test = np.array(data[num_training_samples+num_graph_input:num_training_samples+num_testing_samples+num_graph_input],dtype=np.float32)
print(x_train.shape)

'''------------------------------- Evaluate Accuracy ---------------------------------'''
def auc(x, y):
    return roc_auc_score(np.reshape(y,(-1, )), np.reshape(x, (-1, )))

def evaluate(x, y):
    #print("      -------------------------------- ")
    #print('evaluate')
    #print(" ------------------------------------- ")
    y_preds = model.predict(x, batch_size=32)
    accuracy = []
    for i in range(y_preds.shape[0]):
        y_pred = y_preds[i]
        accuracy.append(auc(np.reshape(y_pred, (-1, )), np.reshape(y[i], (-1, ))))
    return accuracy
'''--------------------------------- Build Model ---------------------------------------'''
# There are 2 models present in keras: Sequential and functional. We use functional model as it is more flexible in allowing any layer to be connected to any other instead of just the next layer.
# instantiate the keras tensor and build the model
x = Input(shape=(num_graph_input, num_nodes, num_nodes))
#since LSTM accepts the input in the shape (samples, timesteps, features) we need to convert our 4D sata to 3D
hidden_layer = Reshape((num_nodes,-1))(x)
hidden_layer = LSTM(lstm_units, input_shape=(num_nodes, 128),return_sequences=True, dropout=0.0, recurrent_dropout=0.0)(hidden_layer)
hidden_layer = LSTM(lstm_units, input_shape=(num_nodes, 128),return_sequences=True, dropout=0.0, recurrent_dropout=0.0)(hidden_layer)
y = Dense(num_nodes, activation='sigmoid',input_shape=(num_nodes,lstm_units))(hidden_layer)

# Now we need to group the different layers created into an object 
model = Model(inputs=x, outputs=y)

model.compile(loss=binary_crossentropy, optimizer=Adam(lr=0.0001))
monitor = EarlyStopping(monitor='loss', min_delta=1e-3, patience=5, verbose=1, mode='auto', restore_best_weights=True)
'''--------------------------- Train the model ----------------------------------'''
# train the model
# model.fit returns a history object that contains all information collected during training.
#history = model.fit(x_train, y_train, callbacks=[monitor, PlotLossesKeras()], batch_size=32, epochs=1500, verbose=1)
history = model.fit(x_train, y_train, callbacks=[PlotLossesKeras()], batch_size=32, epochs=1500, verbose=1)
loss = history.history['loss']
# Calculate accuracy
accuracy = evaluate(x_test, y_test)
print(np.average(accuracy))
y_pred = model.predict(x_test, batch_size=32)
#skplt.metrics.plot_roc_curve(y_test, y_pred)
#plot.show()
model.save("model_lstm_stacked.h5")
print("Saved model to the disk")


