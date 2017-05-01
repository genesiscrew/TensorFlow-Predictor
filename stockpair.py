import timeflow as tflow
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd
import timesynth as ts
from timeflow.trainer import AutomatedTrainingMonitor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn import datasets, linear_model



#Initializing TimeSampler
#time_sampler = ts.TimeSampler(stop_time=20)
#Sampling regular time samples
#regular_time_samples = time_sampler.sample_regular_time(num_points=200)

#Initializing Sinusoidal signal
#sinusoid = ts.signals.Sinusoidal(frequency=0.25)

#Initializing TimeSeries class with the signal and noise objects
#timeseries = ts.TimeSeries(sinusoid, noise_generator=None)
#Sampling using the irregular time samples
#samples, signals, errors = timeseries.sample(regular_time_samples)




data = pd.read_csv("option_data345.csv")
data.columns = ["price", "id", "time", "A", "B"]

data['price']  # as a Series
data['price'].values


samples = data['price'].values[2400:3601]
samples = samples.astype('float32')
look_back = 30
prediction_lag = 60
print(len(samples))



# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
samples = scaler.fit_transform(samples)








# convert an array of values into a X-train matrix
def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1-prediction_lag):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + (look_back+prediction_lag)])
    return np.array(dataX), np.array(dataY)

# convert an array of values into a Y-test matrix
def create_dataset2(dataset, look_back):
    dataY = []
    for i in range(len(dataset)-1):
        dataY.append(dataset[i])
    return np.array(dataY)


# convert an array of values into a X-test matrix
def create_dataset3(dataset, training_length,look_back):
    dataX = []
    # we look back into data by size of lookback plus prediction target, then we
    # go through through array untill point minus lookback plus
    for i in range((training_length-(look_back+prediction_lag)), (len(dataset) - look_back - 1 - prediction_lag)):
        a = dataset[i+(0*prediction_lag):(i + look_back+(0*prediction_lag))]
        dataX.append(a)
    return np.array(dataX)


time_samples = list(range(1201))
#plt.plot(time_samples, samples)
#plt.xlabel('Time')
#plt.ylabel('Magnitude')
#plt.title('Regularly sampled sinusoid with noise')

#X, Y, time_vector = tflow.features.irregular_prediction(time_samples, samples)

# split into train and test sets
train_size = int(len(samples) * 0.83)


test_size = len(samples) - train_size
X, Y = samples[0:train_size], samples[train_size:len(samples)]
print(len(X), len(Y))




num_training_points = 1000
#X = samples[:-1]
#Y = samples[1:]
X_train, Y_train = create_dataset(X, look_back)
X_test = create_dataset3(samples,train_size, look_back)
Y_test = create_dataset2(Y, look_back)
#X_train = X[:num_training_points]
#Y_train = Y[:num_training_points]
#X_test = X[num_training_points:]
#Y_test = Y[num_training_points:]

print(len(Y_train))
print(len(X_train))
print(len(Y_test))
print(len(X_test))


#X_train = np.reshape(X_train,(len(X_train), 1))
Y_train = np.reshape(Y_train,(len(Y_train), 1))
#X_test = np.reshape(X_test,(len(X_test), 1))
Y_test = np.reshape(Y_test,(len(Y_test), 1))



#plt.plot(time_samples[1000:], X_test[0:])
#plt.show()

input_size = look_back
hidden_size = 4
hidden_size2 = 1
hidden_size3 = 30
hidden_size4 = 1
hidden_size5 = 1
output_size = 1

inputs = tflow.placeholders.prediction.input_placeholder(input_size)

input_lstm_layer = tflow.layers.InputLSTMLayer(inputs, input_size)
lstm_layer = tflow.layers.LSTMLayer(input_size, output_size, input_lstm_layer)
#nn_layer = tflow.layers.MultiNNLayer(hidden_size4, output_size, lstm_layer, layers=5, layer_size=[1,1,1,1,1], func='sigmoid', outfunc='regression')
#outlstm_layer1 = tflow.layers.OutputLSTMLayer(output_size, lstm_layer, batch_output=False)
#reg_layer = tflow.layers.RegressionLayer(hidden_size, output_size, lstm_layer)
#input_lstm_layer2 = tflow.layers.InputLSTMLayer(outlstm_layer1.get_outputs(), hidden_size2)
#lstm_layer2 = tflow.layers.LSTMLayer(hidden_size2, output_size, input_lstm_layer2)
#outlstm_layer2 = tflow.layers.OutputLSTMLayer(hidden_size2, lstm_layer2, batch_output=True)
#reg_layer = tflow.layers.RegressionLayer(output_size, output_size, nn_layer)
output_layer = tflow.layers.OutputLSTMLayer(output_size,lstm_layer)

y = tflow.placeholders.prediction.output_placeholder(output_size)


outputs = output_layer.get_outputs()



# Defining MSE as the loss function
loss_func = tf.reduce_mean(tf.pow(tf.subtract(outputs, y), 2))
#loss_func2 = mean_absolute_error(y_true, y_pred)


# Training with Adadelta Optimizer

train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_func)

#global_step = tf.Variable(0, name='global_step', trainable=False)
#train_step = tf.train.AdadeltaOptimizer(learning_rate=0.001,
#    rho=0.95,
#    epsilon=1e-08,
#    use_locking=False,
#    name='Adadelta').minimize(loss_func)

# Starting tensorflow session
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

monitor = AutomatedTrainingMonitor(inputs, y, X_train, Y_train,
                                   train_step, loss_func, sess, training_steps=1,
                                   validation_input=X_test, validation_output=Y_test,
                                   early_stopping_rounds=10)

monitor.train()

output = sess.run(outputs,feed_dict={inputs:X_test})
output = scaler.inverse_transform(output)
Y_test = scaler.inverse_transform(Y_test)
X_test = scaler.inverse_transform(X_test)
#fig1 = plt.figure()
#ax1 = fig1.add_subplot(211)
plt.plot( output, label = 'Predicted')
#fig2 = plt.figure()
#ax2 = fig2.add_subplot(212)
plt.plot(Y_test, label = 'Actual')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Magintude')
plt.title('Test set predictions')

regr = linear_model.LinearRegression()
win = 0;
loss = 0
numTrades = 0
for i in range(len(Y_test)-1-15):

    buy_data = output[i:(i+15)][:,0]
    time_interval = list(range(0,15))
    x1 = np.array(time_interval).reshape(len(time_interval), 1)
    y1 = np.array(buy_data).reshape(len(buy_data), 1)
    print(len(buy_data), len(time_interval))
    regr.fit(x1, y1)
    print('Coefficients: \n', regr.coef_)
    regr.s
    print("interval data")

    pipSecure = 0.0004
    if ((output[i][0] - pipSecure) > X_test[i][look_back-1]) or ((output[i][0] + pipSecure) < X_test[i][look_back-1]):
        if (Y_test[i] > X_test[i][look_back-1] and (output[i][0]-pipSecure) > X_test[i][look_back-1]) or (Y_test[i] < X_test[i][look_back-1] and (output[i][0]+pipSecure) < X_test[i][look_back-1]):
         win = win + 1
         numTrades = numTrades + 1
        else:
            loss = loss + 1
            numTrades = numTrades + 1

accuracy = float(win)/float(numTrades)
print("accuracy is:")
print(accuracy*100)
print("number of trades")
print(numTrades)
plt.show()
