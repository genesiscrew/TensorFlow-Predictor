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


samples = data['price'].values[:1201]



# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
samples = scaler.fit_transform(samples)








time_samples = list(range(1201))
#plt.plot(time_samples, samples)
#plt.xlabel('Time')
#plt.ylabel('Magnitude')
#plt.title('Regularly sampled sinusoid with noise')

#X, Y, time_vector = tflow.features.irregular_prediction(time_samples, samples)


num_training_points = 1000
X = samples[:-1]
Y = samples[1:]
X_train = X[:num_training_points]
Y_train = Y[:num_training_points]
X_test = X[num_training_points:]
Y_test = Y[num_training_points:]

X_train = np.reshape(X_train,(len(X_train), 1))
Y_train = np.reshape(Y_train,(len(Y_train), 1))
X_test = np.reshape(X_test,(len(X_test), 1))
Y_test = np.reshape(Y_test,(len(Y_test), 1))

#plt.plot(time_samples[1000:], X_test[0:])
#plt.show()

input_size = 1
hidden_size = 10
output_size = 1

inputs = tflow.placeholders.prediction.input_placeholder(input_size)
input_lstm_layer = tflow.layers.InputLSTMLayer(inputs, input_size)
lstm_layer = tflow.layers.LSTMLayer(input_size, hidden_size, input_lstm_layer)
reg_layer = tflow.layers.RegressionLayer(hidden_size, output_size, lstm_layer)
output_layer = tflow.layers.OutputLSTMLayer(output_size, reg_layer)

y = tflow.placeholders.prediction.output_placeholder(output_size)


outputs = output_layer.get_outputs()



# Defining MSE as the loss function
loss_func = tf.reduce_mean(tf.pow(tf.subtract(outputs, y), 2))


# Training with Adadelta Optimizer
train_step = tf.train.AdamOptimizer(learning_rate=0.05).minimize(loss_func)

# Starting tensorflow session
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

monitor = AutomatedTrainingMonitor(inputs, y, X_train, Y_train,
                                   train_step, loss_func, sess, training_steps=500,
                                   validation_input=X_test, validation_output=Y_test,
                                   early_stopping_rounds=10)

monitor.train()

output = sess.run(outputs,feed_dict={inputs:X_test})
#fig1 = plt.figure()
#ax1 = fig1.add_subplot(211)
plt.plot(time_samples[num_training_points+1:], output, label = 'Predicted')
#fig2 = plt.figure()
#ax2 = fig2.add_subplot(212)
plt.plot(time_samples[num_training_points+1:], Y_test, label = 'Actual')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Magintude')
plt.title('Test set predictions');
print("prediction complete")
plt.show()
