import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
import time
import matplotlib.pyplot as plt
import time
import os

def get_preprocessed_raw_data(filename):
	try:
		df = pd.read_csv(filename)
		return df
	except(FileNotFoundError):
		print(
			'Error handled. File: \'{filename}\' was not found, run the preprocessing first and check spelling of the name.')
		exit()

def process_data(df_preprocessed):

	seed_train_test = 0
	seed_test_val = 0
	# random state is a seed value
	train_data = df_preprocessed.sample(frac=0.75, random_state=seed_train_test)
	test_data = df_preprocessed.drop(train_data.index)
	Y_train = train_data[['target']]
	X_train = train_data.drop(columns=['target'])
	y_test = test_data[['target']]
	x_test = test_data.drop(columns=['target'])
	X_val, X_test, Y_val, Y_test = train_test_split(x_test, y_test, test_size=0.4, random_state=seed_test_val)
	X_val = np.array(X_val)
	Y_val = np.array(Y_val)
	X_test = np.array(X_test)
	Y_test = np.array(Y_test)
	X_train = np.array(X_train)
	Y_train = np.array(Y_train)

	# Normalize the data
	scaler = MinMaxScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.fit_transform(X_test)
	X_val = scaler.fit_transform(X_val)

	X_train = np.asarray(X_train).astype(np.float32)
	Y_train = np.asarray(Y_train).astype(int)
	X_test = np.asarray(X_test).astype(np.float32)
	Y_test = np.asarray(Y_test).astype(int)
	X_val = np.asarray(X_val).astype(np.float32)
	Y_val = np.asarray(Y_val).astype(int)

	return X_train, Y_train, X_test, Y_test, X_val, Y_val


def plot_and_save_graphs(history, trial_no, show=True, save=True):
	## !!PLOT ALERT!!

	print(history)

	#Data from the last model fitting process
	acc = history.history['accuracy']
	val_acc = history.history['val_accuracy']
	loss = history.history['loss']
	val_loss = history.history['val_loss']

	epochs = range(1, len(acc) + 1)
	#Plotting of Training vs Validation accuracy evolving with epochs
	fig_acc = plt.figure()
	plt.plot(epochs, acc, 'r', label='Training accuracy')
	plt.plot(epochs, val_acc, 'k', label='Validation accuracy')
	plt.title('Training vs Validation Accuracy')
	#plt.title(f'Trial {trial_no}')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.grid()
	# Set limits so figures are more intuitive to read next to each other
	#plt.ylim([0.6, 1])

	fig_loss = plt.figure()
	plt.plot(epochs, loss, 'r', label='Training loss')
	plt.plot(epochs, val_loss, 'k', label='Validation loss')
    #plt.title(f'Trial {trial_no}')
	plt.title('Training vs Validation Loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.grid()
	# Set limits so figures are more intuitive to read next to each other
	#plt.ylim([0, 0.20])

	if(show):
		plt.show()

	if(save):
		# save plots to /figs folder 
		cwd = os.getcwd()
		figs_folder = cwd+'/figs'
		try:
			os.mkdir(figs_folder)
		except OSError: 
			figs_folder = cwd+'/figs'
		t = time.time()
		fig_acc.savefig(figs_folder+'/acc_'+str(t)+"_t_"+str(trial_no)+'.jpg', bbox_inches='tight', dpi=150)
		fig_loss.savefig(figs_folder+'/loss_'+str(t)+'_t_'+str(trial_no)+'.jpg', bbox_inches='tight', dpi=150)
		print(f"Trial {trial_no} plots saved to /figs.")

def write_out_file(arg_epochs, arg_batch_size, acc, conf, nn_type):

	cwd = os.getcwd()
	filepath = cwd+"/outputs"
	filename = f"/deep_{arg_epochs}_{arg_batch_size}_{acc}.txt"

	# Create outputs dir if not existent
	try:
		os.makedirs(filepath)
	except:
		print("Didn't make new folder as /outputs already exists.")

	# Create the output file
	output = open(filepath+filename, 'w')

	output.write(f'Epochs: {arg_epochs}\nBatch size: {arg_batch_size}\nAccuracy: {acc}\nType: {nn_type}\nEach trial:\n---------------------------------------\n')
	output.close()
	output = open(filepath+filename, 'a')
	for i in range(len(conf)):
		# Calculate accuracy for each trial
		df = pd.DataFrame(conf[i])
		tp = df[1][1]
		tn = df[0][0]
		fn = df[0][1]
		fp = df[1][0]
		acc_from_conf_mat = (tp+tn)/(tp+tn+fn+fp)
		precision = tp/(fp+tp)
		recall = tp/(fn+tp)
		f1 = (2*precision*recall)/(precision+recall)
		output.write(f'TRIAL NUMBER {i+1}\nConfusion matrix:\n{df}\nAccuracy:{round(acc_from_conf_mat, 4)}\nPrecision:{round(precision, 4)}\nRecall:{round(recall, 4)}\nF1:{round(f1, 4)}\n')
		output.write('---------------------------------------\n')
	print(f"Parameters and data saved to {filename}.")

def save_model(model, acc):
	# create folder
	cwd = os.path.abspath(os.getcwd())
	m_dir = cwd+'/models'
	try:
		os.mkdir(m_dir)
	except OSError: 
		# Could just be nothing
		m_dir = cwd+'/models'

	m_name = f"/model_with_accuracy_{acc}"

	# serialize model to JSON
	model_json = model.to_json()
	with open(m_dir+m_name+".json", "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights(m_dir+m_name+".h5")
	print(f"Model with accuracy {acc} saved to /models.")


def process_data(df_final):

	#df_final = df_final.drop(columns=['inf_ecg.csv','inf_gsr.csv','inf_ppg.csv','pixart.csv'])
	seed_train_test = 0
	seed_test_val = 0
	# random state is a seed value
	train_data = df_final.sample(frac=0.75, random_state=seed_train_test)
	test_data = df_final.drop(train_data.index)
	Y_train = train_data[['out1', 'out2', 'out3', 'out4', 'out5', 'out6']]
	X_train = train_data.drop(
		columns=['out1', 'out2', 'out3', 'out4', 'out5', 'out6'])
	y_test = test_data[['out1', 'out2', 'out3', 'out4', 'out5', 'out6']]
	x_test = test_data.drop(
		columns=['out1', 'out2', 'out3', 'out4', 'out5', 'out6'])
	X_val, X_test, Y_val, Y_test = train_test_split(
		x_test, y_test, test_size=0.4, random_state=seed_test_val)
	X_val = np.array(X_val)
	Y_val = np.array(Y_val)
	X_test = np.array(X_test)
	Y_test = np.array(Y_test)
	X_train = np.array(X_train)
	Y_train = np.array(Y_train)

	# Normalize the data
	scaler = MinMaxScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.fit_transform(X_test)
	X_val = scaler.fit_transform(X_val)

	X_train = np.asarray(X_train).astype(np.float32)
	Y_train = np.asarray(Y_train).astype(int)
	X_test = np.asarray(X_test).astype(np.float32)
	Y_test = np.asarray(Y_test).astype(int)
	X_val = np.asarray(X_val).astype(np.float32)
	Y_val = np.asarray(Y_val).astype(int)

	return X_train, Y_train, X_test, Y_test, X_val, Y_val


def make_model(arg_epochs, arg_batch_size, learning_rate, activation_func_hidden, activation_func_output, X_train, Y_train, X_test, Y_test, X_val, Y_val):
	"""
	return: epochs, batch size, accuracy, confusion matrix, time to fit the model
	"""

	n_features = X_train.shape[1]

	# Define model
	model = Sequential()
	model.add(Dense(40, activation=activation_func_hidden,
			  kernel_initializer='he_normal', input_shape=(n_features,)))
	model.add(Dense(20, activation=activation_func_hidden,
			  kernel_initializer='he_normal'))
	model.add(Dense(20, activation=activation_func_hidden,
			  kernel_initializer='he_normal'))
	model.add(Dense(6, activation=activation_func_output))
	#model.add(Dense(1, activation=None))

	# Define the optimizer
	model.compile(
		optimizer=Adam(learning_rate),
		loss=MeanSquaredError(),
		metrics=['accuracy']
	)

	start_time = time.time()
	# Fit the model
	history = model.fit(X_train, Y_train, epochs=arg_epochs,
						batch_size=arg_batch_size, verbose=1, validation_data=(X_val, Y_val))
	time_to_fit = round(time.time()-start_time, 2)

	# Testing Data
	y_pred = model.predict(X_test)

	conf = multilabel_confusion_matrix(Y_test, y_pred.round())
	acc = round(accuracy_score(Y_test, y_pred.round()), 5)

	return arg_epochs, arg_batch_size, acc, conf, time_to_fit, history, model


def main():

	preprocessed_data_filename = "preprocessed_raw_data.csv"
	df_preprocessed = get_preprocessed_raw_data(preprocessed_data_filename)

	X_train, Y_train, X_test, Y_test, X_val, Y_val = process_data(df_preprocessed)

	# FITTING NEURAL NETWORK
	epochs = 300
	batch_size = 3

	learning_rate = 5e-3
	activation_func_hidden = 'tanh'
	activation_func_output = 'softmax'

	arg_epochs, arg_batch_size, acc, conf, time_to_fit, history, model = make_model(epochs,\
		batch_size, learning_rate, activation_func_hidden, activation_func_output, X_train, Y_train, X_test, Y_test, X_val, Y_val)

	print(history)
	plot_and_save_graphs(history, 0, show=False, save=True)
	save_model(model, acc)
	print("Confidence matrix:\n", conf)
	print("Achieved accuracy:", acc)
	print("Time to fit was:", time_to_fit)
	write_out_file(arg_epochs, arg_batch_size, acc, conf, "Deep NN")


if __name__ == "__main__":

   main()
