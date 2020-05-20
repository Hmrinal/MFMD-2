import pandas as pd
#read in training data
train_data = pd.read_csv('diabetes_data.csv')

#view data structure
train_data.head()
#create a dataframe with all training data except the target column
train_X = train_data.drop(columns=['diabetes'])

#check that the target variable has been removed
train_X.head()

from keras.utils import to_categorical
#one-hot encode target column
train_y = to_categorical(train_data.diabetes)
train_y_1 = train_y[:,1]
train_y_1 = train_y_1.reshape(-1,1)
#vcheck that target column has been converted
train_y[0:5]

from keras.models import Sequential
from keras.layers import Dense

#create model
model = Sequential()

#get number of columns in training data
n_cols = train_X.shape[1]

#add layers to model
model.add(Dense(5, activation='relu', input_shape=(n_cols,)))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(2, activation='softmax'))
#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#train model
model.fit(train_X, train_y, batch_size=32, epochs=25)
scores1 = model.evaluate(train_X, train_y, verbose=0)
model.save("model.h5")
print("Saved model to disk")


# load and evaluate a saved model
from numpy import loadtxt
from keras.models import load_model
 
# load model
model = load_model('model.h5')
# summarize model.
model.summary()
