from keras.layers import Activation, Dense, Conv2D, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras import backend as K

def build_dqn(lr, n_actions, input_dims, fc1_dims):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=8, strides=4, activation="relu",
                        input_shape=(*input_dims,), data_format="channels_first"))
    model.add(Conv2D(filters=64, kernel_size=4, strides=2, activation="relu", 
                        data_format="channels_first"))
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, activation="relu",
                        data_format="channels_first"))
    model.add(Flatten())
    model.add(Dense(fc1_dims, activation="relu"))
    model.add(Dense(n_actions))

    model.compile(optimizer=Adam(lr=lr), loss="mean_squared_error")

    return model