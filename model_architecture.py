from keras import layers
from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from keras.models import Model, load_model, Sequential
from keras.activations import relu, sigmoid, softmax
import keras.backend as K

def create_lenet5_mnist():
    model = Sequential()

    model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
    model.add(layers.AveragePooling2D())

    model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(layers.AveragePooling2D())

    model.add(layers.Flatten())

    model.add(layers.Dense(units=120, activation='relu'))

    model.add(layers.Dense(units=84, activation='relu'))

    model.add(layers.Dense(units=47, activation = 'softmax'))
    
    return model

def create_tr_model():
    # input with shape of height=32 and width=128 
    inputs = Input(shape=(32,128,1))
     
    # convolution layer with kernel size (3,3)
    conv_1 = Conv2D(64, (3,3), activation = 'relu', padding='same')(inputs)
    # poolig layer with kernel size (2,2)
    pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)
     
    conv_2 = Conv2D(128, (3,3), activation = 'relu', padding='same')(pool_1)
    pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)
     
    conv_3 = Conv2D(256, (3,3), activation = 'relu', padding='same')(pool_2)
     
    conv_4 = Conv2D(256, (3,3), activation = 'relu', padding='same')(conv_3)
    # pooling layer with kernel size (2,1)
    pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)
     
    conv_5 = Conv2D(512, (3,3), activation = 'relu', padding='same')(pool_4)
    # Batch normalization layer
    batch_norm_5 = BatchNormalization()(conv_5)
     
    conv_6 = Conv2D(512, (3,3), activation = 'relu', padding='same')(batch_norm_5)
    batch_norm_6 = BatchNormalization()(conv_6)
    pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)
     
    conv_7 = Conv2D(512, (2,2), activation = 'relu')(pool_6)
     
    squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)
     
    # bidirectional LSTM layers with units=128
    blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(squeezed)
    blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(blstm_1)
     
    outputs = Dense(63, activation = 'softmax')(blstm_2)

    # model to be used at test time
    act_model = Model(inputs, outputs)

    return act_model
