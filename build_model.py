import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.optimizers import Adam
from settings import *
import numpy as np
from ast import literal_eval as make_tuple

# TODO --> Hyperparameter optimization
def build_model(save=False):

    # Build model
    inputs = Input(shape=(3,))
    dense1 = Dense(units=128,
                   activation="relu")(inputs)
    dense2 = Dense(units=128,
                   activation="relu")(dense1)
    out = Dense(units=1)(dense2)
    model = Model(inputs=[inputs],
                  outputs=[out])

    # Compile model
    optimizer = Adam(learning_rate=1e-5)
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    # model.summary()

    # Save model
    if save:
        save_model(model=model,
                   filepath="C:\\Users\\ARSL\\PycharmProjects\\Project_Matt\\models\\predict_state")

    return model

def train_model(X_train, y_train, save=False):

    model = MODEL

    # Train model
    model.fit(x=X_train,
              y=y_train,
              epochs=10,
              batch_size=64,
              shuffle=False)

    # Save model
    if save:
        save_model(model=model,
                   filepath="C:\\Users\\ARSL\\PycharmProjects\\Project_Matt\\models\\predict_state")

    return model

def tup(tup):
    try:
        return make_tuple(tup)
    except:
        return [0, 0]

if __name__ == "__main__":

    import pandas as pd

    df = pd.read_csv('metadata.csv')
    del df['Unnamed: 0']
    df['State2'] = df['State']
    # print(df.head(5))
    df['State2'] = df['State2'].shift(16, axis=0)
    df.fillna(0)
    # print(np.array(list(map(tup, df['State2'].tolist()))))
    # print(np.linalg.norm(np.array(list(map(tup, df['State2'].tolist()))) - np.array(list(map(tup, df['State'].tolist()))), axis=1))
    df['State3'] = np.linalg.norm(np.array(list(map(tup, df['State2'].tolist()))) - np.array(list(map(tup, df['State'].tolist()))), axis=1)
    # plt.hist(df['State3'], range=(0, 10))
    # plt.show()
    df = df[df['State3'] < 11]

    # print(len((df['Frequency']-230/max(df['Frequency']))))

    y_train = np.array(df['State3'].tolist())
    # print(y_train.shape)
    # y_train = np.asarray(y_train).astype(np.float32)
    X_train = np.array(list(zip( (df['Vpp']-2)/max(df['Vpp']), (df['Frequency']-230)/max(df['Frequency']), df['Size']/max(df['Size']) ) ) )
    # print(X_train.shape)
    # X_train = np.asarray(X_train).astype(np.float32)

    # train_model(X_train=X_train, y_train=y_train, save=False)


    # Build model
    inputs = Input(shape=(3,))
    dense1 = Dense(units=128,
                   activation="relu")(inputs)
    dense2 = Dense(units=128,
                   activation="relu")(dense1)
    out = Dense(units=1)(dense2)
    model = Model(inputs=[inputs],
                  outputs=[out])

    # Compile model
    optimizer = Adam(learning_rate=1e-5)
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    model.fit(x=X_train, y=y_train, batch_size=64, shuffle=True, epochs=1000)

    # build_model(save=True)
    # import os
    # print(os.listdir('C:\\Users\\Matthijs\\PycharmProjects\\AI_Actuated_Microswarm_4\\models\\'))