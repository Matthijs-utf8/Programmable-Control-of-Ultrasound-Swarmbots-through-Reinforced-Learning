# import tensorflow as tf
import numpy as np
import os
import re
from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import datetime

def get_data():


    DIR = "C:\\Users\\Matthijs\\PycharmProjects\\AI_Actuated_Microswarm_2\\Include\\snapshots\\Track_tests_friday\\"

    # data = []

    X_train = []
    y_train = []

    old_x = 58
    old_y = 15
    old_action = 3


    for file in os.listdir(DIR):

        if "reset.png" in file:
            continue

        if "0.png" in file:
            action = 0
        elif "1.png" in file:
            action = 1
        elif "2.png" in file:
            action = 2
        elif "3.png" in file:
            action = 3

        # if action:
        #     old_action = action

        expr1 = re.findall("-.*-", file[:-10])[0]
        expr2 = re.findall("\d+", expr1)[0]

        if len(expr2) == 6:
            x = int(expr2[:3])
            y = int(expr2[3:])
            # print(x, y)
        elif len(expr2) == 5:
            if abs(int(expr2[:3]) - x) > 5:
                x = int(expr2[:2])
                y = int(expr2[2:])
                # print(x, y)
            else:
                x = int(expr2[:3])
                y = int(expr2[3:])
                # print(x, y)
        elif len(expr2) == 4:
            x = int(expr2[:2])
            y = int(expr2[2:])
            # print(x, y)
        else:
            continue


        X_train.append([old_action/3, old_x/300, old_y/300])
        y_train.append([x/300, y/300])

        old_x = x
        old_y = y
        old_action = action

    return X_train, y_train

X_train, y_train = get_data()

def main():



    # Build model
    # inputs = Input((3, len(X_train)))
    # dense1 = LSTM(units=128, activation="relu")(inputs)
    inputs = Input((3,))
    dense1 = Dense(units=128, activation="relu")(inputs)
    out = Dense(units=2)(dense1)
    model = Model(inputs=[inputs], outputs=[out])
    optimizer = Adam(learning_rate=1e-5)

    # Compile model
    model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['accuracy'])

    # model.summary()

    print("Training...")

    log_dir = "logs2/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Train model
    model.fit(X_train, y_train, epochs=100, batch_size=64, callbacks=[tensorboard_callback])


if __name__ == "__main__":
    main()



