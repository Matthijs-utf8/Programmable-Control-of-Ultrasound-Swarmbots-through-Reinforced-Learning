# import tensorflow as tf
import numpy as np
import os
import re
from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import datetime
import matplotlib.pyplot as plt
from collections import deque

def get_data():


    DIR = "C:\\Users\\Matthijs\\PycharmProjects\\AI_Actuated_Microswarm_2\\Include\\snapshots\\Track_tests_friday\\"

    data = []

    X_train = []
    y_train = []

    x_que = deque(maxlen=10)
    y_que = deque(maxlen=10)

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

        x_que.append(x)
        y_que.append(y)

        diff_x = old_x - x
        data.append(diff_x)

        diff_y = old_y - y



        X_train.append([old_action/3])
        y_train.append([diff_x, diff_y])

        # X_train.append([old_action/3, old_x/300, old_y/300])
        # y_train.append([x/300, y/300])

        old_x = x
        old_y = y
        old_action = action

    plt.hist(data, bins=10, range=(-5, 5))
    plt.show()

    return np.array(X_train), np.array(y_train)

def make_data():

    increment = 1

    X_train = []
    y_train = []


    for y in range(300):
        if y % 2:
            for x in range(300):
                if x == 0:
                    X_train.append([1 / 3, x / 300, (y - increment) / 300])
                    y_train.append([x / 300, y / 300])
                else:
                    X_train.append([2 / 3, (x - increment) / 300, y / 300])
                    y_train.append([x / 300, y / 300])
        else:
            for x in list(reversed(range(300))):
                if x == 299:
                    X_train.append([1 / 3, x / 300, (y - increment) / 300])
                    y_train.append([x / 300, y / 300])
                else:
                    X_train.append([0 / 3, (x + increment) / 300, y / 300])
                    y_train.append([x / 300, y / 300])

    for x in range(300):
        if x % 2:
            for y in list(reversed(range(300))):
                if y == 299:
                    X_train.append([2 / 3, (x - increment) / 300, y / 300])
                    y_train.append([x / 300, y / 300])
                else:
                    X_train.append([1 / 3, x / 300, (y + increment) / 300])
                    y_train.append([x / 300, y / 300])
        else:
            for y in range(300):
                if y == 0:
                    X_train.append([2 / 3, (x - increment) / 300, y / 300])
                    y_train.append([x / 300, y / 300])
                else:
                    X_train.append([3 / 3, x / 300, (y - increment) / 300])
                    y_train.append([x / 300, y / 300])
    return np.array(X_train), np.array(y_train)

X_train, y_train = get_data()
print(X_train.shape)
# print(X_train[900:1000].tolist())
# print(y_train[900:1000].tolist())
# X_train, y_train = make_data()


def main():



    # Build model
    # inputs = Input((X_train.shape[1], 1))
    # dense1 = LSTM(units=32, activation="relu")(inputs)
    inputs = Input((X_train.shape[1],))
    # dense1 = Dense(units=128, activation="relu")(inputs)
    dense2 = Dense(units=128, activation="relu")(inputs)
    out = Dense(units=2)(dense2)
    model = Model(inputs=[inputs], outputs=[out])
    optimizer = Adam(learning_rate=1e-4)

    # Compile model
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

    # model.summary()

    print("Training...")

    # log_dir = "C:\\Users\\Matthijs\\PycharmProjects\\AI_Actuated_Microswarm_4\\logs2\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Train model
    model.fit(X_train, y_train, epochs=2000, batch_size=256, shuffle=True)

    save_model(model=model, filepath="C:\\Users\\Matthijs\\PycharmProjects\\AI_Actuated_Microswarm_4\\model4")


if __name__ == "__main__":
    main()

    nnet = load_model("C:\\Users\\Matthijs\\PycharmProjects\\AI_Actuated_Microswarm_4\\model4")
    print(np.array(nnet.predict(np.array([[1/3, 1/2, 1/2]])))*300)


