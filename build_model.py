
from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.optimizers import Adam

# TODO --> Hyperparameter optimization
def build_model(save=False):

    # Build model
    inputs = Input(shape=(3,))
    dense1 = Dense(units=128,
                   activation="relu")(inputs)
    dense2 = Dense(units=128,
                   activation="relu")(dense1)
    out = Dense(units=2)(dense2)
    model = Model(inputs=[inputs],
                  outputs=[out])

    # Compile model
    optimizer = Adam(learning_rate=1e-5)
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    # Save model
    if save:
        save_model(model=model,
                   filepath="C:\\Users\\Matthijs\\PycharmProjects\\AI_Actuated_Microswarm_4\\models\\predict_state")

    return model

def train_model(model, X_train, y_train, save=False):

    # Train model
    model.fit(x=X_train,
              y=y_train,
              epochs=1,
              batch_size=64,
              shuffle=True)

    # Save model
    if save:
        save_model(model=model,
                   filepath="C:\\Users\\Matthijs\\PycharmProjects\\AI_Actuated_Microswarm_4\\models\\predict_state")

    return model



if __name__ == "__main__":

    build_model()