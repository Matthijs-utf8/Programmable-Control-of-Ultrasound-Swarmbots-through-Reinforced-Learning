import numpy as np
import pandas as pd
import cv2
from tqdm.auto import tqdm
from ast import literal_eval as make_tuple
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import warnings
warnings.simplefilter("ignore")


def preprocess_data(data, mode="vect"):

    # Define features and labels
    X = np.array(data[["Vpp", "Frequency", "Size", "Action", "X0", "Y0"]], dtype=np.float32)
    if mode == "vect":
        y = np.array(data[["dX", "dY"]], dtype=np.float32)
    elif mode == "magn":
        y = np.array(data[['Magnitude']], dtype=np.float32)
    else:
        return None

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Return train and test data
    return X_train, X_test, y_train, y_test


def train_rf(X_train,
             y_train,
             filename="",
             n_estimators=200,
             min_samples_split=10,
             min_samples_leaf=2,
             max_features="sqrt",
             max_depth=100,
             bootstrap=False):

    rf = RandomForestRegressor(n_estimators=n_estimators,
                               min_samples_split=min_samples_split,
                               min_samples_leaf=min_samples_leaf,
                               max_features=max_features,
                               max_depth=max_depth,
                               bootstrap=bootstrap,
                               n_jobs=-1)

    # if hp_optimize:
    #
    #
    #     pprint(random_grid)
    #
    #
    #     rf_random = RandomizedSearchCV(estimator=rf,
    #                                    param_distributions=random_grid,
    #                                    n_iter=100,
    #                                    cv=3,
    #                                    verbose=2,
    #                                    random_state=0,
    #                                    n_jobs=-1)
    #     rf_random.fit(X_train, y_train)
    #     best_random_model = rf_random.best_params_
    #
    #     print(best_random_model)
    #
    #     return best_random_model
    #
    # else:
    rf.fit(X_train, y_train)

    if filename:
        if ".pkl" not in filename:
            with open(f"{filename}.pkl", 'wb') as f:
                pickle.dump(rf, f)
        else:
            with open(filename, 'wb') as f:
                pickle.dump(rf, f)

    return rf


def evaluate_vectorial_model(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    errors = np.degrees(np.arctan2(*errors.T[::-1])) % 360.0
    print('Vectorial model Performance; Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    return predictions, np.mean(errors)


def evaluate_magnitudal_model(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels.reshape(len(test_labels,)))
    print('Magnitude model Performance; Average Error: {:0.4f} pixels.'.format(np.mean(errors)))
    return predictions, np.mean(errors)


if __name__ == "__main__":

    data = pd.read_csv("processed_csv.csv")
    hyperparams = pd.read_csv("model_performance_real.csv")

    # Remove unwanted column
    del data['Unnamed: 0']
    del hyperparams['Unnamed: 0']

    # Get absolute position of swarm as variables for input to model
    locs = np.array(list(map(make_tuple, data["Pos0"])))
    data["X"] = locs[:, 0]
    data["Y"] = locs[:, 1]

    # Remove irrelevant data
    del data["Vector"]
    del data["Time"]
    del data["Cluster"]
    del data["Pos0"]
    del data["Pos1"]
    data = data[data["Action"] != -1]

    # Remove outliers
    magn_cutoff = 20
    data = data[data["Magnitude"] < magn_cutoff]

    X_train_vect, X_test_vect, y_train_vect, y_test_vect = preprocess_data(data=data, mode="vect")
    X_train_magn, X_test_magn, y_train_magn, y_test_magn = preprocess_data(data=data, mode="magn")

    # Define hyperparameters
    n_estimators = [int(x) for x in np.linspace(start=200, stop=1000, num=4)]
    max_features = ['auto']
    max_depth = [int(x) for x in np.linspace(20, 100, num=4)]
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True]

    for a in tqdm(n_estimators):
        for b in max_features:
            for c in tqdm(max_depth):
                for d in tqdm(min_samples_split):
                    for e in tqdm(min_samples_leaf):
                        for f in bootstrap:

                                vect_model = train_rf(n_estimators=a,
                                                      max_features=b,
                                                      max_depth=c,
                                                      min_samples_split=d,
                                                      min_samples_leaf=e,
                                                      bootstrap=f,
                                                      X_train=X_train_vect,
                                                      y_train=y_train_vect,
                                                      filename="")
                                magn_model = train_rf(n_estimators=a,
                                                      max_features=b,
                                                      max_depth=c,
                                                      min_samples_split=d,
                                                      min_samples_leaf=e,
                                                      bootstrap=f,
                                                      X_train=X_train_magn,
                                                      y_train=y_train_magn,
                                                      filename="")

                                vect_predict, mean_vect_error = evaluate_vectorial_model(model=vect_model,
                                                                                        test_features=X_test_vect,
                                                                                        test_labels=y_test_vect)
                                magn_predict, mean_magn_error = evaluate_magnitudal_model(model=magn_model,
                                                                                         test_features=X_test_magn,
                                                                                         test_labels=y_test_magn)

                                real_pos_1 = np.array(X_test_vect[:, 4:5] + y_test_vect * y_test_magn)
                                predicted_pos_1 = np.array(np.round(X_test_vect[:, 4:5] + vect_predict * magn_predict.reshape((len(magn_predict), 1))), dtype=int)

                                avg_movement = np.mean(np.linalg.norm(real_pos_1 - predicted_pos_1, axis=1))
                                avg_total_error = np.mean(np.linalg.norm(real_pos_1 - X_test_vect[:, 4:5], axis=1))

                                results = {'magn_cutoff': magn_cutoff,
                                           'n_estimators': a,
                                           'max_features': b,
                                           'max_depth': c,
                                           'min_samples_split': d,
                                           'min_samples_leaf': e,
                                           'bootstrap': f,
                                           'AvgMovement': avg_movement,
                                           'AvgMagnitudeError': mean_magn_error,
                                           'AvgAngleError': mean_vect_error,
                                           'AvgTotalError': avg_total_error,
                                           'ErrorCoefficient': avg_total_error / avg_movement}

                                hyperparams = hyperparams.append(results, ignore_index=True)
                                hyperparams.to_csv("model_performance_real.csv")

    best_idx = hyperparams["ErrorCoefficient"].idxmax()

    vect_model = train_rf(  n_estimators = hyperparams["n_estimators"][best_idx],
                            max_features = hyperparams["max_features"][best_idx],
                            max_depth = hyperparams["max_depth"][best_idx],
                            min_samples_split = hyperparams["min_samples_split"][best_idx],
                            min_samples_leaf = hyperparams["min_samples_leaf"][best_idx],
                            bootstrap = hyperparams["bootstrap"][best_idx],
                            X_train=X_train_vect,
                            y_train=y_train_vect,
                            filename="vect_model.pkl")
    magn_model = train_rf(  n_estimators = hyperparams["n_estimators"][best_idx],
                            max_features = hyperparams["max_features"][best_idx],
                            max_depth = hyperparams["max_depth"][best_idx],
                            min_samples_split = hyperparams["min_samples_split"][best_idx],
                            min_samples_leaf = hyperparams["min_samples_leaf"][best_idx],
                            bootstrap = hyperparams["bootstrap"][best_idx],
                            X_train=X_train_magn,
                            y_train=y_train_magn,
                            filename="magn_model.pkl")


    # with open('vectorial_random_best.pkl', 'rb') as f:
    #     vect_model = pickle.load(f)
    # with open('magnitudal.pkl', 'rb') as f:
    #     magn_model = pickle.load(f)


    # img = np.ones((300, 300, 3))
    # for i in range(len(y_test_vect)-1):
    #     cv2.circle(img,
    #                (int(round(X_test_vect[:, 4][i])), int(round(X_test_vect[:, 5][i]))),
    #                0,
    #                (255, 0, 0),
    #                5)
    #     cv2.arrowedLine(img,
    #                     (int(round(X_test_vect[:, 4][i])), int(round(X_test_vect[:, 5][i]))),
    #                     tuple(np.array(np.array((int(round(X_test_vect[:, 4][i])), int(round(X_test_vect[:, 5][i])))) + y_test_vect[i] * 30, dtype=np.int)),
    #                     (0, 255, 0),
    #                     1)
    #     cv2.arrowedLine(img,
    #                     (int(round(X_test_vect[:, 4][i])), int(round(X_test_vect[:, 5][i]))),
    #                     tuple(np.array(np.array((int(round(X_test_vect[:, 4][i])), int(round(X_test_vect[:, 5][i])))) + vect_predict[i] * 30, dtype=np.int)),
    #                     (0, 0, 255),
    #                     1)
    #
    #     cv2.circle(img,
    #                (int(np.round(X_test_vect[:, 4][i] + y_test_vect[i][0] * y_test_magn[i])),
    #                 int(np.round(X_test_vect[:, 5][i] + y_test_vect[i][1] * y_test_magn[i]))),
    #                0,
    #                (0, 255, 0),
    #                5)
    #     cv2.circle(img,
    #                (int(np.round(X_test_vect[:, 4][i] + vect_predict[i][0] * magn_predict[i])),
    #                 int(np.round(X_test_vect[:, 5][i] + vect_predict[i][1] * magn_predict[i]))),
    #                0,
    #                (0, 0, 255),
    #                3)
    #     cv2.imshow("Image", img)
    #     cv2.waitKey(0)



