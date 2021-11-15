import numpy as np
import pandas as pd
import cv2
from tqdm.auto import tqdm
from ast import literal_eval as make_tuple
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import pickle
import warnings
import atexit
import os
atexit.register(cv2.destroyAllWindows)
warnings.simplefilter("ignore")

# PROCESSED_CSV = 'E:\\metadata_for_new_model4_tracked2_processed.csv'
HYPERPARAMS_CSV = 'hyperparams_xgb.csv'

processed_csv = pd.DataFrame()
for file in os.listdir('training_data'):
    csv = pd.read_csv(f'training_data\\{file}')
    del csv['Unnamed: 0']
    del csv['Index']
    csv = csv[csv['Action'] != -1]
    processed_csv = processed_csv.append(csv)


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

    rf.fit(X_train, y_train)

    if filename:
        if ".pkl" not in filename:
            with open(f"{filename}.pkl", 'wb') as f:
                pickle.dump(rf, f)
        else:
            with open(filename, 'wb') as f:
                pickle.dump(rf, f)

    return rf


def evaluate_vectorial_model(predictions, test_labels):
    errors = abs(predictions - test_labels)
    errors = np.degrees(np.arctan2(*errors)) % 360.0
    print('Vectorial model Performance; Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    return predictions, np.mean(errors)


def evaluate_magnitudal_model(predictions, test_labels):
    errors = abs(predictions - test_labels.reshape(len(test_labels,)))
    print('Magnitude model Performance; Average Error: {:0.4f} pixels.'.format(np.mean(errors)))
    return predictions, np.mean(errors)

def evaluate_model(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - np.array(test_labels))
    print('Magnitude model Performance; Average Error: {:0.4f} pixels.'.format(np.mean(errors)))
    return predictions, np.mean(errors)


if __name__ == "__main__":

    data = processed_csv
    print(data.shape)
    # del data['Unnamed: 0']

    # try:
    #     hyperparams_metadata = pd.read_csv(HYPERPARAMS_CSV)  # Make this file if you don't have it yet
    #     del hyperparams_metadata['Unnamed: 0']  # Remove unwanted column
    # except:
    #     processed_metadata = pd.DataFrame(
    #         {"Time": -1}
    #     )

    # Remove outliers
    data = data[data["Action"] != -1]
    magn_cutoff = 20
    # data = data[data["Magnitude"] > 1]
    data = data[data["Magnitude"] < magn_cutoff]

    # Prepare data
    inp = data[["Vpp", "Frequency", "Size", "Action", "X0", "Y0"]]
    # outp_dx = data[['dX']]
    # outp_dy = data[['dY']]
    # outp_magn = data[['Magnitude']]
    outp = data[["dX", "dY", "Magnitude"]]

    # data_dmatrix_dx = xgb.DMatrix(data=inp, label=outp_dx)
    # data_dmatrix_dy = xgb.DMatrix(data=inp, label=outp_dy)
    # data_dmatrix_magn = xgb.DMatrix(data=inp, label=outp_magn)

    TEST_SIZE = 0.2

    # X_train, X_test, y_train_dx, y_test_dx = train_test_split(inp, outp_dx, test_size=TEST_SIZE, random_state=0)
    # _, _, y_train_dy, y_test_dy = train_test_split(inp, outp_dy, test_size=TEST_SIZE, random_state=0)
    # _, _, y_train_magn, y_test_magn = train_test_split(inp, outp_magn, test_size=TEST_SIZE, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(inp, outp, test_size=TEST_SIZE, random_state=1, shuffle=True)
    X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
    # y_test_dx, y_test_dy, y_test_magn =

    params = {"objective": "reg:squarederror", 'colsample_bytree': 0.9, 'learning_rate': 0.5,
              'max_depth': 50, 'alpha': 1, 'lambda': 1}
    #
    # # cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
    # #                     num_boost_round=50, early_stopping_rounds=10, metrics="rmse", as_pandas=True, seed=123)
    #
    # Define regressor
    # xg_reg_dx = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.9, learning_rate=0.5,
    #                           max_depth=50, alpha=1, n_estimators=100, n_jobs=6)
    # xg_reg_dy = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.9, learning_rate=0.5,
    #                           max_depth=50, alpha=1, n_estimators=100, n_jobs=6)
    # xg_reg_magn = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.9, learning_rate=0.5,
    #                           max_depth=50, alpha=1, n_estimators=100, n_jobs=6)
    xg_reg = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.9, learning_rate=0.5,
                              max_depth=50, alpha=1, n_estimators=100, n_jobs=6)




    # xg_reg_dx = xgb.train(params=params, dtrain=data_dmatrix_dx, num_boost_round=50)
    # xg_reg_dy = xgb.train(params=params, dtrain=data_dmatrix_dy, num_boost_round=50)
    # xg_reg_magn = xgb.train(params=params, dtrain=data_dmatrix_magn, num_boost_round=50)
    #
    # xg_reg_dx = xgb.cv(dtrain=data_dmatrix_dx, params=params, nfold=5,
    #                     num_boost_round=50, early_stopping_rounds=10, metrics="mae", as_pandas=True, seed=0)
    # xg_reg_dy = xgb.cv(dtrain=data_dmatrix_dy, params=params, nfold=5,
    #                     num_boost_round=50, early_stopping_rounds=10, metrics="mae", as_pandas=True, seed=0)
    # xg_reg_magn = xgb.cv(dtrain=data_dmatrix_magn, params=params, nfold=5,
    #                     num_boost_round=50, early_stopping_rounds=10, metrics="mae", as_pandas=True, seed=0)
    #
    # Fit data
    print('Training...')
    # xg_reg_dx.fit(X_train, y_train_dx)
    # xg_reg_dy.fit(X_train, y_train_dy)
    # xg_reg_magn.fit(X_train, y_train_magn)
    # xg_reg.fit(X_train, y_train)
    multioutp = MultiOutputRegressor(xg_reg, n_jobs=-1).fit(X_train, y_train)

    # with open('rx_reg_dx2.pkl', 'wb') as f:
    #     pickle.dump(xg_reg_dx, f)
    #     # xg_reg_dx = pickle.load(f)
    # with open('rx_reg_dy2.pkl', 'wb') as f:
    #     pickle.dump(xg_reg_dy, f)
    #     # xg_reg_dy = pickle.load(f)
    # with open('rx_reg_magn2.pkl', 'wb') as f:
    #     pickle.dump(xg_reg_magn, f)
    #     # xg_reg_magn = pickle.load(f)
    with open('rx_reg2.pkl', 'wb') as f:
        pickle.dump(multioutp, f)
        # multioutp = pickle.load(f)

    # Test data
    print('Predicting...')
    preds = multioutp.predict(X_test)

    # preds_dx = xg_reg_dx.predict(X_test)
    # preds_dy = xg_reg_dy.predict(X_test)
    # preds_magn = xg_reg_magn.predict(X_test)

    vect_pred = np.squeeze(np.array([y_test[:, 0], y_test[:, 1]]))
    vect_real = np.array([preds[:, 0], preds[:, 1]])

    evaluate_vectorial_model(predictions=vect_pred, test_labels=vect_real)
    evaluate_magnitudal_model(predictions=preds[:, 2], test_labels=y_test[:, 2])

    # xgb.plot_importance(xg_reg_dx)
    # plt.rcParams['figure.figsize'] = [5, 5]
    # plt.show()
    # xgb.plot_importance(xg_reg_dy)
    # plt.rcParams['figure.figsize'] = [5, 5]
    # plt.show()
    # xgb.plot_importance(xg_reg_magn)
    # plt.rcParams['figure.figsize'] = [5, 5]
    # plt.show()

    # test = mean_absolute_error(y_true=preds_dx, y_pred=y_test_dx.values)
    # print(test)
    # preds_dx, errors_dx = evaluate_model(model=xg_reg_dx, test_features=X_test, test_labels=y_test_dx)
    # preds_dy, errors_dy = evaluate_model(model=xg_reg_dy, test_features=X_test, test_labels=np.array(y_test_dy))
    # preds_magn, errors_magn = evaluate_model(model=xg_reg_magn, test_features=X_test, test_labels=np.array(y_test_magn))

    # # Define hyperparameters
    # n_estimators = [int(x) for x in np.linspace(start=200, stop=1000, num=4)]
    # max_features = ['auto']
    # max_depth = [int(x) for x in np.linspace(20, 100, num=4)]
    # min_samples_split = [2, 5, 10]
    # min_samples_leaf = [1, 2, 4]
    # bootstrap = [True]
    #
    # for a in tqdm(n_estimators):
    #     for b in max_features:
    #         for c in tqdm(max_depth):
    #             for d in tqdm(min_samples_split):
    #                 for e in tqdm(min_samples_leaf):
    #                     for f in bootstrap:
    #
    #                             vect_model = train_rf(n_estimators=a,
    #                                                   max_features=b,
    #                                                   max_depth=c,
    #                                                   min_samples_split=d,
    #                                                   min_samples_leaf=e,
    #                                                   bootstrap=f,
    #                                                   X_train=X_train_vect,
    #                                                   y_train=y_train_vect,
    #                                                   filename="")
    #                             magn_model = train_rf(n_estimators=a,
    #                                                   max_features=b,
    #                                                   max_depth=c,
    #                                                   min_samples_split=d,
    #                                                   min_samples_leaf=e,
    #                                                   bootstrap=f,
    #                                                   X_train=X_train_magn,
    #                                                   y_train=y_train_magn,
    #                                                   filename="")
    #
    #                             vect_predict, mean_vect_error = evaluate_vectorial_model(model=vect_model,
    #                                                                                     test_features=X_test_vect,
    #                                                                                     test_labels=y_test_vect)
    #                             magn_predict, mean_magn_error = evaluate_magnitudal_model(model=magn_model,
    #                                                                                      test_features=X_test_magn,
    #                                                                                      test_labels=y_test_magn)
    #
    #                             real_pos_1 = np.array(X_test_vect[:, 4:5] + y_test_vect * y_test_magn)
    #                             predicted_pos_1 = np.array(np.round(X_test_vect[:, 4:5] + vect_predict * magn_predict.reshape((len(magn_predict), 1))), dtype=int)
    #
    #                             avg_movement = np.mean(np.linalg.norm(real_pos_1 - predicted_pos_1, axis=1))
    #                             avg_total_error = np.mean(np.linalg.norm(real_pos_1 - X_test_vect[:, 4:5], axis=1))
    #
    #                             results = {'magn_cutoff': magn_cutoff,
    #                                        'n_estimators': a,
    #                                        'max_features': b,
    #                                        'max_depth': c,
    #                                        'min_samples_split': d,
    #                                        'min_samples_leaf': e,
    #                                        'bootstrap': f,
    #                                        'AvgMovement': avg_movement,
    #                                        'AvgMagnitudeError': mean_magn_error,
    #                                        'AvgAngleError': mean_vect_error,
    #                                        'AvgTotalError': avg_total_error,
    #                                        'ErrorCoefficient': avg_total_error / avg_movement}
    #
    #                             hyperparams_metadata = hyperparams_metadata.append(results, ignore_index=True)
    #                             hyperparams_metadata.to_csv("model_performance_real.csv")
    #
    # best_idx = hyperparams_metadata["ErrorCoefficient"].idxmax()
    #
    # vect_model = train_rf(  n_estimators = hyperparams_metadata["n_estimators"][best_idx],
    #                         max_features = hyperparams_metadata["max_features"][best_idx],
    #                         max_depth = hyperparams_metadata["max_depth"][best_idx],
    #                         min_samples_split = hyperparams_metadata["min_samples_split"][best_idx],
    #                         min_samples_leaf = hyperparams_metadata["min_samples_leaf"][best_idx],
    #                         bootstrap = hyperparams_metadata["bootstrap"][best_idx],
    #                         X_train=X_train_vect,
    #                         y_train=y_train_vect,
    #                         filename="vect_model.pkl")
    # magn_model = train_rf(  n_estimators = hyperparams_metadata["n_estimators"][best_idx],
    #                         max_features = hyperparams_metadata["max_features"][best_idx],
    #                         max_depth = hyperparams_metadata["max_depth"][best_idx],
    #                         min_samples_split = hyperparams_metadata["min_samples_split"][best_idx],
    #                         min_samples_leaf = hyperparams_metadata["min_samples_leaf"][best_idx],
    #                         bootstrap = hyperparams_metadata["bootstrap"][best_idx],
    #                         X_train=X_train_magn,
    #                         y_train=y_train_magn,
    #                         filename="magn_model.pkl")


    # with open('vectorial_random_best.pkl', 'rb') as f:
    #     vect_model = pickle.load(f)
    # with open('magnitudal.pkl', 'rb') as f:
    #     magn_model = pickle.load(f)


    img = np.ones((1200, 1200, 3))
    for i in range(len(y_test[:, 0])-1):

        pos0 = np.array((int(round(X_test[i, 4])), int(round(X_test[i, 5])))) * 4

        vect_real = np.array([y_test[i, 0]*y_test[i, 2],
                              y_test[i, 1]*y_test[i, 2]]) * 4




        pos_1_real = np.array(np.round(pos0 + vect_real), dtype=int)

        vect_pred = np.array([preds[i, 0] * preds[i, 2],
                              preds[i, 1] * preds[i, 2]]) * 4




        pos_1_pred = np.array(np.round(pos0 + vect_pred), dtype=int)

        vect_real = vect_real / np.linalg.norm(vect_real)
        vect_pred = vect_pred / np.linalg.norm(vect_pred)

        # Pos0
        cv2.circle(img,
                   tuple(pos0),
                   0,
                   (255, 0, 0),
                   4)

        # Pos0 to real pos1
        cv2.arrowedLine(img,
                        tuple(pos0),
                        tuple(np.array(pos0 + vect_real * 20, dtype=int)),
                        (0, 255, 0),
                        1)

        # Pos0 to predicted pos1
        cv2.arrowedLine(img,
                        tuple(pos0),
                        tuple(np.array(pos0 + vect_pred * 20, dtype=int)),
                        (0, 0, 255),
                        1)

        # Pos1_real
        cv2.circle(img,
                   tuple(pos_1_real),
                   0,
                   (0, 255, 0),
                   5)

        # Pos1_predict
        cv2.circle(img,
                   tuple(pos_1_pred),
                   0,
                   (0, 0, 255),
                   3)
        cv2.imshow("Image", img)
        cv2.waitKey(0)



