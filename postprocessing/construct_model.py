import numpy as np
import pandas as pd
import cv2
import tqdm
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
import sklearn
import time
from sklearn.model_selection import GridSearchCV
atexit.register(cv2.destroyAllWindows)
warnings.simplefilter("ignore")

data = pd.read_csv('E:\\square_env_recordings_metadata_training\\dynamics_1.0s.csv')
hyperparams_csv = pd.DataFrame()

def evaluate_vectorial_model(predictions, test_labels):
    errors = abs(predictions - test_labels.reshape(len(test_labels, )))
    print('Average absolute error: {:0.2f} degrees.'.format(np.mean(errors)))
    print('Error coefficient: {:0.1f}%.'.format(100 * np.mean(errors) / np.std(test_labels) ) )
    return np.mean(errors), np.mean(errors) / np.std(test_labels)

def evaluate_magnitudal_model(predictions, test_labels):
    errors = abs(predictions - test_labels.reshape(len(test_labels,)))
    print('Average absolute error: {:0.2f} pixels.'.format(np.mean(errors)))
    print('Error coefficient: {:0.1f}%.'.format( 100 * np.mean(errors) / np.std(test_labels) ) )
    return np.mean(errors), np.mean(errors) / np.std(test_labels)

if __name__ == "__main__":

    # Remove outliers
    data = data[data["Action"] != -1]
    magn_cutoff = 20
    data = data[data["Magnitude"] > 1]
    data = data[data["Magnitude"] < magn_cutoff]
    data['Angle'] = np.degrees(np.arctan(data['dY'] / data['dX']))

    def normalized_sample(csv, feature, sampling=10, repeats=10):
        cols = csv.columns
        bins = csv[feature].value_counts().keys()
        new_csv = []
        for i in range(repeats):
            print(i)
            for n in bins:
                new_csv.append(csv[csv[feature] == n].sample(n=sampling, replace=True).values.tolist())
        new_shape = np.array(new_csv).shape
        return pd.DataFrame(np.array(new_csv).reshape((new_shape[0]*new_shape[1], new_shape[2])), columns=cols)

    # data = normalized_sample(csv=data, feature='Angle', sampling=10, repeats=100)
    # data = normalized_sample(csv=data, feature='Magnitude', sampling=10, repeats=100)
    # data = normalized_sample(csv=data, feature='Action', sampling=100, repeats=100)
    # data = normalized_sample(csv=data, feature='Vpp', sampling=10, repeats=100)
    # data = normalized_sample(csv=data, feature='Frequency', sampling=10, repeats=100)
    # print(data[['Action', 'Frequency', 'Magnitude', 'Vpp', 'dX', 'dY']].corr())

    # Prepare data
    inp = data[["Vpp", "Frequency", "Action", 'X0', 'Y0']]
    outp = data[["Angle", "Magnitude"]]

    TEST_SIZE = 0.2

    X_train, X_test, y_train, y_test = train_test_split(inp, outp, test_size=TEST_SIZE, random_state=1, shuffle=True)
    X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

    angles_train = y_train[:, 0]
    angles_test = y_test[:, 0]
    magns_train = y_train[:, -1]
    magns_test = y_test[:, -1]
    data_train = X_train[:, :3]
    data_test = X_test[:, :3]

    # print(angles_train.shape, magns_train.shape)
    total_train = np.array([angles_train, magns_train]).T
    total_test = np.array([angles_test, magns_test]).T

    model = sklearn.tree.DecisionTreeRegressor()
    trained_model = model.fit(data_train, total_train)



    model_name = 'DecisionTreeRegressor_angle_normed.pkl'
    with open(f'{model_name}', 'wb') as f:
        pickle.dump(trained_model, f)
        # trained_model = pickle.load(f)

    preds = trained_model.predict(data_test)
    angle_abs_error, angle_coeff = evaluate_vectorial_model(predictions=preds[:, 0],
                                                            test_labels=angles_test)
    magn_abs_error, magn_coeff = evaluate_magnitudal_model(predictions=preds[:, 1],
                                                           test_labels=magns_test)


    # preds_vect, pred_magn = multioutp_vect.predict(data_test), multioutp_magn.predict(data_test)
    # evaluate_vectorial_model(predictions=preds_vect, test_labels=angles_test)
    # evaluate_magnitudal_model(predictions=pred_magn, test_labels=magns_test)



    # Define regressor
    # model_magn = xgb.XGBRegressor(objective='reg:squarederror',
    #                              learning_rate=1,
    #                              max_depth=6,
    #                              n_estimators=100,
    #                              n_jobs=-1)
    # model_vect = xgb.XGBRegressor(objective='reg:squarederror',
    #                              learning_rate=1,
    #                              max_depth=6,
    #                              n_estimators=100,
    #                              n_jobs=-1)
    # model_magn = xgb.XGBRegressor()
    # model_vect = xgb.XGBRegressor()
    # model_magn = sklearn.tree.DecisionTreeRegressor()
    # model_vect = sklearn.tree.DecisionTreeRegressor()
    # model_vect = sklearn.ensemble.HistGradientBoostingRegressor(learning_rate=1)
    # model_magn = sklearn.ensemble.HistGradientBoostingRegressor(learning_rate=1)
    # model_vect = sklearn.ensemble.GradientBoostingRegressor(learning_rate=1)
    # model_magn = sklearn.ensemble.GradientBoostingRegressor(learning_rate=1)
    # model_vect = sklearn.ensemble.AdaBoostRegressor()
    # model_magn = sklearn.ensemble.AdaBoostRegressor()
    # model_vect = sklearn.ensemble.RandomForestRegressor()
    # model_magn = sklearn.ensemble.RandomForestRegressor()
    # model_magn = sklearn.ensemble.ExtraTreesRegressor()
    # model_vect = sklearn.ensemble.ExtraTreesRegressor()

    # parameters = {"splitter": ["best", "random"],
    #               "max_depth": [1, 5, 7, 11],
    #               "min_samples_leaf": [1, 3, 5, 6, 7],
    #               "min_weight_fraction_leaf": [0.1, 0.3, 0.5, 0.7],
    #               "max_features": ["auto", "log2", "sqrt", None],
    #               "max_leaf_nodes": [None, 10, 30, 50]}
    #
    # size = 1
    # for param in parameters:
    #     size *= len(parameters[f'{param}'])
    #
    # j = 0
    #
    # for a in parameters['splitter']:
    #     for b in parameters['max_depth']:
    #         for c in parameters['min_samples_leaf']:
    #             for d in parameters['min_weight_fraction_leaf']:
    #                 for e in parameters['max_features']:
    #                     for f in parameters['max_leaf_nodes']:
    #
    #
    #
    #                         j += 1
    #
    #                         try:
    #
    #                             hyperparams = {}
    #                             hyperparams.__setitem__('Model', 'DecisionTreeRegressor')
    #                             hyperparams.__setitem__('splitter', a)
    #                             hyperparams.__setitem__('max_depth', b)
    #                             hyperparams.__setitem__('min_samples_leaf', c)
    #                             hyperparams.__setitem__('min_weight_fraction_leaf', d)
    #                             hyperparams.__setitem__('max_features', e)
    #                             hyperparams.__setitem__('max_leaf_nodes', f)
    #
    #                             model = sklearn.tree.DecisionTreeRegressor(splitter=a, max_depth=b, min_samples_leaf=c,
    #                                                                        min_weight_fraction_leaf=d, max_features=e, max_leaf_nodes=f,
    #                                                                        random_state=1)
    #
    #                             # Fit data
    #                             # print('Training...')
    #                             t0 = time.time()
    #                             trained_model = model.fit(data_train, total_train)
    #
    #                             # Test data
    #                             # print('Predicting...')
    #                             times = []
    #                             for _ in range(1):
    #                                 for i in range(len(data_test) - 1):
    #                                     t0 = time.time()
    #                                     trained_model.predict(data_test[i][np.newaxis, :])
    #                                     times.append(time.time() - t0)
    #                             # print('Average prediction time: {:0.1f}\u03BCs'.format(np.mean(times) * 1e6))
    #
    #
    #
    #                             hyperparams.__setitem__('t_predict', np.round(np.mean(times) * 1e6, 3))
    #                             hyperparams.__setitem__('angle_abs_error', angle_abs_error)
    #                             hyperparams.__setitem__('angle_coeff', angle_coeff)
    #                             hyperparams.__setitem__('magn_abs_error', magn_abs_error)
    #                             hyperparams.__setitem__('magn_coeff', magn_coeff)
    #
    #                             hyperparams_csv.append(hyperparams, ignore_index=True)
    #
    #                         except:
    #                             print('Error')
    #                             pass
    #
    #                         print(time.time() - t0)
    #                         print(f'{j}/{size}', f'{round( (size-j) * (time.time() - t0), 0)}s left')
    #
    # hyperparams_csv.to_csv('Hyperparams0.csv')

    # multioutp_vect, multioutp_magn = model_vect.fit(data_train, angles_train), model_magn.fit(data_train, magns_train)

    # std = np.std([tree.feature_importances_ for tree in trained_model.estimators_], axis=0)
    # forest_importances = pd.Series(trained_model.feature_importances_, index=["Vpp", "Frequency", "Action"])
    #
    # fig, ax = plt.subplots()
    # forest_importances.plot.bar(yerr=std, ax=ax)
    # ax.set_title("Feature importances using MDI")
    # ax.set_ylabel("Mean decrease in impurity")
    # fig.tight_layout()
    # plt.show()


    # img = np.ones((1200, 1200, 3))
    # for i in range(len(y_test[:, 0])-1):
    #
    #     pos0 = np.array((int(round(X_test[i, 3])), int(round(X_test[i, 4])))) * 4
    #     dX_real = np.sqrt((total_test[i, 1] ** 2) / (1 + np.tan(np.radians(total_test[i, 0])) ** 2))
    #     dY_real = np.sqrt(total_test[i, 1] ** 2 - dX_real ** 2)
    #     vect_real = np.array([dX_real, dY_real])
    #     dX_pred = np.sqrt((preds[i, 1] ** 2) / (1 + np.tan(np.radians(preds[i, 0])) ** 2))
    #     dY_pred = np.sqrt(preds[i, 1] ** 2 - dX_pred ** 2)
    #     vect_pred = np.array([dX_pred, dY_pred])
    #
    #     pos_1_real = np.array(np.round(pos0 + 10*vect_real), dtype=int)
    #     pos_1_pred = np.array(np.round(pos0 + 10*vect_pred), dtype=int)
    #
    #     # vect_real = np.array([y_test[i, 0]*y_test[i, 2],
    #     #                       y_test[i, 1]*y_test[i, 2]]) * 4
    #
    #     # pos_1_real = np.array(np.round(pos0 + vect_real), dtype=int)
    #     #
    #     # vect_pred = np.array([preds[i, 0] * preds[i, 2],
    #     #                       preds[i, 1] * preds[i, 2]]) * 4
    #     #
    #     # pos_1_pred = np.array(np.round(pos0 + vect_pred), dtype=int)
    #
    #     vect_real = vect_real / np.linalg.norm(vect_real)
    #     vect_pred = vect_pred / np.linalg.norm(vect_pred)
    #
    #     # Pos0
    #     cv2.circle(img,
    #                tuple(pos0),
    #                0,
    #                (255, 0, 0),
    #                10)
    #
    #     # Pos0 to real pos1
    #     cv2.arrowedLine(img,
    #                     tuple(pos0),
    #                     tuple(pos_1_real),
    #                     (0, 255, 0),
    #                     1)
    #
    #     # Pos0 to predicted pos1
    #     cv2.arrowedLine(img,
    #                     tuple(pos0),
    #                     tuple(pos_1_pred),
    #                     (0, 0, 255),
    #                     1)
    #
    #     # Pos1_real
    #     cv2.circle(img,
    #                tuple(pos_1_real),
    #                0,
    #                (0, 255, 0),
    #                10)
    #
    #     # Pos1_predict
    #     cv2.circle(img,
    #                tuple(pos_1_pred),
    #                0,
    #                (0, 0, 255),
    #                10)
    #     cv2.imshow("Image", img)
    #     cv2.waitKey(0)






