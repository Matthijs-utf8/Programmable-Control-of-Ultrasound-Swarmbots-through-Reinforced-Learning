import numpy as np
import pandas as pd
import cv2
from tqdm.auto import tqdm
from ast import literal_eval as make_tuple
import matplotlib.pyplot as plt
import pylab
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import pickle
import warnings
import atexit
import os
from matplotlib import style
from matplotlib import colors
from scipy.stats import kde
import seaborn as sns
atexit.register(cv2.destroyAllWindows)
warnings.simplefilter("ignore")
style.use('ggplot')

# processed_csv = pd.read_csv('normalized_magnitude_training_data.csv')
# processed_csv = processed_csv[["Vpp", "Frequency", "Size", "Action", 'Magnitude']]
# # print(processed_csv[["Vpp", "Frequency", "Size", "Action", 'Magnitude']].corr())
#
# action_to_vpp = {0: 3, 1: 1, 2: 2, 3: 3}
#
# def size_to_vpp(size, max_size=30):
#     return size / (30 / 2)
#
# def state_to_vpp(size, action):
#     return min(MIN_VPP + MIN_VPP * size_to_vpp(size) * action_to_vpp[action], MAX_VPP)
#
# for n, i in processed_csv.iterrows():
#     print(i.to_list())
#     print(2 + 2 * size_to_vpp(i['Size']) * action_to_vpp[i['Action']])

# PROCESSED_CSV = 'E:\\metadata_for_new_model4_tracked2_processed.csv'
HYPERPARAMS_CSV = 'hyperparams_xgb.csv'
piezo = 1

print('Loading data...')
processed_csv = pd.DataFrame()
a = 1
for file in os.listdir('training_data'):
    csv = pd.read_csv(f'training_data\\{file}')
    del csv['Unnamed: 0']
    del csv['Index']
    csv = csv[csv['Action'] != -1]
    # csv = csv[(csv['Action'] == 0) | (csv['Action'] == 2)]
    csv = csv[csv['Magnitude'] <= 50]
    csv = csv[csv['Magnitude'] >= 0]
    # csv = csv[csv['Size'] > 10]
    csv = csv[csv['Size'] < 50]
    # csv = csv[csv['Size'] <= 15]
    csv = csv[csv['Vpp'] >= 2]
    processed_csv = processed_csv.append(csv)
    # if a > 2:
    #     break
    a += 1

# plt.hist(processed_csv['Magnitude'])
# plt.show()

print(processed_csv.shape)
# print()
# print(processed_csv['Vpp'].value_counts(normalize=True, bins = 5))

range_freq = min(processed_csv['Frequency'].value_counts(bins = 5).keys())
range_vpp = min(processed_csv['Vpp'].value_counts(bins = 5).keys())
range_size = min(processed_csv['Size'].value_counts(bins = 5).keys())
range_magn = min(processed_csv['Magnitude'].value_counts(bins = 4).keys())
# print(processed_csv['Magnitude'].value_counts())


print(range_freq, range_vpp, range_size)
# print(range_freq.left, range_freq.right)

# processed_csv = processed_csv[(processed_csv['Frequency'] < range_freq.left) |
#                               (processed_csv['Frequency'] >= range_freq.right)]
# processed_csv = processed_csv[(processed_csv['Vpp'] < range_vpp.left) |
#                               (processed_csv['Vpp'] >= range_vpp.right)]
# processed_csv = processed_csv[(processed_csv['Size'] < range_size.left) |
#                               (processed_csv['Size'] >= range_size.right)]
processed_csv = processed_csv[(processed_csv['Magnitude'] < range_magn.left) |
                              (processed_csv['Magnitude'] >= range_magn.right)]
# range_magn = min(processed_csv['X0'].value_counts(bins = 10).keys())
# processed_csv = processed_csv[(processed_csv['X0'] < range_magn.left) |
#                               (processed_csv['X0'] >= range_magn.right)]
# range_magn = min(processed_csv['Y0'].value_counts(bins = 10).keys())
# processed_csv = processed_csv[(processed_csv['Y0'] < range_magn.left) |
#                               (processed_csv['Y0'] >= range_magn.right)]



print(processed_csv.shape)

# rocessed_csv = processed_csv[processed_csv['Frequency'] / processed_csv['Frequencz'].count()]
feature = 'Size'
# print([processed_csv[col].value_counts() for col in [feature]])
least = min([min(processed_csv[col].value_counts()) for col in [feature]])
vpps = processed_csv[feature].value_counts().keys()

new_csv = pd.DataFrame()

for n in vpps:
    new_csv = new_csv.append(processed_csv[processed_csv[feature] == n].sample(n=least))

print(new_csv.shape)

processed_csv = new_csv.copy()

feature = 'Vpp'
# print([processed_csv[col].value_counts() for col in [feature]])
least = min([min(processed_csv[col].value_counts()) for col in [feature]])
vpps = processed_csv[feature].value_counts().keys()

new_csv = pd.DataFrame()

for n in vpps:
    new_csv = new_csv.append(processed_csv[processed_csv[feature] == n].sample(n=least))

print(new_csv.shape)

processed_csv = new_csv.copy()
#

feature = 'Magnitude'
# print([processed_csv[col].value_counts() for col in [feature]])
least = min([min(processed_csv[col].value_counts()) for col in [feature]])
vpps = processed_csv[feature].value_counts().keys()

new_csv = pd.DataFrame()

for n in vpps:
    new_csv = new_csv.append(processed_csv[processed_csv[feature] == n].sample(n=least))

print(new_csv.shape)

processed_csv = new_csv.copy()

# processed_csv.to_csv('normalized_magnitude_training_data.csv')



# counts_vpp = dict(zip(processed_csv['Vpp'].value_counts().keys(), processed_csv['Vpp'].value_counts().values))
# counts_freq = dict(zip(processed_csv['Frequency'].value_counts().keys(), processed_csv['Frequency'].value_counts().values))
# counts_size = dict(zip(processed_csv['Size'].value_counts().keys(), processed_csv['Size'].value_counts().values))
#
#
# processed_csv['Vpp'] = processed_csv['Vpp'].apply(lambda x: x / counts_vpp[x])
# processed_csv['Frequency'] = processed_csv['Frequency'].apply(lambda x: x / counts_freq[x])
# processed_csv['Size'] = processed_csv['Size'].apply(lambda x: x / counts_size[x])
#
# max_vpp = max(processed_csv['Vpp'])
# max_freq = max(processed_csv['Frequency'])
# max_size = max(processed_csv['Size'])
#
# processed_csv['Vpp'] = processed_csv['Vpp'].apply(lambda x: x / max_vpp * 5)
# processed_csv['Frequency'] = processed_csv['Frequency'].apply(lambda x: x / max_freq * 20 + 240)
# processed_csv['Size'] = processed_csv['Size'].apply(lambda x: x / max_size * 20)

# processed_

# print(processed_csv['Size'].iloc[0])
# print(processed_csv.shape)

# processed_csv = processed_csv.sample(frac=0.1, random_state=1, ignore_index=True)



# def mean_norm(df):
#     return df.apply(lambda x: (x-x.mean())/ x.std(), axis=0)
#
# def minmax_norm(df):
#     return (df - df.min()) / ( df.max() - df.min())
#
# def quantile_norm(df_input):
#     sorted_df = pd.DataFrame(np.sort(df_input.values,axis=0), index=df_input.index, columns=df_input.columns)
#     mean_df = sorted_df.mean(axis=1)
#     mean_df.index = np.arange(1, len(mean_df) + 1)
#     quantile_df =df_input.rank(method="min").stack().astype(int).map(mean_df).unstack()
#     return(quantile_df)

if __name__ == "__main__":
    # pass

    # print(processed_csv.columns)
    #
    # plt.figure(5, figsize=(10, 10))
    # plt.title('Vpp vs. Magnitude')
    # plt.hist2d(processed_csv['Frequency'], processed_csv['dX'])
    # # plt.colorbar()
    # plt.tight_layout()
    # plt.show()
    #
    # plt.figure(6, figsize=(10, 10))
    # plt.title('Vpp vs. Magnitude')
    # plt.hist2d(processed_csv['Frequency'], processed_csv['dY'])
    # # plt.colorbar()
    # plt.tight_layout()
    # plt.show()

    # sns.pairplot(processed_csv[['Vpp', 'Frequency', 'Size', 'Action', 'Magnitude']], kind='reg')
    # plt.show()

    # data = processed_csv
    # print(data.shape)
    #
    # print(x, y)
    # data['Frequency'] = data['Frequency'] - 240
    # data['Vpp'] = data['Vpp'].resample(0.1)
    # print()

    # data = mean_norm(data)

    # for column in data:
    #     data[column] = mean_norm(data[column])

    # plt.figure(0, figsize=(10, 10))
    # plt.title('dX')
    # plt.semilogy()
    # plt.hist(data['dX'], range=(-1, 1), bins=40)
    # # plt.show()
    #
    # plt.figure(1, figsize=(10, 10))
    # plt.title('dY')
    # plt.semilogy()
    # plt.hist(data['dY'], range=(-1, 1), bins=40)
    # # plt.show()
    #
    # plt.figure(2, figsize=(10, 10))
    # plt.title('Magnitude')
    # plt.semilogy()
    # plt.hist(data['Magnitude'], range=(0, 20), bins=35)
    # # plt.show()
    #
    # plt.figure(3, figsize=(10, 10))
    # plt.title('Vpp')
    # # plt.semilogy()
    # plt.hist2d(processed_csv['Vpp'],
    #            processed_csv['Size'],
    #            bins=[6, 20],
    #            density=True,
    #            weights=processed_csv['Magnitude'],
    #            cmap=plt.cm.BuGn_r,
    #            norm=colors.PowerNorm(0.75))
    # plt.colorbar()
    # plt.tight_layout()
    # plt.show()

    fig1 = plt.figure(4, figsize=(10, 10))
    # plt.title(f'dX vs. Frequency (Piezo {piezo})')
    plt.ylabel('Size')
    plt.xlabel('Vpp')
    # plt.hist2d(data['Frequency'], data['Magnitude'], norm=colors.PowerNorm(0.25), bins=5, cmap=plt.cm.BuGn_r)
    data = processed_csv[['Vpp', 'Size']].to_numpy() * processed_csv[['Magnitude']].to_numpy()
    x, y = data.T
    # y = y / (np.max(y) - np.min(y)) * 20 + 240
    # x = x / (np.max(x) - np.min(x)) * 3
    nbins = 30
    k = kde.gaussian_kde(data.T)
    xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', norm=colors.PowerNorm(0.5), cmap='rainbow')
    # plt.contour(xi, yi, zi.reshape(xi.shape))
    plt.colorbar()
    plt.tight_layout()
    # fig1.savefig(f'Vpp_Size_normalized_Magnitude_colorplot.png')

    # fig1 = plt.figure(4, figsize=(10, 10))
    # plt.title(f'dX vs. Frequency (Piezo {piezo})')
    # plt.ylabel('dX')
    # plt.xlabel('Frequency')
    # # plt.hist2d(data['Frequency'], data['Magnitude'], norm=colors.PowerNorm(0.25), bins=5, cmap=plt.cm.BuGn_r)
    # data = processed_csv[['Frequency', 'dX']].to_numpy()
    # x, y = data.T
    # nbins = 30
    # k = kde.gaussian_kde(data.T)
    # xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
    # zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    # plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', norm=colors.PowerNorm(0.2), cmap='rainbow')
    # # plt.contour(xi, yi, zi.reshape(xi.shape))
    # plt.colorbar()
    # plt.tight_layout()
    #
    # fig2 = plt.figure(5, figsize=(10, 10))
    # plt.title(f'dX vs. Size (Piezo {piezo})')
    # plt.ylabel('dX')
    # plt.xlabel('Size')
    # # plt.hist2d(data['Frequency'], data['Magnitude'], norm=colors.PowerNorm(0.25), bins=5, cmap=plt.cm.BuGn_r)
    # data = processed_csv[['Size', 'dX']].to_numpy()
    # x, y = data.T
    # nbins = 30
    # k = kde.gaussian_kde(data.T)
    # xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
    # zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    # plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', norm=colors.PowerNorm(0.2), cmap='rainbow')
    # # plt.contour(xi, yi, zi.reshape(xi.shape))
    # plt.colorbar()
    # plt.tight_layout()
    #
    # fig2 = plt.figure(5, figsize=(10, 10))
    # plt.title(f'dX vs. Vpp (Piezo {piezo})')
    # plt.ylabel('dX [-]')
    # plt.xlabel('Voltage [Vpp]')
    # # plt.hist2d(data['Frequency'], data['Magnitude'], norm=colors.PowerNorm(0.25), bins=5, cmap=plt.cm.BuGn_r)
    # data = processed_csv[['Size', 'dX']].to_numpy()
    # x, y = data.T
    # nbins = 30
    # k = kde.gaussian_kde(data.T)
    # xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
    # zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    # plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', norm=colors.PowerNorm(0.2), cmap='rainbow')
    # # plt.contour(xi, yi, zi.reshape(xi.shape))
    # plt.colorbar()
    # plt.tight_layout()
    #

    import matplotlib
    from scipy import interpolate
    import scipy

    # fig5 = plt.figure(5, figsize=(10, 10))
    # plt.ylabel('X [pixels]')
    # plt.xlabel('Y [pixels]')
    x, y = processed_csv['X0'].to_numpy() ,processed_csv['Y0'].to_numpy()

    u = processed_csv['dX'].to_numpy() * processed_csv['Magnitude'].to_numpy()
    v = processed_csv['dY'].to_numpy() * processed_csv['Magnitude'].to_numpy()

    # # vect_field = np.array(((processed_csv['X0'].to_numpy(), processed_csv['Y0'].to_numpy()),
    # #                        (processed_csv['dX'].to_numpy() * processed_csv['Magnitude'].to_numpy(),
    # #                         processed_csv['dY'].to_numpy() * processed_csv['Magnitude'].to_numpy())))
    # #
    # # print(vect_field.shape)
    #
    # # for i in range(len(x)):
    # # n=0.5
    # # color = np.sqrt(((u - n) / 2) * 2 + ((v - n) / 2) * 2)
    #
    norm = matplotlib.colors.SymLogNorm(0.1)
    # # o = np.random.random(1000)
    # # occurrence = o / np.sum(o)
    # # norm.autoscale(occurrence)
    cm = matplotlib.cm.bwr
    #
    # sm = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)
    # sm.set_array([])
    #
    # # avg_vect_field = np.zeros((200, 300, 2))
    #
    xx = np.linspace(0, 299, 100)
    yy = np.linspace(0, 299, 100)
    xx, yy = np.meshgrid(xx, yy)
    #
    #
    # # 3-relation plotting: size, frequency, velocity
    # velocity = processed_csv['Magnitude'].to_numpy()
    # size = processed_csv['Size'].to_numpy()
    # frequency = processed_csv['Frequency'].to_numpy()
    #
    #
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # plt.xlabel('Size')
    # plt.ylabel('Frequency')
    # ax.scatter3D(size, frequency, velocity, c=velocity, cmap='Greens')
    # plt.show()









    points = np.transpose(np.vstack((x, y)))
    u_interp = interpolate.griddata(points, u, (xx, yy), method='cubic').flatten()
    # print(u_interp.shape)
    v_interp = interpolate.griddata(points, v, (xx, yy), method='cubic').flatten()
    magn_interp = np.sqrt(u_interp ** 2 + v_interp ** 2)
    # magn_interp = interpolate.griddata(points, , (xx, yy), method='cubic')
    # M = np.hypot(u_interp, v_interp)


    # plt.quiver(x, y, u, v, color=cm(norm(processed_csv['dX'].to_numpy())), alpha=0.5, units='x', scale=12)
    # plt.quiver(xx.flatten(), yy.flatten(), u_interp, v_interp, u_interp, cmap=matplotlib.cm.bwr, norm=norm)
    # print(q)
    # q.set_array(u_interp.flatten())
    # print(q)
    # plt.tight_layout()
    # plt.colorbar()
    plt.show()

    # fig5.savefig(f'dY_vectorfield.png')

    # for size in processed_csv['Size'].value_counts().keys():
    #
    #     temp = processed_csv[processed_csv['Size'] == size].copy()
    #     print(size, temp.shape)
    #
    #
    #     fig1 = plt.figure(4, figsize=(10, 10))
    #     plt.title(f'Frecuency vs. Velocity (Size {size})')
    #     plt.ylabel('Velocity [pixels/second]')
    #     plt.xlabel('Frequency [kHz]')
    #     # plt.hist2d(data['Frequency'], data['Magnitude'], norm=colors.PowerNorm(0.25), bins=5, cmap=plt.cm.BuGn_r)
    #     data = temp[['Frequency', 'Magnitude']].to_numpy()
    #     x, y = data.T
    #     nbins = 30
    #     k = kde.gaussian_kde(data.T)
    #     xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
    #     zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    #     plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', norm=colors.PowerNorm(0.1), cmap='rainbow')
    #     # plt.contour(xi, yi, zi.reshape(xi.shape))
    #     plt.colorbar()
    #     plt.tight_layout()
    #     plt.show()
    #
    #
    # fig2 = plt.figure(5, figsize=(10, 10))
    # plt.title(f'Vpp vs. Velocity (Piezo {piezo})')
    # plt.ylabel('Velocity [pixels/second]')
    # plt.xlabel('Voltage [Vpp]')
    # # plt.hist2d(data['Vpp'], data['vppMagnitude'], norm=colors.PowerNorm(0.5), bins=6)
    # data = processed_csv[['Vpp', 'Magnitude']].to_numpy()
    # x, y = data.T
    # nbins = 30
    # k = kde.gaussian_kde(data.T)
    # xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
    # zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    # plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', norm=colors.PowerNorm(0.2), cmap='rainbow')
    # # plt.contour(xi, yi, zi.reshape(xi.shape))
    # plt.colorbar()
    # plt.tight_layout()
    #
    # # fig3 = plt.figure(6, figsize=(10, 10))
    # # # plt.title('Action vs. Magnitude')
    # # plt.ylabel('Velocity [pixels/second]')
    # # plt.xlabel('Piezo [-]')
    # # # plt.hist2d(data['Action'], data['Magnitude'], norm=colors.PowerNorm(0.5), bins=4)
    # # data = processed_csv[['Action', 'Magnitude']].to_numpy()
    # # x, y = data.T
    # # nbins = 30
    # # k = kde.gaussian_kde(data.T)
    # # xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
    # # zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    # # plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', norm=colors.PowerNorm(0.2), cmap='rainbow')
    # # # plt.contour(xi, yi, zi.reshape(xi.shape))
    # # plt.colorbar()
    # # plt.tight_layout()
    #
    # fig4 = plt.figure(7, figsize=(10, 10))
    # plt.title(f'Size vs. Velocity (Piezo {piezo})')
    # plt.ylabel('Velocity [pixels/second]')
    # plt.xlabel('Swarm diameter [pixels]')
    # # plt.hist2d(data['Size'], data['Magnitude'], norm=colors.SymLogNorm(10), bins=20)
    # data = processed_csv[['Size', 'Magnitude']].to_numpy()
    # x, y = data.T
    # nbins = 30
    # k = kde.gaussian_kde(data.T)
    # xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
    # zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    # plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', norm=colors.PowerNorm(0.2), cmap='rainbow')
    # # plt.contour(xi, yi, zi.reshape(xi.shape))
    # plt.colorbar()
    # plt.tight_layout()
    # plt.show()

    # fig1.savefig(f'Magn_normalized_freq_magn_hist2d_piezo_{piezo}.png')
    # fig2.savefig(f'Magn_normalized_vpp_magn_hist2d_piezo_{piezo}.png')
    # # fig3.savefig('Magn_normalized_action_magn_hist2d.png')
    # fig4.savefig(f'Magn_normalized_Size_magn_hist2d_piezo_{piezo}.png')

















# # from environment_pipeline import test
# #
# # test = test()
# # test.try1(sleeper=5)
# # test.motorpos(1)
# #
# # test.motorpos(2)
#
# from settings import *
# import numpy as np
#
# def get_action(size: int, pos_x: int, pos_y: int, offset_to_target: tuple):
#     """
#     Calculate best vpp and frequency for a given size of swarm and distance to target
#     + calculate best piezo to actuate
#     :param size: Size of swarm
#     :param offset_to_target: (x, y) offset to target
#     :return: Vpp, frequency and choice of piezo
#     """
#
#     # Generate potential combinations of inputs to the model
#     steps = np.array([[Vpp, frequency, size, action, pos_x, pos_y]
#              for Vpp in np.linspace(MIN_VPP, MAX_VPP, int((MAX_VPP - MIN_VPP) / VPP_STEP_SIZE + 1))
#              for frequency in np.linspace(MIN_FREQUENCY, MAX_FREQUENCY, int((MAX_FREQUENCY - MIN_FREQUENCY) / FREQUENCY_STEP_SIZE + 1))
#              for action in range(4)])
#
#     # Map all combinations to model to get the predicted motion
#     # results = np.array(list(map(predict_state, steps)))  # TODO --> Optimize
#     results = MODEL.predict(steps)
#
#     # Find the Vpp and frequency that belong to the best inputs
#     # 'Best' is defined as the input that brings the robot closest to the target in either the x or y direction
#     direction = np.array((results[:, 0], results[:, 1]))
#     direction = direction / np.linalg.norm(direction, axis=0)
#     results = direction*results[:, 2]
#
#     # print(np.array(list(zip(steps, results.T))))
#     print(np.linalg.norm(results.T - offset_to_target, axis=1))
#
#     Vpp, frequency, size, action, pos_x, pos_y = steps[np.abs(np.linalg.norm(results.T - offset_to_target, axis=1)).argmin()]
#
#     # Choose one of four piezos
#     # IMPORTANT --> 0: dx > 0 (move left), 1: dy > 0 (move up), 2: dx < 0 (move right), 3: dy < 0 (move down)
#     # action = np.argmax(np.abs(offset_to_target))
#     # if np.sign(offset_to_target[action]) == -1:
#     #     action += 2
#
#     return int(action), Vpp, frequency
#
# # print(get_action(10, 150, 150, (-23, 12)))
#
# inputs = [f'Force_{n}' for n in range(50)]
# for inp in ["Vpp", "Frequency", "Size", "Action", "X0", "Y0"]:
#     inputs.append(inp)
# print(inputs)
#
#
