import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import warnings
import atexit
from matplotlib import style
from matplotlib import colors
from scipy.stats import kde
import matplotlib
from scipy import interpolate
atexit.register(cv2.destroyAllWindows)
warnings.simplefilter("ignore")
style.use('seaborn-paper')

print('Loading data...')
PRED_SECONDS = 1.0
DYNAMICS_CSV = pd.read_csv(f"E:\\square_env_recordings_metadata_training\\dynamics_{PRED_SECONDS}s.csv")
del DYNAMICS_CSV["Unnamed: 0"]
DYNAMICS_CSV = DYNAMICS_CSV[DYNAMICS_CSV["Magnitude"] <= 20] # Threshold filter magnitude


def normalized_sample(csv, feature, sampling=10, repeats=10):
    """
    Uniformly distirbute a feature from csv
    :param csv:         Working csv
    :param feature:     Feature to redistribute
    :param sampling:    Number of samples per batch
    :param repeats:     Number of batches
    :return:            Uniformly distributed csv in the dimension of feature
    """
    bins = csv[feature].value_counts().keys()
    new_csv = pd.DataFrame()
    for i in range(repeats):
        for n in bins:
            new_csv = new_csv.append(csv[csv[feature] == n].sample(n=sampling, replace=True))
    return new_csv.copy()

def create_vector_fields(csv0):
    """
    Create vector fields of the dynamics of environment per piezo
    :param csv0: Working csv
    :return:     Vector fields
    """
    size = 300

    vect_fields = np.empty((0, size, size, 2))

    for piezo in range(4):
        csv = csv0.copy()

        # Get data from piezo
        if type(piezo) == int:
            print(f"Selecting piezo: {piezo}")
            csv = csv[csv["Action"] == piezo]

        # Normalize dX and dY
        csv = normalized_sample(normalized_sample(csv, "dX"), "dY")

        # Get initial points and movements
        x, y = csv['X0'].to_numpy(dtype=int), csv['Y0'].to_numpy(dtype=int)
        u = np.array( np.round(csv['dX'].to_numpy() * csv['Magnitude'].to_numpy(), 0), dtype=int)
        v = np.array( np.round(csv['dY'].to_numpy() * csv['Magnitude'].to_numpy(), 0), dtype=int)

        # Initialize grid
        xx = np.linspace(0, 299, size)
        yy = np.linspace(0, 299, size)
        xx, yy = np.meshgrid(xx, yy)

        # Interpolate to grid
        points = np.transpose(np.vstack((x, y)))
        u_interp = interpolate.griddata(points, u, (xx, yy), method='linear', fill_value=1e-6)
        v_interp = interpolate.griddata(points, v, (xx, yy), method='linear', fill_value=1e-6)
        magn_interp = np.sqrt(u_interp ** 2 + v_interp ** 2)

        vect_fields = np.vstack( (vect_fields, np.stack((u_interp, v_interp), axis=2)[np.newaxis, :, :, :]))

    return vect_fields


def plot_velocity_field(csv, piezo=None):
    """
    Make quiver plot of vector fields
    :param csv:     Working csv
    :param piezo:   Piezo inputs to use
    :return:        Quiver plot
    """
    # Set figure properties
    fig = plt.figure(1, figsize=(10, 10))
    plt.ylabel('X [pixels]', fontsize=20)
    plt.xlabel('Y [pixels]', fontsize=20)
    plt.tick_params(axis="both", length=6, width=3, labelsize=13)

    # Get data from piezo
    if type(piezo) == int:
        print(f"Selecting piezo: {piezo}")
        csv = csv[csv["Action"] == piezo]

    # Normalize dX and dY
    csv = normalized_sample(normalized_sample(csv, "dX"), "dY")

    # Get initial points and movements
    x, y = csv['X0'].to_numpy(), csv['Y0'].to_numpy()
    u = csv['dX'].to_numpy() * csv['Magnitude'].to_numpy()
    v = csv['dY'].to_numpy() * csv['Magnitude'].to_numpy()

    # Initialize grid
    xx = np.linspace(0, 299, 300)
    yy = np.linspace(0, 299, 300)
    xx, yy = np.meshgrid(xx, yy)

    # Interpolate to grid
    points = np.transpose(np.vstack((x, y)))
    u_interp = interpolate.griddata(points, u, (xx, yy), method='linear').flatten()
    v_interp = interpolate.griddata(points, v, (xx, yy), method='linear').flatten()
    magn_interp = np.sqrt(u_interp ** 2 + v_interp ** 2)

    # Define vector colors
    vect_colors = {0: u_interp, 1: v_interp, 2: u_interp, 3: v_interp, None: magn_interp}

    test = (np.arctan2(*np.array((u_interp, v_interp))) - 0.5*np.pi) % (2*np.pi)# - np.pi

    # Plot quiver plot
    plt.quiver(xx.flatten(), yy.flatten(), u_interp, v_interp,
               test,
               cmap=matplotlib.cm.hsv
               )
    plt.tight_layout()
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=15)
    cbar.set_ticks(np.linspace(0, 2, 5)*np.pi)
    cbar.set_ticklabels([f"{n}\u03C0" for n in np.linspace(0, 2, 5)])

    # plt.savefig(f"data_analysis_figures\\vectorfield_piezo{piezo}.png")
    # plt.savefig(f"data_analysis_figures\\vectorfield_piezo{piezo}.pdf")

    plt.show()

    return fig

def colorplot_2d(csv, feature_x, feature_y, normalize_x, normalize_y, sampling=10, repeats=10, piezo=None):
    """
    Make 2d colormap of realationship between two features
    :param csv:         Working csv
    :param feature_x:   Feature X
    :param feature_y:   Feature Y
    :param normalize_x: True or False
    :param normalize_y: True or False
    :param sampling:    For feature redistribution
    :param repeats:     For feature redistribution
    :param piezo:       Piezo to select
    :return:            2d colormap
    """

    # Set figure properties
    fig = plt.figure(1, figsize=(10, 10))
    plt.ylabel(feature_y, fontsize=20)
    plt.xlabel(feature_x, fontsize=20)
    plt.tick_params(axis="both", length=6, width=3, labelsize=13)

    # Get data from piezo
    if type(piezo) == int:
        print(f"Selecting piezo: {piezo}")
        csv = csv[csv["Action"] == piezo]

    # Get data
    if normalize_x:
        csv = normalized_sample(csv=csv, feature=feature_x, sampling=sampling, repeats=repeats)
    if normalize_y:
        csv = normalized_sample(csv=csv, feature=feature_y, sampling=sampling, repeats=repeats)
    print(csv.shape)
    x = csv[feature_x].to_numpy() #* (csv['Magnitude'].to_numpy() / max(csv["Magnitude"]))
    y = csv[feature_y].to_numpy() #* (csv[feature_y].to_numpy() / max(csv[feature_y]))

    # Create colormesh
    nbins = 20
    k = kde.gaussian_kde(np.array((x, y)), weights=csv["Magnitude"].to_numpy() / max(csv["Magnitude"]))
    xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape),
                   shading='gouraud',
                   norm=colors.PowerNorm(0.5),
                   cmap='rainbow')
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=15)
    plt.tight_layout()

    # fig.savefig(f"data_analysis_figures\\{feature_x}_vs_{feature_y}_piezo{piezo}.png")
    # fig.savefig(f"data_analysis_figures\\{feature_x}_vs_{feature_y}_piezo{piezo}.pdf")

    plt.show()
    return fig

if __name__ == '__main__':
    # plot_velocity_field(csv=DYNAMICS_CSV, piezo=1)
    # feature_x = "Vpp"
    # feature_y = "Magnitude"
    # fig = colorplot_2d(csv=DYNAMICS_CSV,
    #              feature_x=feature_x,
    #              feature_y=feature_y,
    #              normalize_x=True,
    #              normalize_y=False,
    #              sampling=1000,
    #              repeats=10,
    #              piezo=None)
    # np.save('C:\\Users\\ARSL\\PycharmProjects\\Project_Matt\\venv\\Include\\AI_Actuated_Micrswarm_4\\models\\Vector_fields.npy', create_vector_fields(csv0=DYNAMICS_CSV))
    pass
