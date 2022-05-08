"""
    Description:
    Pre-processing functions for flux partitioning data.
    Authors: Weiwei Zhan, Pierre Tessier
 """
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf

data_dir = Path("../data")


def impose_noise(data_nn, SIFnoise_std=0.1, NEEnoise_std=0.08):
    """
    Add noise to NEE & SIF simulations.

    :param data_nn: dataset containing the raw data
    :type data_nn: pd.DataFrame
    :param SIFnoise_std: standard deviation of the SIF Gaussian noises (std=0.1)
    :type SIFnoise_std: float
    :param NEEnoise_std: heteroscedastic noise that scales for NEE magnitude (8% of the NEE magnitude)
    :type NEEnoise_std: float
    :return: dataset with added noise, and dataset with daytime observations only
    :rtype: (pd.DataFrame, pd.DataFrame)
    """

    std_SIF = SIFnoise_std
    std_NEE = NEEnoise_std * data_nn.NEE_canopy.abs()

    data_night = data_nn.loc[data_nn.APAR_label == 0, :].copy()
    data_day = data_nn.loc[data_nn.APAR_label == 1, :].copy()

    np.random.seed(42)
    noise_SIF = np.random.normal(0, std_SIF, data_day.shape[0])
    noise_NEE = np.random.normal(0, std_NEE)

    # add SIF noise
    data_day['SIF_obs'] = data_day['SIFcanopy_760nm'] + noise_SIF
    data_night['SIF_obs'] = data_night['SIFcanopy_760nm']
    data_nn = pd.concat([data_day, data_night]).sort_index()

    # add NEE noise
    data_nn.loc[:, 'NEE_obs'] = data_nn['NEE_canopy'] + noise_NEE

    return data_nn, data_day


def standard_x(x_train, x_test=None):
    """
    Data normalization (centering and variance normalization).

    :param x_train: training data
    :type x_train: np.ndarray
    :param x_test: testing data
    :type x_test: np.ndarray | None
    :return: normalized data
    :rtype: np.ndarray | (np.ndarray, np.ndarray)
    """
    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    x_train1 = ((x_train - x_mean) / x_std).values
    if x_test is not None:
        x_test1 = ((x_test - x_mean) / x_std).values
        return x_train1, x_test1
    return x_train1


def load_dataset(filename, data_dir=data_dir):
    """
    Load a SCOPE simulation dataset and performs all the pre-processing steps.

    It includes: data normalization, train/test/val split, inputs creation, transfer to default device.

    :param filename: name of the file to use
    :type filename: str
    :param data_dir: path of the data directory
    :type data_dir: str | Path
    :return: split dataset and all required tensors
    :rtype: tuple[pd.DataFrame | tf.Tensor]
    """
    data_nn = pd.read_csv(Path(data_dir) / filename, index_col=0)

    data_nn.index = pd.to_datetime(data_nn.index)
    data_nn['Date'] = data_nn.index.date
    data_nn['Time'] = data_nn.index.time
    data_nn = data_nn.sort_index()

    # add noise to NEE & SIF simulations
    data_nn, _ = impose_noise(data_nn)

    train, val, test = data_nn.iloc[:3945, :].copy(), data_nn.iloc[3945:5260, :].copy(), data_nn.iloc[5260:, :].copy()
    train['train_label'] = 'Training set'
    test['train_label'] = 'Test set'
    val['train_label'] = "Validation set"

    # split into train & test datasets
    var_NEE = 'NEE_obs'

    EV1_label = ['Tair', 'RH', 'PAR', 'SWC', 'u', 'LAI', 'APAR_canopy']
    EV2_label = ['Tair', 'SWC', 'u', 'LAI']

    EV1_train = train[EV1_label].astype('float32')  # EV for GPP
    EV2_train = train[EV2_label].astype('float32')  # EV for Reco
    NEE_train = train[var_NEE].astype('float32')
    APAR_max = train["APAR_canopy"].values.max()
    # label_train = train['APAR_canopy'].values / APAR_max
    # label_train = train['APAR_label'].values
    label_train = train['APAR_canopy'] > 0

    EV1_test = test[EV1_label].astype('float32')  # EV for GPP
    EV2_test = test[EV2_label].astype('float32')  # EV for Reco
    NEE_test = test[var_NEE].astype('float32')
    # label_test = test['APAR_canopy'].values / APAR_max
    # label_test = test['APAR_label'].values
    label_test = test['APAR_canopy'] > 0

    EV1_val = val[EV1_label].astype('float32')  # EV for GPP
    EV2_val = val[EV2_label].astype('float32')  # EV for Reco
    NEE_val = val[var_NEE].astype('float32')
    # label_val = val['APAR_canopy'].values / APAR_max
    # label_val = val['APAR_label'].values
    label_val = val["APAR_canopy"] > 0

    EV1_train1, EV1_test1 = standard_x(EV1_train, EV1_test)
    EV2_train1, EV2_test1 = standard_x(EV2_train, EV2_test)
    _, EV1_val1 = standard_x(EV1_train, EV1_val)
    _, EV2_val1 = standard_x(EV2_train, EV2_val)

    # Y_data Normalization
    NEE_max_abs = (np.abs(NEE_train.values)).max()
    NEE_train1 = NEE_train.values / NEE_max_abs
    NEE_test1 = NEE_test.values / NEE_max_abs
    NEE_val1 = NEE_val.values / NEE_max_abs

    out = [
        EV1_train1, EV2_train1, NEE_train1, label_train,
        EV1_test1, EV2_test1, NEE_test1, label_test,
        EV1_val1, EV2_val1, NEE_val1, label_val,
        NEE_max_abs
    ]
    for i, x in enumerate(out):
        out[i] = tf.convert_to_tensor(x, dtype=tf.float32)
    return train, test, val, *out
