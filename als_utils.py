# -*- coding: utf-8 -*-
"""
Utilities for TAU ALS ML project

Created on Thu Sep    8 23:57:12 2016

@author: yoel
"""
from __future__ import print_function

import IPython as _IPython
import logging as _logging
import collections as _clct

import numpy as _np
import pandas as _pd
#import openpyxl as _xl
from six.moves import cPickle as _pickle

import matplotlib as _mpl
import matplotlib.pylab as _plt
import scipy.cluster.hierarchy as _sch

import services as _srv

_srv.stream_log()
_logger = _logging.getLogger('.'.join((_srv._package_logger, __name__)))
_logger.debug('generated module logger, and tied to package logger')

_ip = _IPython.get_ipython()

try:
    _ip.Completer.limit_to__all__
except:
    _logger.debug('failed to limit ipython autocomplete')


# %% Utils

def find_group(df, key, case_sensitive=True):
    """returns list of df (DataFrame) column numbers with key in name"""

    cols = []
    if case_sensitive:
        for k, c in enumerate(df.columns.values):
            if key in c:
                cols.append(k)

    else:
        key = key.lower()
        for k, c in enumerate(df.columns.values):
            if key in c.lower():
                cols.append(k)

    return cols


def rename_column(col_name):
    """Standardize manual column\feature names"""
    words = col_name.split()
    if 'nuc' == words[0].lower()[:3]:
        words[0] = 'NUC'
    elif 'mito' == words[0].lower()[:4]:
        words[0] = 'MITO'
    return '_'.join(words)



# %%

def reformat_as_dataset(data_df, labels_dict):
    """convert DataFrame to classifier ready arrays"""

    data = data_df.iloc[:, _col0:].values

    zscore_params = {}
    zscore_params['mean'] = data_df.iloc[:, _col0:].mean().values
    zscore_params['std'] = data_df.iloc[:, _col0:].std().values

    for col in range(data.shape[1]):
        data[:, col] -= zscore_params['mean'][col]
        data[:, col] /= zscore_params['std'][col]

    labels_set = set(val for val in labels_dict.itervalues())
    n_labels = len(labels_set)
    label_encoding = _clct.OrderedDict()
    while len(labels_set) > 0:
        code = str(len(label_encoding))
        label_encoding[labels_set.pop()] = code

    labels = _np.zeros((data.shape[0], n_labels))
    for well in labels_dict.iterkeys():
        mask = data_df['Well'].values.flatten() == well
        col = int(label_encoding[labels_dict[well]])
        labels[mask, col] = 1

    label_decoding = {}
    for tp in label_encoding.iteritems():
        label_decoding[tp[1]] = tp[0]

    dataset = {}
    dataset['data'] = data
    dataset['labels'] = labels
    dataset['zscore_params'] = zscore_params
    dataset['label_decoding'] = label_decoding
    dataset['features'] = data_df.columns.values[_col0:]

    _logger.info('finished reformating data to ML ready format')

    return dataset


def reformat_as_raw_dataset(data_df, labels_dict):
    """convert DataFrame to classifier ready arrays"""

    data = data_df.iloc[:, _col0:].values

    labels_set = set(val for val in labels_dict.itervalues())
    n_labels = len(labels_set)
    label_encoding = _clct.OrderedDict()
    while len(labels_set) > 0:
        code = str(len(label_encoding))
        label_encoding[labels_set.pop()] = code

    labels = _np.zeros((data.shape[0], n_labels))
    for well in labels_dict.iterkeys():
        mask = data_df['Well'].values.flatten() == well
        col = int(label_encoding[labels_dict[well]])
        labels[mask, col] = 1

    label_decoding = {}
    for tp in label_encoding.iteritems():
        label_decoding[tp[1]] = tp[0]

    dataset = {}
    dataset['data'] = data
    dataset['labels'] = labels
    dataset['label_decoding'] = label_decoding
    dataset['features'] = data_df.columns.values[_col0:]

    _logger.info('finished reformating data to ML ready format')

    return dataset


def load_dataset(filename):
    """load pickled dataset"""

    with open(filename, 'rb') as f:

        dataset = _pickle.load(f)

        print('variables in output:')
        for key in dataset:
            print(key)

    _logger.info('finished loading pickled dataset')

    return dataset


def save_dataset(dataset, filename):

    with open(filename, 'wb') as f:

        _pickle.dump(dataset, f, _pickle.HIGHEST_PROTOCOL)

    _logger.info('finished saving pickled dataset')


def heatmap(X, labels, feature_names=None, observation_names=None,
            cluster_observations=True, cluster_features=True,
            observations_method='ward', features_method='ward',
            main_title=None, legend=None, legend_cmap='Accent'):
    """Generate heatmap

    Input
    -----

    X: ndarray, NxM
        rows = observations, columns = features

    labels: ndarray, N

    feature_names: string list, M (optional)

    observatoin_names: string list, N (optional)

    cluster_X: boolean
        default True

    X_method: string
        default 'ward', for other options see scipy.cluster.hierarchy.linkage

    Returns
    -------

    figure handle

    """
    lower = _np.percentile(X, 1)
    upper = _np.percentile(X, 99)
    X[X < lower] = lower
    X[X > upper] = upper

    if cluster_features:
        d = _sch.distance.pdist(X.T, 'correlation')
        L = _sch.linkage(d, method=features_method)
        ind = _sch.fcluster(L, 0.5*d.max(), 'distance')
        cols = _np.argsort(ind)
        X = X[:, cols]
    else:
        cols = _np.arange(X.shape[1])

    if cluster_observations:
        d = _sch.distance.pdist(X, 'correlation')
        L = _sch.linkage(d, method=observations_method)
        ind = _sch.fcluster(L, 0.5*d.max(), 'distance')
        rows = _np.argsort(ind)
        X = X[rows, :]
    else:
        rows = _np.arange(X.shape[0])

    _mpl.rcParams.update(_mpl.rcParamsDefault)
    _plt.style.use('fivethirtyeight')

    fig_hm = _plt.figure()
    _plt.imshow(
        X, aspect='auto', interpolation='none', cmap='seismic')
    cb = _plt.colorbar()
    ax_hm = _plt.gca()
    ax_hm.grid('off')
    if main_title is not None:
        ax_hm.set_title(main_title)

    if feature_names is None:
        heatmap_rect = [0.1, 0.1, 0.7, 0.85]
        ax_hm.axes.xaxis.set_visible(False)
    else:
        heatmap_rect = [0.1, 0.25, 0.7, 0.7]
        ax_hm.set_xticks(_np.arange(len(cols)))
        ax_hm.set_xticklabels(feature_names[cols], rotation='vertical')

    if observation_names is None:
        label_offset = 0
        ax_hm.axes.yaxis.set_visible(False)
    else:
        label_offset = -0.02
        ax_hm.set_yticks(_np.arange(len(rows)))
        ax_hm.set_yticklabels(observation_names[rows])

    labels_rect = [0.05 + label_offset, heatmap_rect[1], 0.02, heatmap_rect[3]]
    colorbar_rect = [0.85, heatmap_rect[1], 0.05, heatmap_rect[3]]
    ax_hm.set_position(heatmap_rect)
    cb.ax.set_position(colorbar_rect)
    ax_hm_labels = fig_hm.add_axes(labels_rect)

    if labels.ndim == 1:
        labels = labels[rows, _np.newaxis]
    ax_hm_labels.imshow(
        labels, aspect='auto', interpolation='none', cmap=legend_cmap)
    ax_hm_labels.grid('off')
    ax_hm_labels.axes.xaxis.set_visible(False)
    ax_hm_labels.axes.yaxis.set_visible(False)
    if legend is not None:
        ax_hm_labels.set_title(legend, fontsize=14)

    _plt.show()

    return fig_hm
