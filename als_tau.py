# -*- coding: utf-8 -*-
"""
ALS Project

Created on Mon Sep    5 00:15:16 2016

@author: YoelS

notes:
need a "well" object, with (access) to its raw data
needs to be able to return:
- per feature summaries (count, mean, std, ? histogram)
- needs a method to act on proccesed features:
  either hold them, or get pipeline as input
would be nice to pass functions instead of coding them internally

sophisticated pipeline approach:
each function inherits a base class
b.c. should declare input + output types (dimensions?)
to allow fast compatability check
b.c. should act as wrapper, and test in\output compliance on the fly
b.c. log messages need to include private function name that triggered it
"""

from __future__ import print_function

import os, sys
import collections

#import IPython

import numpy as np
import pandas as pd
from six.moves import cPickle as pickle

from scipy import stats
from sklearn import svm
from sklearn import tree
from sklearn import cluster
from sklearn import metrics
from sklearn import manifold
from sklearn import neighbors
from sklearn import decomposition
from sklearn import cross_validation
from sklearn.externals import joblib

import matplotlib as mpl
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

import als_utils as als

#ip = IPython.get_ipython()
#ip.run_line_magic('matplotlib', 'qt')
#ip.enable_pylab()

mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use('fivethirtyeight')
plt.close('all')

enable_save = False

fig_enable = {
    'cells' : False,
    'cells clustered' : False,
    'PCA' : False,
    'wells 1D' : False,
    'wells 1D clustered' : False,
    'wells ND' : True,
    'wells ND clustered' : True}

sys.path = [
    '',
    '/usr/lib/python2.7',
    '/usr/lib/python2.7/plat-x86_64-linux-gnu',
    '/usr/lib/python2.7/lib-tk',
    '/usr/lib/python2.7/lib-old',
    '/usr/lib/python2.7/lib-dynload',
    '/usr/local/lib/python2.7/dist-packages',
    '/usr/local/lib/python2.7/dist-packages/six-1.10.0-py2.7.egg',
    '/usr/lib/python2.7/dist-packages',
    '/usr/lib/python2.7/dist-packages/PILcompat',
    '/usr/lib/python2.7/dist-packages/gtk-2.0',
    '/usr/lib/python2.7/dist-packages/ubuntu-sso-client',
    '/usr/local/lib/python2.7/dist-packages/IPython/extensions',
    '/home/yoel/.ipython',
    '/home/yoel/python',
    '/home/yoel',
    '/home/yoel/anaconda2/lib/python27.zip',
    '/home/yoel/anaconda2/lib/python2.7',
    '/home/yoel/anaconda2/lib/python2.7/plat-linux2',
    '/home/yoel/anaconda2/lib/python2.7/lib-tk',
    '/home/yoel/anaconda2/lib/python2.7/lib-old',
    '/home/yoel/anaconda2/lib/python2.7/lib-dynload',
    '/home/yoel/anaconda2/lib/python2.7/site-packages',
    '/home/yoel/anaconda2/lib/python2.7/site-packages/Sphinx-1.4.1-py2.7.egg',
    '/home/yoel/anaconda2/lib/python2.7/site-packages/setuptools-23.0.0-py2.7.egg']


# %% Excel Parsing

#workbook organiztion:
#plate
#- wells, B2 - G11 (e.g.)
#-- blocks\fields (fld), 1-9
#--- single cells
#
#labeling per well:
#HC (control?) vs. ALS + subject id (e.g. SA0955\4)
#
#features are divided into channels:
#1) NUCLEAR
#2) CELL
#3) MITOTRACKER
#4) TMRE


#FOLDER = r'C:\Users\YoelS\Desktop\Tau'
FOLDER = '/home/yoel/Desktop/Data/ALS_TAU/try3/'
FILENAME_RAW = 'MITO 2 PLATE 3D_1.2016.08.01.18.08.06.XLS'
cell_classifier_filename = os.path.join(FOLDER, 'cell_classifier.pickle')

# applied some manual file manipulation
single_cell_raw_data = pd.read_excel(
    FILENAME_RAW, sheetname='Nuc_cell_TMRE_MITOTRACKER')

first_data_column = 3

n_cells = single_cell_raw_data.shape[0]

feature_names = single_cell_raw_data.columns.values[3:].copy()
for k, s in enumerate(feature_names):
    feature_names[k] = als.rename_column(s)

n_features = len(feature_names)

patient_id = np.unique(single_cell_raw_data.ID)
n_patients = len(patient_id)

well_id = np.unique(single_cell_raw_data.Well)
n_wells = len(well_id)

well_labels = np.zeros((n_wells, ))
for k, val in enumerate(well_id):
    mask = single_cell_raw_data.Well == val
    l = single_cell_raw_data.Label[mask].values[0]
    if l == 'ALS':
        well_labels[k] = 1


# %% Single-Cell Heatmap

data = single_cell_raw_data.iloc[: , first_data_column:].values
for col in range(data.shape[1]):
    data[:, col] -= np.mean(data[:, col])
    data[:, col] /= np.std(data[:, col])

labels_binary = np.zeros((n_cells, 1))
labels_binary[single_cell_raw_data.Label.values == 'ALS'] = 1

labels_id = np.zeros((n_cells, 1))
for col, val in enumerate(patient_id):

    if col < 5:
        trick = 0
    else:
        trick = 4

    labels_id[single_cell_raw_data.ID.values == val] = col + trick

ids = single_cell_raw_data.ID.values
index = np.argsort(ids)

data = data[index, :]
labels_binary = labels_binary[index]
labels_id = labels_id[index, :]

if fig_enable['cells']:

    fig_hm_unclustered = als.heatmap(
        data, labels_id, feature_names=feature_names,
        cluster_observations=False,
        main_title='Feature Z-Scores',
        legend='5 HC, 5 ALS', legend_cmap='PiYG')

    if enable_save:
        with open(os.path.join(FOLDER, 'heatmap_unclustered.pig'), 'wb') as f:
            pickle.dump(fig_hm_unclustered, f, pickle.HIGHEST_PROTOCOL)

if fig_enable['cells clustered']:

    fig_hm = als.heatmap(
        data, labels_binary, feature_names=feature_names,
        main_title='Feature Z-Scores - Clustered by Rows (i.e. Cells)',
        legend='pink=HC, green=ALS', legend_cmap='PiYG')

    if enable_save:
        with open(os.path.join(FOLDER, 'heatmap_clustered.pig'), 'wb') as f:
            pickle.dump(fig_hm, f, pickle.HIGHEST_PROTOCOL)


# %% PCA

data = single_cell_raw_data.iloc[: , first_data_column:].values
for col in range(data.shape[1]):
    data[:, col] -= np.mean(data[:, col])
    data[:, col] /= np.std(data[:, col])

labels_binary = np.zeros((n_cells, 1))
labels_binary[single_cell_raw_data.Label.values == 'ALS'] = 1

pca = decomposition.PCA()
X_pca = pca.fit_transform(data)

mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use('fivethirtyeight') # 'ggplot')

if fig_enable['PCA']:

    fig_pca, ax_pca = plt.subplots(num='PCA Variance')
    ax_pca.plot(100 * np.cumsum(pca.explained_variance_ratio_))
    ax_pca.grid('on')
    ax_pca.set_title('PCA Explained Variance Ratio')
    ax_pca.set_xlabel('#Component')
    ax_pca.set_ylabel('% Cumulative Variance')
    ax_pca.set_ylim([-1, 105])
    ax_pca.set_xlim([-1, 40])

plt.show()


# %% Single-Cell Classification

data = single_cell_raw_data.iloc[: , first_data_column:].values
for col in range(data.shape[1]):
    data[:, col] -= np.mean(data[:, col])
    data[:, col] /= np.std(data[:, col])

labels_binary = np.zeros((n_cells, 1))
labels_binary[single_cell_raw_data.Label.values == 'ALS'] = 1

n_token = 20

binary_labels = ['HC', 'ALS']

index = np.argsort(np.random.rand(n_cells))
i_train = int(np.floor(0.6 * n_cells))
i_valid = int(np.floor(0.8 * n_cells))

# only use at end
test_data = data[index[i_valid:], :]
test_labels_binary = labels_binary[index[i_valid:]]

print('Single Cell Classifier')
print('=' * n_token + ' Test with SVM ' + '=' * n_token)

cell_classifier = svm.LinearSVC(class_weight='balanced')
cell_classifier.fit(
    data[index[:i_valid], :], labels_binary[index[:i_valid]].flatten())

test_tags = cell_classifier.predict(test_data)

print('-' * n_token + ' Binary ' + '-' * n_token)
print('Test Scores')
print(metrics.classification_report(test_labels_binary, test_tags))

print('Test Confusion Table')
print(binary_labels)
print(metrics.confusion_matrix(test_labels_binary, test_tags))


# %% Classify Wells 1D

data = single_cell_raw_data.iloc[: , first_data_column:].values
for col in range(data.shape[1]):
    data[:, col] -= np.mean(data[:, col])
    data[:, col] /= np.std(data[:, col])

well_descriptors = pd.DataFrame(columns=well_id)
for w_id in well_id:

    mask = single_cell_raw_data.Well.values == w_id

    # zscores, unlike raw data
    data_k = data[mask, :]

    scores = cell_classifier.decision_function(data_k)

    counts, _ = np.histogram(scores, bins=np.arange(-2, 2, 0.1))
    counts = np.float64(counts)
    counts /= np.sum(counts)

    well_descriptors.loc[:, w_id] = counts

desc_data = well_descriptors.values.T

#fig_desc, ax_desc = plt.subplots(num='Well Descriptors')
#ax_desc.imshow(desc_data, aspect='auto', interpolation='none')
#ax_desc.grid('off')
#ax_desc.set_title('Well Descriptors')
#ax_desc.set_yticks(np.arange(n_wells))
#ax_desc.set_yticklabels(well_id)
#ax_desc.set_xlabel('Cell Classifier Scores')

if fig_enable['wells 1D']:

    fig_1D_desc = als.heatmap(
        desc_data, well_labels,
        cluster_observations=True, cluster_features=False,
        observation_names=well_descriptors.columns.values,
        main_title='Cell Classification Score Histograms - per Well',
        legend='pink=HC\ngreen=ALS', legend_cmap='PiYG')

if fig_enable['wells 1D clustered']:

    fig_1D_desc_clust = als.heatmap(
        desc_data, well_labels,
        cluster_observations=False, cluster_features=False,
        observation_names=well_descriptors.columns.values,
        main_title='Cell Classification Score Histograms - per Well',
        legend='pink=HC\ngreen=ALS', legend_cmap='PiYG')

plt.show()


# %% Plate Class (TBD)

plate = {}
plate['well_meta'] = pd.DataFrame()
plate['well_data'] = {}
data = single_cell_raw_data.iloc[: , first_data_column:].values
for w_id in well_id:

    mask = single_cell_raw_data.Well.values == w_id
    data_k = data[mask, :]
    plate['well_data'][w_id] = {}
    plate['well_data'][w_id]['data'] = data_k
    plate['well_data'][w_id]['mean'] = np.mean(data_k, axis=0)
    plate['well_data'][w_id]['std'] = np.std(data_k, axis=0)

    index_1 = np.argmax(single_cell_raw_data.Well.values == w_id)
    plate['well_meta'].loc[w_id, 'patient_id'] = single_cell_raw_data.loc[
        index_1, 'ID']
    plate['well_meta'].loc[w_id, 'label'] = single_cell_raw_data.loc[
        index_1, 'Label']
    plate['well_meta'].loc[w_id, 'n_cells'] = data_k.shape[0]


def get_labels(palte, keys=None):
    """Return well labels, according to keys"""

    if keys is None:
        keys = plate['well_data'].keys()

    labels = []
    for w_id in keys:
        labels.append(plate['well_meta'].loc[w_id, 'label'])

    return labels


def cross_validation_keys(plate, n_cross):
    """Well keys for cross-validation train and test, shuffled and balanceds"""

    all_keys = plate['well_meta'].index.values

    cross_valid_keys = {}
    for k_cross in range(n_cross):
        key = str(k_cross)
        cross_valid_keys[key] = {}
        cross_valid_keys[key]['valid'] = np.array([])
        cross_valid_keys[key]['train'] = []

    # balance classes
    classes = np.unique(plate['well_meta'].label)
    for k_class in classes:

        class_keys = plate['well_meta'].index[
            plate['well_meta'].label == k_class].values

        index1 = np.argsort(np.random.rand(len(class_keys)))
        index2 = np.int32(
            np.linspace(0, len(class_keys), n_cross + 1))

        for k_cross in range(n_cross):
            key = str(k_cross)
            cross_valid_keys[key]['valid'] = np.append(
                cross_valid_keys[key]['valid'],
                class_keys[index1[index2[k_cross]:index2[k_cross + 1]]])

    # remove validation keys from train keys
    for k_cross in range(n_cross):
        key1 = str(k_cross)
        for key2 in all_keys:
            if key2 not in cross_valid_keys[key1]['valid']:
                cross_valid_keys[key1]['train'].append(key2)
        cross_valid_keys[key1]['train'] = np.array(
            cross_valid_keys[key1]['train'])

    # shuffle
    for k_cross in range(n_cross):
        key1 = str(k_cross)
        for key2 in cross_valid_keys[key1].keys():
            index = np.argsort(np.random.rand(
                len(cross_valid_keys[key1][key2])))
            cross_valid_keys[key1][key2] = cross_valid_keys[key1][key2][index]

    return cross_valid_keys


def get_z_score_params(plate, keys=None):
    """Calculate Z-Score params from wells defined by keys"""

    n_cells = []
    means = []
    stds = []

    if keys is None:
        keys = plate['well_data'].keys()

    for w_id in keys:
        n_cells.append(plate['well_meta'].loc[w_id, 'n_cells'])
        means.append(plate['well_data'][w_id]['mean'])
        stds.append(plate['well_data'][w_id]['std'])

    n_total_cells = np.float(sum(n_cells))
    w_cells = np.array(n_cells) / n_total_cells
    stds = np.array(stds)
    means = np.array(means)

    n_well = len(n_cells)
    n_features = means.shape[1]

    pooled_means = np.zeros((1, n_features))
    for k_w in range(n_well):
        pooled_means += (means[k_w, :] * w_cells[k_w])

    pooled_stds = np.zeros((1, n_features))
    for k_w in range(n_well):
        pooled_stds += (w_cells[k_w] * (
            (stds[k_w, :] ** 2) +
            ((means[k_w, :] - pooled_means) ** 2)))
    pooled_stds = np.sqrt(pooled_stds)

    z_score_params = {}
    z_score_params['means'] = pooled_means.flatten()
    z_score_params['stds'] = pooled_stds.flatten()

    return z_score_params


def calcualte_descriptors(plate, z_score_params, keys=None, n_bins=30):
    """Return descriptors for all wells defined in keys (default = all wells)
    by normalize data with z_score_params
    and concatenting the normalized-feature histograms"""

    if keys is None:
        keys = plate['well_data'].keys()

    bins = np.linspace(-3, 3, n_bins + 1)
    n_features = plate['well_data'][keys[0]]['data'].shape[1]

    smoothing_filter = np.array([1.0, 2.0, 4.0, 2.0, 1.0])
    smoothing_filter /= np.sum(smoothing_filter)

    well_descriptors = []
    for w_id in keys:

        temp = np.array([])
        for col in range(n_features):

            z_scores = (plate['well_data'][w_id]['data'][:, col] -
                z_score_params['means'][col]) / z_score_params['stds'][col]

            counts, _ = np.histogram(z_scores, bins=bins)

            # histogram smoothing
            counts = np.convolve(
                np.float64(counts), smoothing_filter, mode='same')

            counts /= np.sum(counts)

            temp = np.hstack((temp, counts))

        well_descriptors.append(temp)

    return np.array(well_descriptors)


# %% Visualize N-Histogram Descriptors

all_keys = plate['well_meta'].index.values

zs_params = get_z_score_params(plate, keys=all_keys)

X = calcualte_descriptors(plate, zs_params, keys=all_keys)

labels = np.zeros(len(all_keys))
labels[plate['well_meta'].label.values == 'ALS'] = 1

#index = np.hstack((np.arange(0, 60, 2), np.arange(1, 60, 2)))
#Xo = X[index, :]
#lo = labels[index]
#ono = all_keys[index]
#fig_ordered = als.heatmap(
#    Xo, lo, observation_names=ono,
#    cluster_features=False, cluster_observations=False,
#    main_title='Multiple Histogram Descriptors Ordered Manually',
#    legend='pink=HC\ngreen=ALS', legend_cmap='PiYG')

if fig_enable['wells ND']:

    fig_ND_desc = als.heatmap(
        X, labels, observation_names=all_keys,
        cluster_features=False, cluster_observations=False,
        main_title='Multiple Histogram Descriptors',
        legend='pink=HC\ngreen=ALS', legend_cmap='PiYG')

    if enable_save:
        with open(os.path.join(FOLDER, 'heatmap_ND.pig'), 'wb') as f:
            pickle.dump(fig_hm_unclustered, f, pickle.HIGHEST_PROTOCOL)

if fig_enable['wells ND clustered']:

    fig_ND_desc = als.heatmap(
        X, labels, observation_names=all_keys,
        cluster_features=False, cluster_observations=True,
        observations_method='ward',
        main_title='Multiple Histogram Descriptors',
        legend='pink=HC\ngreen=ALS', legend_cmap='PiYG')

    if enable_save:
        with open(os.path.join(FOLDER, 'heatmap_ND_clustered.pig'), 'wb') as f:
            pickle.dump(fig_hm_unclustered, f, pickle.HIGHEST_PROTOCOL)

plt.show()

# %% Statistical Significance Test (Permutations)

n_cross_valid = 3

well_classifier = svm.LinearSVC(class_weight='balanced', C=100)

confusion_columns = ['TN', 'FP', 'FN', 'TP']

n_iter = 30

base_conf = pd.DataFrame(
    data=np.zeros((n_iter * n_cross_valid, 4)),
    columns=confusion_columns)

count = -1
for k_iter in range(n_iter):

    cv_keys = cross_validation_keys(plate, n_cross_valid)

    for k_cv in range(n_cross_valid):

        count += 1

        key = str(k_cv)

        zs_params = get_z_score_params(plate, keys=cv_keys[key]['train'])

        X_train = calcualte_descriptors(
            plate, zs_params, keys=cv_keys[key]['train'])

        labels = get_labels(plate, keys=cv_keys[key]['train'])
        y_train = np.zeros((len(labels), ))
        y_train[np.array(labels) == 'ALS'] = 1

        X_valid = calcualte_descriptors(
            plate, zs_params, keys=cv_keys[key]['valid'])

        labels = get_labels(plate, keys=cv_keys[key]['valid'])
        y_valid = np.zeros((len(labels), ))
        y_valid[np.array(labels) == 'ALS'] = 1

        well_classifier.fit(X_train, y_train)

        y_predict = well_classifier.predict(X_valid)

        base_conf.iloc[count, :] = metrics.confusion_matrix(
            y_valid, y_predict).flatten()


#if n_iter > 0:
#    als.save_dataset(
#        base_conf, os.path.join(FOLDER, 'base_conf.pickle'))


# %% Label Permutations

n_iter = 1000

perm_conf = pd.DataFrame(
    data=np.zeros((n_iter * n_cross_valid, 4)),
    columns=confusion_columns)

count = -1
for k_iter in range(n_iter):

    if k_iter % 50 == 0:
        print('{} Permutation iterations = {} {}'.format(
             '=' * 10, k_iter, '=' * 10))

    cv_keys = cross_validation_keys(plate, n_cross_valid)

    for k_cv in range(n_cross_valid):

        count += 1

        key = str(k_cv)

        zs_params = get_z_score_params(plate, keys=cv_keys[key]['train'])

        X_train = calcualte_descriptors(
            plate, zs_params, keys=cv_keys[key]['train'])

        labels = get_labels(plate, keys=cv_keys[key]['train'])
        y_train = np.zeros((len(labels), ))
        y_train[np.array(labels) == 'ALS'] = 1
        y_train = y_train[np.argsort(np.random.rand(len(y_train)))]

        X_valid = calcualte_descriptors(
            plate, zs_params, keys=cv_keys[key]['valid'])

        labels = get_labels(plate, keys=cv_keys[key]['valid'])
        y_valid = np.zeros((len(labels), ))
        y_valid[np.array(labels) == 'ALS'] = 1
        y_valid = y_valid[np.argsort(np.random.rand(len(y_valid)))]

        well_classifier.fit(X_train, y_train)

        y_predict = well_classifier.predict(X_valid)

        perm_conf.iloc[count, :] = metrics.confusion_matrix(
            y_valid, y_predict).flatten()

#if n_iter > 900:
#    als.save_dataset(
#        perm_conf, os.path.join(FOLDER, 'perm_conf.pickle'))


# %% Score Distribution - Permutation vs. True

"""
prevalence = P / total
TPR, sensitivity, recall = TP / P = TP / (TP + FN)
FNR, miss rate = FN/P = FN / (TP + FN)
FPR, fall-out = FP/N = FP / (TN + FP)
TNR, specificity = TN/N = TN / (TN + FP)
Accuracy = TP + TN / total
PPV, precision = TP / test-P = TP / (TP + FP)
FDR = FP / test-P = FP / (TP + FP)
FOR = FN / test-N = FN / (FN + TN)
NPV = TN / test-N = TN / (FN + TN)
positive likelihood ratio, LR+ = TPR / FPR = (TP * N) / (FP * P)
negative likelihood ratio, LR- = FNR / TNR = (FN * N) / (TN * P)
diagnostics odds ratio = LR+ / LR- = (TP * TN) / (FP * FN)

F1 = 2 * (precision * recall) / (precision + recall) =
"""

def pval(distribution, value):
    " ""two-sided pval utility for 1D values"" "

    n_observations = float(len(distribution))
    mu = np.mean(distribution)

    if value > mu:
        v_up = value
        v_low = mu - np.abs(mu - value)
    if value <= mu:
        v_low = value
        v_up = mu + np.abs(mu - value)

    n_out = np.sum(distribution >= v_up) + np.sum(distribution <= v_low)

    return n_out / n_observations


def f_acc(df):
    return (df['TP'].values + df['TN'].values) / df.sum(axis=1).values


def f_sen(df):
    return df['TP'].values / (df['TP'].values + df['FN'].values)

def f_spec(df):
    return df['TN'].values / (df['TN'].values + df['FP'].values)


def f_prec(df):
    return df['TP'].values / (df['TP'].values + df['FP'].values)


def f_f1(df):

    precision = f_prec(df)
    recall = f_sen(df)
    f1 = 2 * (precision * recall) / (precision + recall)
    f1[np.isnan(f1)] = 0

    return f1


foos = {
    'Accuracy' : f_acc,
    'F1' : f_f1,
    'Sensitivity' : f_sen,
    'Specificity' : f_spec}

plt.close('all')
for k_score in foos.keys():

    fig, ax = plt.subplots(num=k_score)
    foo = foos[k_score]

    all_base = foo(base_conf)
    all_perm = foo(perm_conf)

    n_iter = len(all_base) / n_cross_valid
    robust_base = np.median(
        np.reshape(all_base, (n_iter, n_cross_valid)),
        axis=1)

    n_iter = len(all_perm) / n_cross_valid
    robust_perm = np.median(
        np.reshape(all_perm, (n_iter, n_cross_valid)),
        axis=1)

    # calculate pvals
    n_round = 3
    pv = pval(robust_perm, np.mean(robust_base))
    pv = np.round(pv * (10 ** n_round))/ (10 ** n_round)

    for l in ['base', 'perm']:

        if l == 'base':
            label = 'true labels'
            vals = robust_base
            n_bins = 6

        elif l == 'perm':
            label = 'by chance'
            vals = robust_perm
            n_bins = 20

        dither = np.random.randn(len(vals)) * 0.04
        vals += dither
        vals[vals < 0] = 0
        vals[vals > 1] = 1

        counts, bins = np.histogram(vals , bins=n_bins)

        bins = np.append(bins, 1)

        counts = np.double(counts) / np.max(counts)
        counts = np.append(np.insert(counts, 0, 0), 0)

        ax.plot(bins, counts, label=label)

    ax.set_title(k_score + '\n2-sided P_val = ' + str(pv))
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.05, 1.05])
    ax.legend(numpoints=1, loc='best', fontsize=16)

    if enable_save:
        with open(os.path.join(FOLDER, k_score + '.pig'), 'wb') as f:
            pickle.dump(fig, f, pickle.HIGHEST_PROTOCOL)

plt.show()
