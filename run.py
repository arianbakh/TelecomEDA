import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import seaborn as sns
import sys
import warnings

from math import sin, cos, sqrt, atan2, radians
from matplotlib.backends import backend_gtk3


warnings.filterwarnings('ignore', module=backend_gtk3.__name__)


# Directory Settings
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
DATA_FIELDS = ['sms_in', 'sms_out', 'call_in', 'call_out', 'internet']
SAMPLES = 100
EARTH_RADIUS = 6373.0  # kilometers


def _data_generator():
    for file_name in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.txt'):
            with open(file_path, 'r') as data_file:
                for line in data_file.readlines():
                    split_line = line.strip().split('\t')
                    if len(split_line) == 8 and split_line[2] == '39':
                        yield split_line


def _get_data():
    data = {}
    time_frames = set()
    print('Reading time frames...')
    for line_number, split_line in enumerate(_data_generator()):
        # progress bar
        if (line_number + 1) % 10000 == 0:
            sys.stdout.write('\r[%d]' % (line_number + 1))
            sys.stdout.flush()
        time_frames.add(split_line[1])
    print()  # newline after progress bar
    sorted_time_frames = sorted(time_frames)
    print('Reading data...')
    for line_number, split_line in enumerate(_data_generator()):
        # progress bar
        if (line_number + 1) % 10000 == 0:
            sys.stdout.write('\r[%d]' % (line_number + 1))
            sys.stdout.flush()
        square_id = split_line[0]
        if square_id not in data:
            data[square_id] = {}
            for data_field in DATA_FIELDS:
                data[square_id][data_field] = dict.fromkeys(sorted_time_frames, 0.0)
        for i, data_field in enumerate(DATA_FIELDS):
            time_frame = split_line[1]
            data_field_value = float(split_line[3 + i] or 0)
            data[square_id][data_field][time_frame] = data_field_value
    print()  # newline after progress bar
    return data


def _plot_time_series(time_series_keys,
                      time_series_values,
                      time_series_keys_name,
                      time_series_values_name,
                      time_series_name):
    data_frame = pd.DataFrame({
        'keys': time_series_keys,
        'values': time_series_values,
    })
    plt.subplots(figsize=(10, 5))
    ax = sns.lineplot(x='keys', y='values', data=data_frame)
    ax.set_title(time_series_name, fontsize=16)
    ax.set_xlabel(time_series_keys_name, fontsize=16)
    ax.set_ylabel(time_series_values_name, fontsize=16)
    plt.savefig(os.path.join(OUTPUT_DIR, '%s.png' % time_series_name))
    plt.close('all')


def _compare_time_series(data, key1, key2):
    square_id1 = key1.split('_')[-1]
    data_field1 = '_'.join(key1.split('_')[:-1])
    square_id2 = key2.split('_')[-1]
    data_field2 = '_'.join(key2.split('_')[:-1])
    keys_intersection = set(data[square_id1][data_field1].keys()) & set(data[square_id2][data_field2].keys())
    values1 = []
    keys = []
    for key, value in data[square_id1][data_field1].items():
        if key in keys_intersection:
            keys.append(key)
            values1.append(value)
    values2 = []
    for key, value in data[square_id2][data_field2].items():
        if key in keys_intersection:
            values2.append(value)
    value_name1 = '%s_%s' % (data_field1, square_id1)
    value_name2 = '%s_%s' % (data_field2, square_id2)
    data_frame = pd.DataFrame({
        'keys': keys,
        value_name1: values1,
        value_name2: values2,
    })
    melted_data_frame = pd.melt(
        data_frame,
        id_vars=['keys'],
        value_vars=[
            value_name1,
            value_name2,
        ]
    )
    plt.subplots(figsize=(10, 5))
    ax = sns.lineplot(x='keys', y='value', hue='variable', style='variable', data=melted_data_frame)
    ax.set(xlabel='', ylabel='', title='')
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    plt.savefig(os.path.join(OUTPUT_DIR, 'compare_%s_and_%s.png' % (value_name1, value_name2)))
    plt.close('all')


def _save_correlations_csv(correlations, file_name):
    with open(os.path.join(OUTPUT_DIR, '%s.csv' % file_name), 'w') as correlation_file:
        correlation_file.write('key1,key2,value\n')
        for key1, key2, value in sorted(correlations, key=lambda x: -abs(x[2])):
            correlation_file.write('%s,%s,%f\n' % (key1, key2, value))


def _save_average_correlations_csv(average_correlations):
    with open(os.path.join(OUTPUT_DIR, 'average_correlations.csv'), 'w') as average_correlations_file:
        average_correlations_file.write('key1,key2,correlation_type,value\n')
        for key1, key2, correlation_type, value in sorted(average_correlations, key=lambda x: -x[3]):
            average_correlations_file.write('%s,%s,%s,%f\n' % (key1, key2, correlation_type, value))


def _save_correlations(data):
    sample_keys = random.sample(data.keys(), SAMPLES)
    average_correlations = []
    for data_field1, data_field2 in itertools.combinations(DATA_FIELDS, 2):
        print('Calculating correlation of %s and %s' % (data_field1, data_field2))
        correlations_same = []
        correlations_diff = []
        for square_id1 in sample_keys:
            vector1 = np.array(list(data[square_id1][data_field1].values()))
            for square_id2 in sample_keys:
                vector2 = np.array(list(data[square_id2][data_field2].values()))
                correlation = np.corrcoef(vector1, vector2)[0, 1]
                if correlation:
                    correlation_tuple = (
                        '%s_%s' % (data_field1, square_id1),
                        '%s_%s' % (data_field2, square_id2),
                        correlation
                    )
                    if square_id1 == square_id2:
                        correlations_same.append(correlation_tuple)
                    else:
                        correlations_diff.append(correlation_tuple)
        _save_correlations_csv(correlations_same, '%s_with_%s_same' % (data_field1, data_field2))
        _save_correlations_csv(correlations_diff, '%s_with_%s_diff' % (data_field1, data_field2))
        average_correlations.append(
            (
                data_field1,
                data_field2,
                'same',
                np.mean([abs(item[2]) for item in correlations_same])
            )
        )
        average_correlations.append(
            (
                data_field1,
                data_field2,
                'diff',
                np.mean([abs(item[2]) for item in correlations_diff])
            )
        )
    _save_average_correlations_csv(average_correlations)


def _get_centroid(coordinates):
    longitudes = [item[0] for item in coordinates]
    latitudes = [item[1] for item in coordinates]
    return np.mean(longitudes), np.mean(latitudes)


def _get_distance(point1, point2):
    lon1 = radians(point1[0])
    lat1 = radians(point1[1])
    lon2 = radians(point2[0])
    lat2 = radians(point2[1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = EARTH_RADIUS * c
    return distance * 1000  # meters


def _get_centroids():
    centroids = {}
    with open(os.path.join(DATA_DIR, 'milano-grid.geojson'), 'r') as geojson_file:
        geojson = json.loads(geojson_file.read())
        for feature in geojson['features']:
            cell_id = str(feature['properties']['cellId'])
            coordinates = feature['geometry']['coordinates'][0]
            centroids[cell_id] = _get_centroid(coordinates)
    return centroids


def _get_square_distance(centroids, square_id1, square_id2):
    return _get_distance(centroids[square_id1], centroids[square_id2])


def _get_distance_correlations(centroids):
    distance_correlations = []
    for file_name in os.listdir(OUTPUT_DIR):
        file_path = os.path.join(OUTPUT_DIR, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.csv') and '_with_' in file_name:
            split_file_name = file_name.split('.')[0].split('_with_')
            key1 = split_file_name[0]
            key2 = '_'.join(split_file_name[1].split('_')[:-1])
            correlation_type = split_file_name[1].split('_')[-1]
            if correlation_type == 'diff':
                correlations = []
                distances = []
                with open(file_path, 'r') as data_file:
                    for i, line in enumerate(data_file.readlines()):
                        split_line = line.strip().split(',')
                        if i > 0 and split_line:
                            correlations.append(float(split_line[2]))
                            square_id1 = split_line[0].split('_')[-1]
                            square_id2 = split_line[1].split('_')[-1]
                            distance = _get_square_distance(centroids, square_id1, square_id2)
                            distances.append(distance)
                distance_correlations.append(
                    (
                        key1,
                        key2,
                        np.corrcoef(distances, correlations)[0, 1]
                    )
                )
    return distance_correlations


def run():
    data = _get_data()
    _save_correlations(data)
    centroids = _get_centroids()
    distance_correlations = _get_distance_correlations(centroids)
    _save_correlations_csv(distance_correlations, 'distance_correlations')
    _compare_time_series(data, 'sms_out_9146', 'internet_1399')
    _compare_time_series(data, 'call_in_5063', 'call_out_5063')


if __name__ == '__main__':
    run()
