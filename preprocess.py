"""
preprocess.py

Contains functions for data loading and preprocessing.
    - preprocess
    - load_dataset
"""

import os
import numpy as np
import pandas as pd

DROP_LABELS = ['unclear']

def preprocess(data):
    """
    Preproccess the data by
        1. Removing cells labeled unclear, etc. as in DROP_LABELS.
        2. Remove features with value = 0 across all cells.
        3. Remove cells with the number of nonzero features less than
            3 median absolute deviation (MAD) from the median of
            the number of nonzero features for cells in the data.
    This procedure follows the preprocessing step in
        Abdelaal, T., Michielsen, L., Cats, D. et al. A comparison of automatic cell
        identification methods for single-cell RNA sequencing data. Genome Biol 20, 194 (2019). 
        https://doi.org/10.1186/s13059-019-1795-z

    Args:
        data (pd.DataFrame): the data. We require that it is in the following format.
            - Each row corresponds to a cell.
            - The first column is 'cell' with the cell names in this column.
            - The final column is 'label' with the cell types in this column.
            - Other columns are the features.
    """
    # Filter out cells with 'unclear' label
    data = data.loc[~data['label'].isin(DROP_LABELS)]

    # Find features with values = 0 across all cells
    columns = data.columns[1:-1]  # Exclude 'cell' and 'label' columns
    nonzero_columns = []
    for col in columns:
        if (data[col] != 0).any():
            nonzero_columns.append(col)
            
    # Filter out columns with values = 0 across all cells
    data = data[['cell', 'label'] + nonzero_columns]

    # Filter out cells with number of detected genes less than 3 MAD 
    # from the median of number of detected genes
    num_detected = data.iloc[:, 2:].apply(lambda features: np.sum(features != 0), axis=1)
    median_num_detected = np.median(num_detected)
    absolute_deviation = np.abs(num_detected - median_num_detected)
    mad = np.median(absolute_deviation)
    data['num_detected'] = num_detected
    data = data.loc[median_num_detected - data['num_detected'] <= 3 * mad]

    # Extract the features and the labels
    X = data.iloc[:, 2:-1].to_numpy()
    y = data.loc[:, 'label'].to_numpy()

    return X, y

def load_dataset(datadir, dataset):
    """
    Load the specified dataset.

    Args:
        datadir (str): path to the directory containing the dataset.
        dataset (str): the dataset to load. Currently, the options are 'muraro' and 'tabula-muris'.

    Returns:
        data (pd.DataFrame): the data in the format compatible with the preprocess function.
    """
    if dataset == 'muraro':
        # Loading data
        samples = pd.read_csv(os.path.join(datadir, 'data.csv'), delimiter='\t').transpose()
        samples.index.name = 'cell'
        samples.reset_index(inplace=True)
        labels = pd.read_csv(os.path.join(datadir, 'cell_type_annotation_Cels2016.csv'), 
                                delimiter='\t', skiprows=1, names=['cell', 'label'])

        # Formatting data
        samples['cell'] = samples['cell'].astype('string')
        labels['cell'] = labels['cell'].astype('string')
        labels['cell'] = labels['cell'].apply(lambda name: name.replace('.', '-'))

        # Merge samples with features to labels
        data = samples.merge(labels, on='cell', how='inner')

    elif dataset == 'tabula-muris':
        raise NotImplementedError()

    else:
        raise ValueError('Invalid dataset: {}'.format(dataset))
    
    return data