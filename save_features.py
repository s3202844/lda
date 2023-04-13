import os
import re
import warnings
import numpy as np
import pandas as pd

from pflacco.classical_ela_features import calculate_dispersion
from pflacco.classical_ela_features import calculate_ela_distribution
from pflacco.classical_ela_features import calculate_ela_level
from pflacco.classical_ela_features import calculate_ela_meta
from pflacco.classical_ela_features import calculate_information_content
from pflacco.classical_ela_features import calculate_nbc
from pflacco.classical_ela_features import calculate_pca


warnings.filterwarnings("ignore")


def read_x():
    X = []
    m = 1000+1
    f = open("data/samplingX_010D.txt", "r")
    lines = f.readlines()
    f.close()
    for i in range(100):
        x = []
        content = lines[m*i:m*i+m]
        for j in range(1, m):
            temp = []
            line = re.split(r"[ ]+", content[j][:-1])
            for n in line[1:]:
                temp += [float(n)]
            x += [temp]
        X += [x]
    return X


def calculate_features(X, y):
    keys = []
    values = []
    disp = calculate_dispersion(X, y)
    keys += list(disp.keys())[:-1]
    values += list(disp.values())[:-1]
    ela_distr = calculate_ela_distribution(X, y)
    keys += list(ela_distr.keys())[:-1]
    values += list(ela_distr.values())[:-1]
    ela_level = calculate_ela_level(X, y)
    keys += list(ela_level.keys())[:-1]
    values += list(ela_level.values())[:-1]
    ela_meta = calculate_ela_meta(X, y)
    keys += list(ela_meta.keys())[:-1]
    values += list(ela_meta.values())[:-1]
    ic = calculate_information_content(X, y)
    keys += list(ic.keys())[:-1]
    values += list(ic.values())[:-1]
    nbc = calculate_nbc(X, y)
    keys += list(nbc.keys())[:-1]
    values += list(nbc.values())[:-1]
    pca = calculate_pca(X, y)
    keys += list(pca.keys())[:-1]
    values += list(pca.values())[:-1]
    return keys, values


def read_file(path):
    """read the file from given path and parse content. content format: 100x1000
    float numbers. The numbers are stored in 100 lines. Each line contains
    1000 numbers that separated by space. Finally, the content should be
    normalized to 0-1.

    Args:
        path (str): file path

    Returns:
        numpy array: 100x1000 float numbers
    """
    with open(path, "r") as f:
        content = f.read()
    content = content.splitlines()
    content = [line.split(" ")[:-1] for line in content]
    content = [[float(num) for num in line] for line in content]
    content = np.array(content)
    content = (content - content.min()) / (content.max() - content.min())
    return content


def parse_filename(filename):
    """Parse the file name. file name format: "{problem_id}_{experiment_id}_
    {subtract_lim}_{rotate_lim}_{scale_factor}_{is_subtract}_{is_rotate}_
    {is_scale}.txt". subtract_lim, rotate_lim and scale_factor are float, others
    are int. Use regex to remove ".txt" and parse the rest of the file name.

    Args:
        filename (str): file name

    Returns:
        dict: parsed file name
    """
    filename = re.sub(".txt", "", filename)
    filename = re.split("_", filename)
    filename = [float(num) if "." in num else int(num) for num in filename]
    return {
        "problem_id": filename[0],
        "experiment_id": filename[1],
        "subtract_lim": filename[2],
        "rotate_lim": filename[3],
        "scale_factor": filename[4],
        "is_subtract": filename[5],
        "is_rotate": filename[6],
        "is_scale": filename[7],
    }


def build_dataset(folder_path):
    """Use read_file() and parse_filename() to read files under input directory
    and build pandas dataset. Dataset header: ["problem_id", "experiment_id",
    "subtract_lim", "rotate_lim", "scale_factor","is_subtract", "is_rotate",
    "is_scale", "content"]. Dataset content: array of 100x1000 float numbers
    from read_file().
    ]

    Args:
        folder_path (str): directory path

    Returns:
        pandas.DataFrame: dataset
    """
    dataset = []
    for filename in os.listdir(folder_path):
        content = read_file(os.path.join(folder_path, filename))
        filename = parse_filename(filename)
        filename["content"] = content.tolist()
        dataset.append(filename)
    return pd.DataFrame(dataset)


def add_features_to_dataset(X, dataset, path_to_save):
    """For each record in dataset, use calculate_features() to calulate features
    of the "content" in the record. Then add the features to the record as a new 
    column. For example, X is a numpy array of 100x1000x10 float numbers, and
    "content" is a numpy array of 100x1000 float numbers. 100 is the number of
    samples. So we need to calculate features for each sample, and store all the
    features as a array of 100 "values" from calculate_features(). Finally, add
    this array of 100 "values" to dataset as a new column.

    Args:
        dataset (pandas.DataFrame): dataset
        X (numpy array): 100x1000x10 float numbers

    Returns:
        pandas.DataFrame: dataset
    """
    is_written = False
    header = dataset.columns.values.tolist(
    ) + ["feature_name", "feature_value"]
    records = dataset.values.tolist()
    for i in range(len(records)):
        print(i)
        record = records[i]
        content = record[-1]
        feature_name = []
        feature_value = []
        for j in range(len(X)):
            keys, values = calculate_features(X[j], content[j])
            feature_name = keys
            feature_value += [values]
        records[i] += [feature_name, feature_value]
        temp_dataset = pd.DataFrame([records[i]], columns=header)
        # save dataset to csv file
        if not is_written:
            temp_dataset.to_csv(path_to_save, index=False)
            is_written = True
        else:
            temp_dataset.to_csv(path_to_save, mode="a",
                                index=False, header=False)


X = read_x()
df = build_dataset("data/temp/")
add_features_to_dataset(X, df, "data/dataset1.csv")
