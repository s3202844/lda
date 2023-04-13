import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def parse_features_dataset(filepath):
    """Parse features dataset from file. dataset header: "problem_id,
    experiment_id, subtract_lim, rotate_lim, scale_factor, is_subtract,
    is_rotate, is_scale, content, feature_name, feature_value". These columns
    are stored as string: "content" is a 2d list of 100x1000 float numbers,
    "feature_name" is a list of features, "feature_value" is a 2d list of
    100xlen(feature_name) float numbers.

    Args:
        filepath (str): path to the file

    Returns:
        pandas.DataFrame: dataset
    """
    df = pd.read_csv(filepath)
    df["content"] = df["content"].apply(lambda x: eval(x))
    df["feature_name"] = df["feature_name"].apply(lambda x: eval(x))
    df["feature_value"] = df["feature_value"].apply(lambda x: eval(x))
    return df


def get_samples(dataset, is_subtract=0, is_rotate=0, is_scale=0):
    """Get samples from dataset.

    Args:
        dataset (pandas.DataFrame): dataset

    Returns:
        numpy array: samples
        numpy array: labels
    """
    # get samples and labels, ordered by problem_id
    df = dataset[dataset["is_subtract"] == is_subtract]
    df = df[df["is_rotate"] == is_rotate]
    df = df[df["is_scale"] == is_scale]
    df = df.sort_values(by=["problem_id"])
    # get feature_value and first 8 columns as labels
    feature_value = df["feature_value"].values.tolist()
    y = df.iloc[:, :8].values.tolist()
    # get samples and labels
    samples = []
    labels = []
    for i in range(len(feature_value)):
        for j in range(100):
            f = feature_value[i][j]
            samples.extend([f])
            labels.extend([y[i]])
    samples = np.array(samples)
    labels = np.array(labels)
    return samples, labels


def format_labels(header):
    """Format header to string by problem_id, is_subtract, is_rotate, is_scale.

    Args:
        header (numpy array): labels

    Returns:
        list: formatted labels
        list: colors
    """
    marker_size = 200
    # set 5 colors for 5 problems, consider color blindness
    base_colors = ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725']
    # Adjust transparency of base_colors according to subtract_lim or
    # scale_factor
    label = "problem {}".format(int(header[0]))
    if int(header[5]) == 1:
        label += ", random translation limit: {}".format(header[2])
        alpha = 1 - float(header[2]) / 100
        marker = "v"
        s = marker_size * alpha
    elif int(header[6]) == 1:
        label += ", random rotation."
        alpha = 1
        marker = "x"
        s = 20
    elif int(header[7]) == 1:
        label += ", scale factor: {}".format(header[4])
        alpha = float(header[4]) / 10
        marker = "o"
        s = 10
    else:
        alpha = 1
        marker = "o"
        s = marker_size
    color = base_colors[int(header[0]) - 1] + "{:02x}".format(int(255 * alpha))
    # color = base_colors[int(header[0]) - 1]
    return label, color, marker, s


def fit_lda(samples, y):
    """Fit LDA to samples.

    Args:
        samples (numpy array): samples
        y (numpy array): labels

    Returns:
        sklearn.discriminant_analysis.LinearDiscriminantAnalysis: LDA
    """
    # use LDA to reduce samples to 2d
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(samples, y)
    return lda


def plot_lda(lda, samples, labels, title):
    """Plot LDA.

    Args:
        lda (sklearn.discriminant_analysis.LinearDiscriminantAnalysis): LDA
        samples (numpy array): samples
        y (numpy array): labels
    """
    plot_rate = 0.1
    # plot LDA
    X = lda.transform(samples)
    fig, ax = plt.subplots()
    legends = []
    # for i in range(len(colors)):
    #     c = colors[i]
    #     if c not in legends:
    #         legends += [c]
    #         plt.scatter([0], [0], c=c, label=labels[i], s=0)
    for i in range(len(labels)):
        is_trans = labels[i][5] == 1 or labels[i][6] == 1 or labels[i][7] == 1
        if is_trans and np.random.rand() > plot_rate:
            continue
        label, color, marker, s = format_labels(labels[i])
        ax.scatter(X[i, 0], X[i, 1], label=label, facecolors=color,
                   marker=marker, s=s, edgecolors='black', linewidths=1)
    # plt.legend()
    plt.tight_layout()
    # save plot
    fig.savefig("results/search_space/{}.png".format(title))
    ax.clear()


if __name__ == "__main__":
    df = parse_features_dataset("dataset.csv")
    samples, labels = get_samples(df)
    lda = fit_lda(samples, np.ravel(labels[:, 0]))
    plot_lda(lda, samples, labels, title="lda")
    samples_subtract, labels_subtract = get_samples(df, is_subtract=1)
    samples_rotate, labels_rotate = get_samples(df, is_rotate=1)
    samples_scale, labels_scale = get_samples(df, is_scale=1)
    # plot LDA for subtract
    samples_subtract = np.concatenate((samples, samples_subtract))
    labels_subtract = np.concatenate((labels, labels_subtract))
    plot_lda(lda, samples_subtract, labels_subtract, title="lda_subtract")
    # plot LDA for rotate
    samples_rotate = np.concatenate((samples, samples_rotate))
    labels_rotate = np.concatenate((labels, labels_rotate))
    plot_lda(lda, samples_rotate, labels_rotate, title="lda_rotate")
    # plot LDA for scale
    samples_scale = np.concatenate((samples, samples_scale))
    labels_scale = np.concatenate((labels, labels_scale))
    plot_lda(lda, samples_scale, labels_scale, title="lda_scale")
