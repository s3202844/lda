import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from umap import UMAP


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
        # map subtract_lim (90, 60, 30) to alpha (0.2, 0.5, 0.8)
        if float(header[2]) == 90. or float(header[2]) == 900.:
            alpha = 0.2
        elif float(header[2]) == 60. or float(header[2]) == 600.:
            alpha = 0.5
        else:
            alpha = 0.8
        marker = "v"
        s = marker_size * alpha
    elif int(header[6]) == 1:
        label += ", random rotation."
        alpha = 1
        marker = "x"
        s = 20
    elif int(header[7]) == 1:
        label += ", scale factor: {}".format(header[4])
        top = np.abs(np.log2(float(header[4])))
        if top == 3.:
            alpha = 0.2
        elif top == 2.:
            alpha = 0.5
        else:
            alpha = 0.8
        marker = "<" if float(header[4]) < 1 else ">"
        s = marker_size * alpha
    else:
        alpha = 1
        marker = "o"
        s = 50
    color = base_colors[int(header[0]) - 1] + "{:02x}".format(int(255 * alpha))
    # color = base_colors[int(header[0]) - 1]
    return label, color, marker, s


def legend_elements(type):
    """Get legend elements.

    Args:
        type (str): "raw", "subtract", "rotate", "scale"

    Returns:
        list: legend elements
    """
    marker_size = 200
    # set 5 colors for 5 problems, consider color blindness
    base_colors = ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725']
    # define legend
    legend_elements = []
    for i in range(5):
        line = plt.Line2D([0], [0], marker='o', color='w',
                          label='problem {}'.format(i + 1),
                          markerfacecolor=base_colors[i],
                          markersize=50**0.5)
        # set marker edge color to black
        line.set_markeredgecolor('black')
        legend_elements.append(line)
    if type == "subtract":
        alphas = [0.8, 0.5, 0.2]
        for i in range(3):
            # map subtract_lim (90, 60, 30) to alpha (0.2, 0.5, 0.8)
            alpha = alphas[i]
            color = "#000000{:02x}".format(int(255 * alpha))
            s = marker_size * alpha
            line = plt.Line2D([0], [0], marker='v', color='w',
                              label='translation limit: {}'.format(30.*(i+1)),
                              markerfacecolor=color, markersize=s**0.5)
            # set marker edge color to black
            line.set_markeredgecolor('black')
            legend_elements.append(line)
    if type == "rotate":
        line = plt.Line2D([0], [0], marker='x', color='w',
                          label='random rotation',
                          markersize=20**0.5)
        # set marker edge color to black
        line.set_markeredgecolor('black')
        legend_elements.append(line)
    if type == "scale":
        alphas = [0.8, 0.5, 0.2]
        for i in range(3):
            alpha = alphas[i]
            color = "#000000{:02x}".format(int(255 * alpha))
            s = marker_size * alpha
            line = plt.Line2D([0], [0], marker='>', color='w',
                              label='scale factor: {}'.format(2**(i+1)),
                              markerfacecolor=color, markersize=s**0.5)
            # set marker edge color to black
            line.set_markeredgecolor('black')
            legend_elements.append(line)
        for i in range(3):
            alpha = alphas[i]
            color = "#000000{:02x}".format(int(255 * alpha))
            s = marker_size * alpha
            line = plt.Line2D([0], [0], marker='<', color='w',
                              label='scale factor: {}'.format(0.5**(i+1)),
                              markerfacecolor=color, markersize=s**0.5)
            # set marker edge color to black
            line.set_markeredgecolor('black')
            legend_elements.append(line)
    return legend_elements


def fit_reducer(samples):
    """Fit umap to samples.

    Args:
        samples (numpy array): samples

    Returns:
        umap object: umap object
    """
    # fit umap
    reducer = UMAP(n_components=2, n_neighbors=20, min_dist=0.9,
                   metric='euclidean', random_state=42)
    reducer.fit(samples)
    return reducer


def plot_reducer(fig, ax, reducer, samples, labels, title, exp_type, origin=False):
    """Plot LDA.

    Args:
        lda (sklearn.discriminant_analysis.LinearDiscriminantAnalysis): LDA
        samples (numpy array): samples
        y (numpy array): labels
    """
    plot_rate = 0.04
    # plot LDA
    X = reducer.transform(samples)
    is_subtract = False
    is_rotate = False
    is_scale = False
    for i in range(len(labels)):
        is_subtract = True if labels[i][5] == 1 else is_subtract
        is_rotate = True if labels[i][6] == 1 else is_rotate
        is_scale = True if labels[i][7] == 1 else is_scale
        is_trans = labels[i][5] == 1 or labels[i][6] == 1 or labels[i][7] == 1
        if is_trans and np.random.rand() > plot_rate:
            continue
        _, color, marker, s = format_labels(labels[i])
        if not origin and not is_trans:
            ax.scatter(-X[i, 0], X[i, 1], facecolors=color, marker=marker,
                    s=s, edgecolors='black' if not is_rotate else None,
                    linewidths=1, alpha=0.2)
        else:
            ax.scatter(-X[i, 0], X[i, 1], facecolors=color, marker=marker,
                    s=s, edgecolors='black' if not is_rotate else None,
                    linewidths=1)
    # set legend
    legend_type = "subtract" if is_subtract else "rotate" if is_rotate else \
        "scale" if is_scale else "raw"
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    # locate at bottom left, size is slightly smaller than default, 2 columns
    ax.legend(handles=legend_elements(legend_type),
              loc='lower left', ncol=2, fontsize=9)
    fig.tight_layout()
    # save plot
    if exp_type == "x":
        fig.savefig("results/search_space/{}.png".format(title))
    elif exp_type == "y":
        fig.savefig("results/response/{}.png".format(title))
    else:
        fig.savefig("results/{}.png".format(title))
    # ax.clear()


if __name__ == "__main__":
    # df = parse_features_dataset("data/dataset_x.csv")
    # samples, labels = get_samples(df)
    # reducer = fit_reducer(samples)
    # plot_reducer(reducer, samples, labels, title="2d", exp_type="x")
    df_x = parse_features_dataset("data/dataset_x.csv")
    df_y = parse_features_dataset("data/dataset_y.csv")
    df_concat = pd.concat([df_x, df_y])
    samples, labels = get_samples(df_concat)
    reducer = fit_reducer(samples)
    fig0, ax0 = plt.subplots()
    plot_reducer(fig0, ax0, reducer, samples, labels, title="2d", exp_type=None, origin=True)
    for exp_type in ["x", "y"]:
        if exp_type == "x":
            df = df_x
        elif exp_type == "y":
            df = df_y
        # prepare data
        samples_subtract, labels_subtract = get_samples(df, is_subtract=1)
        samples_rotate, labels_rotate = get_samples(df, is_rotate=1)
        samples_scale, labels_scale = get_samples(df, is_scale=1)
        # plot LDA for subtract
        samples_subtract = np.concatenate((samples, samples_subtract))
        labels_subtract = np.concatenate((labels, labels_subtract))
        fig, ax = plt.subplots()
        plot_reducer(fig, ax, reducer, samples_subtract, labels_subtract,
                     title="2d_subtract", exp_type=exp_type)
        # plot LDA for rotate
        if samples_rotate.shape[0] > 0:
            samples_rotate = np.concatenate((samples, samples_rotate))
            labels_rotate = np.concatenate((labels, labels_rotate))
            fig, ax = plt.subplots()
            plot_reducer(fig, ax, reducer, samples_rotate, labels_rotate,
                         title="2d_rotate", exp_type=exp_type)
        # plot LDA for scale
        samples_scale = np.concatenate((samples, samples_scale))
        labels_scale = np.concatenate((labels, labels_scale))
        fig, ax = plt.subplots()
        plot_reducer(fig, ax, reducer, samples_scale, labels_scale,
                     title="2d_scale", exp_type=exp_type)
    # # df = parse_features_dataset("data/dataset_y.csv")
    # # samples, labels = get_samples(df)
    # # reducer = fit_reducer(samples)
    # # # plot_reducer(reducer, samples, labels, title="2d", exp_type="x")
    # # for problem_id in range(1, 6):
    # #     X = reducer.transform(samples)
    # #     fig, ax = plt.subplots()
    # #     is_subtract = False
    # #     is_rotate = False
    # #     is_scale = False
    # #     for i in range(len(labels)):
    # #         is_subtract = True if labels[i][5] == 1 else is_subtract
    # #         is_rotate = True if labels[i][6] == 1 else is_rotate
    # #         is_scale = True if labels[i][7] == 1 else is_scale
    # #         is_trans = labels[i][5] == 1 or labels[i][6] == 1 or labels[i][7] == 1
    # #         if is_trans:
    # #             continue
    # #         if int(labels[i][0]) == problem_id:
    # #             alpha = 1
    # #         else:
    # #             alpha = 0.2
    # #         _, color, marker, s = format_labels(labels[i])
    # #         ax.scatter(-X[i, 0], X[i, 1], facecolors=color, marker=marker,
    # #                 s=s, edgecolors='black' if not is_rotate else None,
    # #                 linewidths=1, alpha=alpha)
    # #     # set legend
    # #     legend_type = "subtract" if is_subtract else "rotate" if is_rotate else \
    # #         "scale" if is_scale else "raw"
    # #     ax.set_xlabel("Component 1")
    # #     ax.set_ylabel("Component 2")
    # #     # locate at bottom left, size is slightly smaller than default, 2 columns
    # #     ax.legend(handles=legend_elements(legend_type),
    # #             loc='lower left', ncol=2, fontsize=9)
    # #     fig.tight_layout()
    # #     # save plot
    # #     fig.savefig("results/umap_{}.png".format(problem_id))
    # #     ax.clear()
