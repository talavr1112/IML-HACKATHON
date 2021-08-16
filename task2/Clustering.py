from Preprocessing import get_time_date
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import plotly.graph_objs as go
import plotly.express as plt_express
import json


def get_cluster_df(df):
    """
    Get data to be clustered.
    :param df:
    :return: cluster data frame
    """
    cluster_df = pd.concat([df[["X Coordinate", "Y Coordinate"]],
                            get_time_date(df["Date"])],
                           axis=1)
    # plot_data_cluster(cluster_df)
    return cluster_df


def load_j_cluster():
    """
    Load data frame jason
    :return:
    """
    with open("j_cluster.json", "r") as fd:
        day_cluster_dic = json.load(fd)
    return day_cluster_dic


def points_per_day(df):
    """
    Cluster data frame by k-means algorithm for each day.
    :param df:
    :return:
    """
    cluster_df = get_cluster_df(df)
    cluster_df.dropna(inplace=True)
    points_dict = {}
    for i in range(min(cluster_df["Weekday"]), max(cluster_df["Weekday"]) + 1):
        points_dict[i] = make_Kmeans_clustering(cluster_df, i)

    with open("j_cluster.json", "w") as fd:
        json.dump(points_dict, fd)

    return points_dict


def make_Kmeans_clustering(cluster_df, day):
    """
    This function activate K means algorithm for a given day and data frame.
    :param cluster_df:
    :param day:
    :return: cluster by centroids.
    """
    d = cluster_df[cluster_df["Weekday"] == day].drop(["Weekday"], axis=1)

    est = KMeans(n_clusters=100)
    est = est.fit(d)

    centroids = est.cluster_centers_

    # plot_clustering(d, label, centroid)

    return centroids.tolist()


def plot_clustering(points, labels, centroid):
    """
    This function plot the clustering.
    :param points:
    :param labels:
    :param centroid:
    :return:
    """
    fig = plt_express.scatter_3d(points,
                                 x="X Coordinate", y="Y Coordinate", z="Time",
                                 color=labels)

    fig.update_traces(marker=dict(size=1), selector=dict(mode='markers'))

    clock_time = np.apply_along_axis(lambda x: str(int(x[0] // 60)) + ":" + str(int(x[0] % 60)), axis=0,
                                     arr=centroid[:, 2].reshape((1, -1)))

    fig.add_trace(go.Scatter3d(x=centroid[:, 0], y=centroid[:, 1], z=centroid[:, 2],
                               text=clock_time))

    fig.show()


def plot_data_cluster(df, day=-1):
    """
    This function plot the clustering for specific day.
    :param df:
    :param day:
    :return:
    """
    if day == -1:
        fig = plt_express.scatter_3d(df,
                                     x="X Coordinate", y="Y Coordinate", z="Time",
                                     color="Weekday")
    else:
        fig = plt_express.scatter_3d(df[df["Weekday"] == day],
                                     x="X Coordinate", y="Y Coordinate", z="Time",
                                     color="Weekday")
    fig.update_traces(marker=dict(size=2), selector=dict(mode='markers'))
    fig.show()


def test_Q2(df, x_y_z):

    np_test = get_cluster_df(df)

    np_test.dropna(inplace=True)
    np_test = np_test.to_numpy()


    import math

    score_dic = {}
    for day, all_points in x_y_z.items():
        day_test = np_test[np_test[:, -1] == day][:,:-1]

        for point in all_points:

            distance = np.apply_along_axis(lambda x: math.sqrt((x[0] - point[0]) ** 2 +
                                                    (x[1] - point[1]) ** 2) <= 1640.42, axis=1, arr=day_test)
            time = np.apply_along_axis(lambda x: abs(x[-1] - point[-1]) <= 30, axis=1, arr=day_test)

            print("day : ", str(day), "   p: ", str(point))
            print("d: ", len(distance[distance == True]))
            print("t: ", len(time[time == True]))
            print()
            if day in score_dic:
                score_dic[day].append(np.count_nonzero(distance & time))
            else:
                score_dic[day] = [np.count_nonzero(distance & time)]

    for i, j in score_dic.items():
        print(str(i) + " : " + str(j) + " sum: " + str(np.sum(j)))












