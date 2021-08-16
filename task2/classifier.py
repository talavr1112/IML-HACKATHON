import Preprocessing
import Clustering
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def send_police_cars(date_list):
    """
    This function sends police cars by predicting which area would have the highest probability to have
    a crime- using clustering.
    :param date_list:
    :return:
    """
    ret_list = []
    day_cluster_dic = Clustering.load_j_cluster()
    for date in date_list:

        index = np.random.choice(len(day_cluster_dic["0"]), 30, replace=False)

        temp_date = pd.to_datetime(date)
        pred_day = temp_date.weekday()
        points = np.array(day_cluster_dic[str(pred_day)])[index]

        clusters_by_day = np.apply_along_axis(
                lambda x: pd.Timestamp(year=temp_date.year, month=temp_date.month,
                                            day=temp_date.day, hour=int(x[0] // 60),
                                            minute=int(x[0] % 60)),
                axis=1, arr=points[:, -1].reshape((-1, 1)))

        temp_list = []
        for i in range(len(clusters_by_day)):
            temp_list.append((points[i][0], points[i][1], clusters_by_day[i]))

        ret_list.append(temp_list)

    return ret_list


def predict(path):
    """
    Make prediction by random forest - bagging will decrease the variance and will result
    in generalization.
    :param path:
    :return: committee decision.
    """
    X_pred = pd.read_csv(path, index_col=0)
    random_forest_model = Preprocessing.load_model()
    X_ = Preprocessing.preprocess_data(X_pred, pred=True)
    return random_forest_model.predict(X_)


if __name__ == '__main__':
    df = pd.read_csv("full_data/Dataset_crimes.csv", index_col=0)

    X, y = Preprocessing.preprocess_data(df)

    # train model
    random_forest = RandomForestClassifier(n_estimators=200, max_depth=7)

    random_forest.fit(X, y)

    Preprocessing.save_model(random_forest)

    # create clustering
    points_d = Clustering.points_per_day(df)














