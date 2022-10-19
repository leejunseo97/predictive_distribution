import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from regression_functions.metrics_tracker import MetricsTracker
from regression_functions.predictor import Predictor


class BayesianLinearRegression(Predictor):
    def __init__(self):
        self.df = pd.read_csv('2122_plant_sum_week.csv')
        self.m_n = None
        self.s_n = None
        self.likelihood_var = None
        self.design_matrix = None
        self.degree_x = None

    def create_target_df(self, name_x, name_y):
        df = self.df[[name_x, name_y]].dropna().sort_values(name_x)
        target_x = df[name_x].to_numpy().reshape(-1, 1)
        target_y = df[name_y].to_numpy().reshape(-1, 1)
        return target_x, target_y

    def train(self, name_x, name_y, degree_x, likelihood_var, m_0, s_0):
        self.degree_x = degree_x
        x, y = self.create_target_df(name_x, name_y)
        design_matrix = x
        for i in range(degree_x+1):
            design_matrix = np.concatenate((design_matrix, pow(x, i)), axis=1)
        inv_s_0 = np.linalg.inv(s_0)
        self.s_n = np.linalg.inv(inv_s_0 + (1/(likelihood_var**2))*design_matrix.T.dot(design_matrix))
        self.m_n = self.s_n.dot(inv_s_0.dot(
            m_0) + (1/(likelihood_var**2))*design_matrix.T.dot(y))
        self.likelihood_var = likelihood_var
        self.design_matrix = design_matrix

    def predict(self):
        m = self.design_matrix.dot(self.m_n).flatten()
        var = np.array(list(map(lambda x: x.dot(self.s_n).dot(
            x.T) + self.likelihood_var**2, self.design_matrix)))
        return np.random.normal(m, var), m, var

    def spline(self, smooth_condition):
        tck, u = interpolate.splprep([result_x, result_y], s=0.07)
        splined_x, splined_y = interpolate.splev(np.linspace(0, 1, 100), tck, der=0)
        tck, u = interpolate.splprep([result_x, var], s=0.07)
        splined_x, splined_var = interpolate.splev(np.linspace(0, 1, 100), tck, der=0)



if __name__ == '__main__':
    metrics_tracker = MetricsTracker()
    bayesianLinearRegression = BayesianLinearRegression()

    degree_x = 8
    metrics_tracker.profile(bayesianLinearRegression.train, 'dp', 'img_flower_cluster', degree_x, 0.5, np.ones(
        degree_x + 1).reshape((degree_x+1, 1)), np.eye(degree_x + 1))
    test_y_linear_bayesian, mu, var = metrics_tracker.profile(bayesianLinearRegression.predict)



