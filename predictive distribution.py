import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate

from regression_functions.metrics_tracker import MetricsTracker
from regression_functions.predictor import Predictor


class BayesianLinearRegression(Predictor):
    def __init__(self):
        self.df = pd.read_csv("2122_plant_sum_week.csv")
        self.m_n = None
        self.s_n = None
        self.likelihood_var = None
        self.design_matrix = None
        self.degree_x = None

    def create_target_df(self, name_x, name_y):
        df = self.df[[name_x, name_y]].dropna().sort_values(name_x)
        self.target_x = df[name_x].to_numpy().reshape(-1, 1)
        self.target_y = df[name_y].to_numpy().reshape(-1, 1)
        return self.target_x, self.target_y

    def train(self, name_x, name_y, degree_x, likelihood_var, m_0, s_0):
        self.degree_x = degree_x
        x, y = self.create_target_df(name_x, name_y)
        design_matrix = x
        for i in range(2,degree_x + 1):
            design_matrix = np.concatenate((design_matrix, pow(x, i)), axis=1)
        bias = np.ones((x.shape[0], 1))
        design_matrix = np.concatenate((bias, design_matrix), axis=1)
        inv_s_0 = np.linalg.inv(s_0)
        self.s_n = np.linalg.inv(inv_s_0 + (1 / (likelihood_var**2)) * design_matrix.T.dot(design_matrix))
        self.m_n = self.s_n.dot(inv_s_0.dot(m_0) + (1 / (likelihood_var**2)) * design_matrix.T.dot(y))
        self.likelihood_var = likelihood_var
        self.design_matrix = design_matrix

    def predict(self, x):
        design_matrix = x
        for i in range(2,degree_x + 1):
            design_matrix = np.concatenate((design_matrix, pow(x, i)), axis=1)
        bias = np.ones((x.shape[0], 1))
        design_matrix = np.concatenate((bias, design_matrix), axis=1)
        m = design_matrix.dot(self.m_n).flatten()
        var = np.array(list(map(lambda x: x.dot(self.s_n).dot(x.T) + self.likelihood_var**2, design_matrix)))
        x = design_matrix[:,1].flatten()
        return np.random.normal(m, var), x, m, var

    def spline(self, x, y, var, smooth_condition):
        tck, u = interpolate.splprep([x, y], s=smooth_condition)
        splined_x, splined_y = interpolate.splev(np.linspace(0, 1, 100), tck, der=0)
        tck, u = interpolate.splprep([x, var], s=smooth_condition)
        splined_x, splined_var = interpolate.splev(np.linspace(0, 1, 100), tck, der=0)
        return splined_x, splined_y, splined_var


if __name__ == "__main__":
    ENVRIONMENT_NAME = ['co2', 'dp', 'hd', 'hmdt', 'hmdt_abs', 'hmdt_sat', 'ior', 'temp', 'vpd']
    GROWTH_NAME = ['img_growth_length', 'img_stem_diameter', 'img_flower_height', 'img_flower_cluster', 'img_fruit_cluster', 'plant_leaf_length_width_rt', 'plant_leaf_width_length_rt', 'plant_lai', 'plant_flower_height_stem_dia_rt', 'fruit_weight_fresh', 'man_plant_height', 'man_growth_length', 'man_flower_height', 'man_stem_diameter', 'man_fl_cluster', 'man-fr-cluster', 'man_leaf_max_length', 'man_leaf_max_width', 'man_leaf', 'man_flower', 'man_fruit', 'man_fruit_length', 'man_fruit_width']
    degree_x = 7
    likelihood_var = 0.5
    spline_smooth_condition = 0.07
    spline = True
    n_sigma = 1
    name_x = "dp"
    name_y = "img_flower_cluster"

    metrics_tracker = MetricsTracker()
    BLR = BayesianLinearRegression()
    for name_x in GROWTH_NAME:
        for name_y in ENVRIONMENT_NAME:
            # 훈련
            metrics_tracker.profile(BLR.train, name_x, name_y, degree_x, likelihood_var, np.ones(degree_x + 1).reshape((degree_x + 1, 1)), np.eye(degree_x + 1))
            # 테스트
            print(f'[({name_x}) to ({name_y}) data fitting...]')
            test_y_linear_bayesian, x, m, var = metrics_tracker.profile(BLR.predict, BLR.design_matrix)
            splined_x, splined_m, splined_var = BLR.spline(
                BLR.target_x.flatten(), m, var, spline_smooth_condition)
            # 플롯
            plt.title(f'Bayesian Linear Regression (Degree {degree_x})', fontsize=15, fontweight='bold')
            plt.xlabel(f'{name_x}', fontsize=12, fontweight='bold')
            plt.ylabel(f'{name_y}', fontsize=12, fontweight='bold')
            plt.scatter(BLR.target_x, BLR.target_y)
            # plt.plot(x, m, label='mean of posterior')
            plt.plot(splined_x, splined_m, label='splined mean of posterior')
            plt.fill_between(splined_x, splined_m - n_sigma*splined_var, splined_m +
                             n_sigma*splined_var, alpha=0.4, label=f'{n_sigma} sigma variance')
            plt.legend(loc=2, prop={'weight':'bold'})
            plt.tight_layout()
            plt.savefig(f'growth_env_figs/{name_x}_{name_y}.png')
            plt.clf()
    # # 플롯
    # if spline:
    #     splined_x, splined_m, splined_var = BLR.spline(BLR.target_x.flatten(), m, var, spline_smooth_condition)
    #     # plt.plot(x, m, label='mean of posterior')
    #     plt.plot(splined_x, splined_m, label='splined mean of posterior')
    #     plt.fill_between(splined_x, splined_m - n_sigma*splined_var, splined_m + n_sigma*splined_var, alpha=0.4, label=f'{n_sigma} sigma variance')
    # else:
    #     plt.plot(x, m, label='mean of posterior')
    #     plt.fill_between(x, m - 1*var, m + 1*var, alpha=0.4, label=f'{n_sigma} sigma variance')
    # plt.scatter(BLR.target_x, BLR.target_y)
    # plt.title(f'Bayesian Linear Regression (Degree {degree_x})', fontsize=15, fontweight='bold')
    # plt.tight_layout()

    # plt.xlabel(f'{name_x}', fontsize=12, fontweight='bold')
    # plt.ylabel(f'{name_y}', fontsize=12, fontweight='bold')
    # plt.legend(loc=2, prop={'weight':'bold'})
    # plt.tight_layout()
    # plt.show()
