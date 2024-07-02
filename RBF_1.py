from pyDOE import lhs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


class RBF(object):
    """
        param = {
            'k': None,
            'x': None,
            'y': None,
            'var_mod': None,
            'ker_mod': None,
        }
    """

    def __init__(self, param):
        self.k = int(param['k'])
        self.x = param['x']
        self.y = param['y']
        self.n = self.x.shape[0]
        self.d = self.x.shape[1]
        self.var_mod = param['var_mod']
        self.ker_mod = param['ker_mod']
        self.center = None
        self.var = None
        self.w = None
        self.b = None

    def cal_center(self):
        km = KMeans(n_clusters=self.k, init='k-means++', max_iter=300, n_init=10, random_state=0)
        km.fit(self.x)
        self.center = km.cluster_centers_

    def cal_var(self):
        diff = pairwise_distances(self.center, self.center)
        max_diff = np.max(diff[0])
        bate = np.power((max_diff * self.n * self.d), (-1 / self.d))
        match self.var_mod:
            case 'bate1':
                return bate
            case 'bate2':
                return bate * np.power(2, 4 * 1)
            case 'bate3':
                return bate * np.power(2, 4 * 2)
            case 'bate4':
                return bate * np.power(2, 4 * 3)
            case 'max_dis':
                return max_diff
            case 'max_div_k':
                return max_diff / np.sqrt(2 * self.k)
            case 'mean_dis':
                return np.sum(diff[0]) / (self.k - 1)
            case 'dou_mean_dis':
                return 2 * np.sum(diff[0]) / (self.k - 1)

    @staticmethod
    def cal_hlo(ker_mod, dis, var):
        match ker_mod:
            case 'gaussian':
                return np.exp(-0.5 * np.power(dis / var, 2))
            case 'reflect':
                return 1/(1 + np.exp(np.power(dis / var, 2)))
            case 'laplacian':
                return np.exp(-(dis / var))
            case 'inverse':
                return 1 / np.sqrt(np.power(dis, 2) + np.power(var, 2))
            case 'multiquadric':
                return np.sqrt(np.power(dis, 2) + np.power(var, 2))
            case 'rational':
                return 1 - (np.power(dis, 2) / (np.power(dis, 2) + np.power(var, 2)))

    def fit(self):
        self.cal_center()
        dis = pairwise_distances(self.x, self.center)
        self.var = self.cal_var()
        var = np.full_like(dis, self.var)
        hlo = self.cal_hlo(self.ker_mod, dis, var)
        ones = np.ones((hlo.shape[0], 1))
        hlo = np.hstack((hlo, ones))
        hlo_inv = np.linalg.pinv(hlo)
        w_b = np.dot(hlo_inv, self.y)
        self.w = w_b[0:self.k]
        self.b = w_b[self.k]

    def model_data(self):
        return self


def predict(param):
    x = param['x']
    w = param['w']
    c = param['c']
    b = param['b']
    v = param['v']
    m = param['ker_mod']
    if np.ndim(x) == 1:
        x = x[np.newaxis, :]
    d = pairwise_distances(x, c)
    v = np.full_like(d, v)
    hlo = RBF.cal_hlo(m, d, v)
    y = np.dot(hlo, w) + b
    return y


if __name__ == '__main__':
    ub = 1.0
    lb = 0.0
    n = 100
    d = 1
    method = ['max_centers_distance', 'max_centers_distance_divided_by_k']
    noise = np.random.uniform(low=-0.1, high=0.1, size=(n, d))
    sample = lhs(d, samples=n)
    t_x = sample * (ub - lb) + lb
    y_noise = (0.5 + (0.4 * np.cos((t_x * np.pi * 2.5)))) + noise
    rbf_param = {
        'k': 10,
        'x': t_x,
        'y': y_noise,
        'var_mod': ['bate1', 'bate2', 'bate3', 'bate4', 'max_dis', 'max_div_k', 'mean_dis', 'dou_mean_dis'][4],
        'ker_mod': ['gaussian', 'reflect', 'laplacian', 'inverse', 'multiquadric', 'rational'][0],
    }
    rbf = RBF(rbf_param)
    rbf.fit()
    p_x = lhs(d, samples=n) * (ub - lb) + lb
    t_y = (0.5 + (0.4 * np.cos((p_x * np.pi * 2.5))))
    pre_param = {
        'x': p_x,
        'w': rbf.model_data().w,
        'c': rbf.model_data().center,
        'b': rbf.model_data().b,
        'v': rbf.model_data().var,
        'ker_mod': rbf.model_data().ker_mod
    }
    p_y = predict(pre_param)

    plt.figure(num=1)
    p_1, p_2 = rbf_param['var_mod'], rbf_param['ker_mod']
    title_text = f'{p_1}&{p_2}'
    plt.scatter(p_x, t_y, c='#ED5C27', label='y_actual')
    plt.scatter(p_x, p_y, c='#C0FF3E', label='y_predict')
    plt.legend(loc='lower right')
    plt.title(title_text)
    plt.show()




