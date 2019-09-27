import numpy as np
from eolearn.core import EOTask, FeatureType
from scipy.optimize import curve_fit

import itertools as it


def doubly_logistic(x, c1, c2, a1, a2, a3, a4, a5):
    return c1 + c2 * np.piecewise(x, [x < a1, x >= a1],
                                  [lambda y: np.exp(-((a1 - y) / a4) ** a5),
                                   lambda y: np.exp(-((y - a1) / a2) ** a3)])


def _fit_optimize(x, y, p0=None):
    """
    :param x: Vertical coordinates of points
    :type x: List of floats
    :param y: Horizontal coordinates of points
    :type y: List of floats
    :param p0: Initial parameter guess
    :type p0: List of floats
    :return: List of optimized parameters [c1, c2, a1, a2, a3, a4, a5]
    """

    bounds_lower = [np.min(y), -np.inf, x[0], 0.15, 1, 0.15, 1, ]
    bounds_upper = [np.max(y), np.inf, x[-1], np.inf, np.inf, np.inf, np.inf, ]
    if p0 is None:
        p0 = [np.mean(y), 0.2, (x[-1]-x[0])/2, 0.15, 10, 0.15, 10]
    optimal_values = curve_fit(doubly_logistic, x, y, p0,
                               bounds=(bounds_lower, bounds_upper), maxfev=1000000,
                               absolute_sigma=True)
    return optimal_values[0]


class DoublyLogisticApproximationTask(EOTask):
    """
    EOTask class for calculation of doubly logistic approximation on each pixel for a feature. The task creates new
    feature with the function parameters for each pixel as vectors.

    :param feature: A feature on which the function will be approximated
    :type feature: str
    :param new_feature: Name of the new feature where parameters of the function are saved
    :type new_feature: str
    :param p0: Initial parameter guess
    :type p0: List of floats length 7 corresponding to each parameter
    """

    def __init__(self, feature, new_feature='DOUBLY_LOGISTIC_PARAM', p0=None, mask_data=False):
        self.feature = feature
        self.p0 = p0
        self.new_feature = new_feature
        self.mask_data = mask_data

    def execute(self, eopatch):
        data = eopatch.data[self.feature]

        times = np.asarray([time.toordinal() for time in eopatch.timestamp])
        times = (times - times[0])/(times[-1] - times[0])

        t, h, w, _ = data.shape

        if self.mask_data:
            valid_data_mask = eopatch.mask['VALID_DATA']
        else:
            valid_data_mask = np.ones((t, h, w), dtype=bool)

        all_parameters = np.zeros((h, w, 7))

        for ih, iw in it.product(range(h), range(w)):
            valid_curve = data[:, ih, iw,:][valid_data_mask[:, ih, iw]]
            valid_times = times[valid_data_mask[:, ih, iw].squeeze()]

            all_parameters[ih, iw] = _fit_optimize(valid_times, valid_curve, self.p0)

        eopatch.data_timeless[self.new_feature] = all_parameters

        return eopatch
