import numpy as np
from scipy.optimize import minimize


class HillModel:

    def __init__(self, x, y, **kwargs):
        self.x = x
        self.y = y
        self.fit(**kwargs)

    def _predict(self, x, params):
        return self.model(x, params[:2], params[2:])

    def predict(self, x):
        return self._predict(x, self.popt)

    def objective(self, params):
        prediction = self._predict(self.x, params)
        return self.SSR(prediction, self.y)

    @staticmethod
    def SSR(x, y):
        return ((x-y)**2).sum()

    @staticmethod
    def model(x, n, km):
        #return (x**n) / (1 + ((x**n) + (km**n)).sum())
        return ( (x/km)**n ) / (((x/km)**n).sum(axis=1).reshape(-1, 1) + 1 )

    def fit(self, p0=None, bounds=None, method='SLSQP', tol=1e-10, **kwargs):
        if p0 is None:
            p0 = np.array([1, 1, 1, 1])
        if bounds is None:
            bounds = [(0.1, 25), (0.1, 25), (0.01, 1000), (0.01, 1000)]
        result = minimize(self.objective, p0, method=method, bounds=bounds, tol=tol, **kwargs)

        # raise error if fit fails
        if result['success'] != True:
            print('HillModel fit failed.')
            #raise ValueError('Optimization failed during HillModel fit.')

        self.result = result
        self.popt = result['x']
        self.hill_coefficient = self.popt[:2]
        self.km = self.popt[2:]
        self.yp = self.predict(self.x)
