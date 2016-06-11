
import numpy as np
from pandas import DataFrame, Series
from matplotlib import pyplot as plt


class Oval:
    def __init__(self, a, b):
        self._a = a
        self._b = b
        self._df = None
        self._dt = None

    def _get_x(self, t):
        return self._a * np.cos(t)

    def _get_y(self, t):
        return self._b * np.sin(t)

    def _calc_e1(self, df):
        dt = df.index.values[1] - df.index.values[0]
        dxdt = np.diff(df['x'].values) / dt
        dydt = np.diff(df['y'].values) / dt
        dsdt = np.sqrt(np.power(dxdt, 2) + np.power(dydt, 2))
        df['e1_x'] = Series(dxdt * dsdt, index=df.index.values[1:])
        df['e1_y'] = Series(dydt * dsdt, index=df.index.values[1:])
        return df.fillna(0)

    def _calc_e2(self, df):
        theta = np.pi / 2.0
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        e2 = R.dot(df.ix[:, 2:].values.T)
        df['e2_x'] = Series(e2[0], index=df.index.values)
        df['e2_y'] = Series(e2[1], index=df.index.values)
        return df

    def get_dataframe(self, start, end, num):
        ts = np.linspace(start, end, num)
        self._dt = ts[1] - ts[0]

        df = DataFrame(index=ts)
        df['x'] = Series(self._get_x(ts), index=ts)
        df['y'] = Series(self._get_y(ts), index=ts)

        df = self._calc_e1(df)
        df = self._calc_e2(df)

        return df

    def __call__(self, start, end, num=100):
        return self.get_dataframe(start, end, num)

if __name__ == '__main__':
    oval = Oval(2, 1)

    df = oval(0, 2 * np.pi)

    plt.plot(df['x'], df['y'])
    i = 20
    plt.plot([df['x'].values[i], df['x'].values[i] + df['e1_x'].values[i]], [df['y'].values[i], df['y'].values[i] + df['e1_y'].values[i]])
    plt.plot([df['x'].values[i], df['x'].values[i] + df['e2_x'].values[i]], [df['y'].values[i], df['y'].values[i] + df['e2_y'].values[i]])

    plt.show()