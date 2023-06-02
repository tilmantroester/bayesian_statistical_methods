import numpy as np

np.random.seed(346)

m_true, b_true = 1.3, -0.2

def model(x):
    return m_true*x + b_true

n = 20

sigma_y = 0.3

x = np.sort(np.random.uniform(0.1, 2.3, size=n))
y_err = 0.3*np.ones_like(x)
y = model(x) + y_err*np.random.normal(size=n)

np.savetxt(
    "data_0.txt",
    np.vstack((x, y, y_err)).T,
    header=f"m={m_true}, b={b_true}\nx y yerr"
)


m_true, b_true = -0.7, 1.2
f_true = 0.51

def model(x):
    return m_true*x + b_true

n = 30

x = np.sort(np.random.uniform(0.1, 2.3, size=n))
y_err = 0.05 + np.abs(0.2*np.random.normal(size=n))
y_scatter = (y_err + f_true*model(x)**2)*np.random.normal(size=n)
y = model(x) + y_scatter

np.savetxt(
    "data_1.txt",
    np.vstack((x, y, y_err)).T,
    header=f"m={m_true}, b={b_true}, f={f_true}\nx y yerr"
)