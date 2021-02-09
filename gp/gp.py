#%%
import matplotlib.pyplot as plt
import numpy as np
#%%
mean = [0, 0]
cov = [[1, 0], [0, 100]]  # diagonal covariance
#%%
x, y = np.random.multivariate_normal(mean, cov, 5000).T
plt.plot(x, y, 'x')
plt.axis('equal')
plt.show()