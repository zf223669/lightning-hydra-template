import numpy as np
from scipy.signal import savgol_filter
np.set_printoptions(precision=2)
x = np.array([2,2,5,2,1,0,1,4,9])

savgol_filter(x,5,2,mode='nearest')

