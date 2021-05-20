import numpy as np

from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

x = np.array([[0.26, 0.75],[0.26, 0.75], [0.29, 0.72],[0.31, 0.73],[0.31, 0.73]])
y= np.array([[0.26, 0.27], [0.28, 0.36], [0.21, 0.35], [0.19, 0.45], [0.18, 0.48], [0.33, 0.36], [0.36, 0.49], [0.30, 0.49], [0.22, 0.61], 
             [0.22, 0.79], [0.25, 0.95], [0.30, 0.62], [0.29, 0.80], [0.26, 0.96], [0.25, 0.26], [0.28, 0.26], [0.24, 0.26], [0.30, 0.26]])

'''with open('out.txt') as f:
    for coordinates in f:
        print(coordinates)'''

distance, path = fastdtw(x, y, dist=euclidean)

print(distance)
print(path)

#f.close()