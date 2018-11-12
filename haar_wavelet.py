https://kastnerkyle.github.io/posts/wavelets/



%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt


def haar_matrix(size):
    level = int(np.ceil(np.log2(size)))
    H = np.array([1.])[:, None]
    NC = 1. / np.sqrt(2.)
    LP = np.array([1., 1.])[:, None] 
    HP = np.array([1., -1.])[:, None]
    for i in range(level):
        H = NC * np.hstack((np.kron(H, LP), np.kron(np.eye(len(H)),HP)))
    H = H.T
    return H

def dwt(x):
    H = haar_matrix(x.shape[0])
    x = x.ravel()
    #Zero pad to next power of 2
    x = np.hstack((x, np.zeros(H.shape[1] - x.shape[0])))
    return np.dot(H, x)

def idwt(x):
    H = haar_matrix(x.shape[0])
    x = x.ravel()
    #Zero pad to next power of 2
    x = np.hstack((x, np.zeros(H.shape[0] - x.shape[0])))
    return np.dot(H.T, x)

def wthresh(a, thresh):
    #Soft threshold
    res = np.abs(a) - thresh
    return np.sign(a) * ((res > 0) * res)

rstate = np.random.RandomState(0)
s = pr + 2 * rstate.randn(*pr.shape)
threshold = t = 5
wt = dwt(s)
wt = wthresh(wt, t)
rs = idwt(wt)

plt.plot(s, color='steelblue')
plt.title('Noisy Signal')
plt.figure()
plt.plot(dwt(s), color='darkred')
plt.title('Wavelet Transform of Noisy Signal')
plt.figure()
plt.title('Soft Thresholded Transform Coefficients')
plt.plot(wt, color='darkred')
plt.figure()
plt.title('Reconstructed Signal after Thresholding')
plt.plot(rs, color='steelblue')