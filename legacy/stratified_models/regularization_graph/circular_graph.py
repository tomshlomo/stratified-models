# import numpy as np
# n = 7
# s = np.roll(np.eye(n), 1, axis=1)
# d = s - np.eye(n)
# l = d.T @ d
# t = 1
# p = np.linalg.inv(t * l + np.eye(n))
# print(p)
# nvec = np.arange(n//2 + 1)
# q = 1/(t*(2 - 2 * np.cos(2*np.pi*nvec/n)) + 1)
# print(q)
# print((np.fft.hfft(q, n=n, norm='forward')) / p[0] )
