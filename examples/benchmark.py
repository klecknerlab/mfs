import mfs
import numpy as np

k = 2*np.pi
scat = mfs.Scatter(N=1024, k=k, a=1/k, source_depth=0.5, Nq=8, verbose=True)
scat.incoming_planewaves([1, -1], [(0, 0, 1), (0, 0, -1)])
# scat.incoming_planewaves([1], [(0, 0, 1)])

dx = np.pi/k
X = [(0, 0, dx), (0, 0, -dx), (dx, 0, 0), (-dx, 0, 0)]


for i, use_numba in enumerate([True, False]):
    scat.use_numba = use_numba

    print(f'**** Numba {"Enabled" if use_numba else "Disabled"} ****')
    scat.solve(X)
    F = scat.force()
    print(f'  Fz = {F[:, 2]}')

    print()
