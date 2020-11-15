import mfs
import numpy as np

k = 1
scat = mfs.Scatter(N=512, k=k, a=1, source_depth=0.5, Nq=16, verbose=True)
scat.incoming_planewaves([1, -1], [(0, 0, 1), (0, 0, -1)])
# scat.incoming_planewaves([1], [(0, 0, 1)])

dx = 3/k
X = [(0, 0, dx), (0, 0, -dx), (dx, 0, 0), (-dx, 0, 0)]

# print('Quadrature sum:', (scat.quad_weight).sum())
# print('Quadrature sum:', mfs.dot(scat.quad_wnormal, scat.quad_normal).sum())

for i, use_numba in enumerate([True, False]):
    scat.use_numba = use_numba

    print(f'**** Numba {"Enabled" if use_numba else "Disabled"} ****')
    scat.solve(X)
    F = scat.force()
    print(f'  Fz = {F[:, 2]}')

    print()
