import mfs
import numpy as np
import matplotlib.pyplot as plt

k = 2*np.pi
scat = mfs.Scatter(N=256, k=k, a=0.5, source_depth=0.5, Nq=8, verbose=True)
scat.incoming_planewaves([1, -1], [(0, 0, 1), (0, 0, -1)])
# scat.incoming_planewaves([1], [(0, 0, 1)])

dx = 1
scat.solve([(0, 0, dx), (0, 0, -dx), (2*dx, 0, 0), (-2*dx, 0, 0)])

N = 200
lim = 4
points = 2*lim*(np.arange(N) + 0.5)/N - lim
X = np.zeros((N, N, 3))
X[:, :, 0] += points.reshape(1, -1)
X[:, :, 2] += points.reshape(-1, 1)
extent = [-lim, lim, -lim, lim]

F = scat.force()
# print(f'   F_z =', F[:, 0])

plt.imshow(scat.p2(X), extent=extent, origin='lower', clim=(-50, 50))

arrow_scale = 0.25

for XX, FF in zip(scat.X, F):
    plt.gca().add_artist(plt.Circle((XX[0], XX[2]), scat.a, color='w', ec='k', zorder=1))

    delta = arrow_scale * FF
    tail = XX - 0.5*delta

    plt.arrow(tail[0], tail[2], delta[0], delta[2], color='r', zorder=2, width=0.1, head_width=0.2, length_includes_head=True)

plt.ylabel('$z / \lambda$')
plt.xlabel('$x / \lambda$')

cbar = plt.colorbar()
cbar.set_label("$p_2 (a.u.)$")

plt.show()
