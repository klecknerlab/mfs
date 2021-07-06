import numpy as np
import mfs
import matplotlib.pyplot as plt

π = np.pi

v0 = 343
omega = 40E3
rho = 1.225

λ = v0/omega
k = 2*π / λ

a = 4E-3


p1 = 3E3
phi0 = p1/(rho * omega)
Fscale = 1E3

scat = mfs.Scatter(N=256, k=k, a=a, rho=rho, source_depth=0.5, Nq=16, verbose=True)
scat.incoming_planewaves(np.array([0.5j, 0.5])*phi0, [(0, 0, 1), (0, 0, -1)])

d = np.sqrt(3)/2
spacing = 1 * λ
cluster = spacing * np.array([
    (-1, 0, 0),
    (0, 0, 0),
    (+1, 0, 0),
    (-0.5, d, 0),
    (+0.5, d, 0),
    (-0.5, -d, 0),
    (+0.5, -d, 0),
])

cluster[:, 0:2] = cluster[:, (1, 0)] # Rotate 90 degrees, force is higher!

X = np.empty((8, 3))
X[:7] = cluster
X[7] = (40E-3, 0, 0)

scat.solve(X[:8])

extents = [-20, 60, -20, 20]
spacing = 0.5
xscale = 1000


dx = (extents[1] - extents[0])
nx = int(dx // spacing)

dy = (extents[3] - extents[2])
ny = int(dy // spacing)

X = np.zeros((ny, nx, 3))

X[:, :, 0] += (((np.arange(nx) + 0.5)/nx) * dx + extents[0]).reshape(1, -1)
X[:, :, 1] += (((np.arange(ny) + 0.5)/ny) * dy + extents[2]).reshape(-1, 1)


F = scat.force()
print(F*1E3)


plt.imshow(scat.p2(X/xscale), extent=extents, origin='lower', clim=(-200, 200))

plt.colorbar().set_label(r'$\left<p_2\right>$ (Pa)')


for XX in scat.X:
    plt.gca().add_artist(plt.Circle((XX[0]*xscale, XX[1]*xscale), scat.a*xscale, color='w', ec='k', zorder=1))

plt.show()
