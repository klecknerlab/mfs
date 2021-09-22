#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.special as sp
import time

try:
    import numba
    HAS_NUMBA = True

    real = numba.float64
    comp = numba.complex128 # complex is a built in type!

except:
    import warnings
    warnings.warn('Numba not installed; calculations will be slower!')
    HAS_NUMBA = False

# This code uses lots of indexing tricks to handle dot products, sums over
#  various indices, etc.
# The array indices are noted in the code to make what is happening more clear.
# Note that following conventions:
#   - "Nd" is the number of field dimensions -- 1 for scalars, 3 for vectors
#   - "N" is the number of source points per particle
#   - "Nq" is the number of quadratude points in theta
#   - "Nquad" is the total number of quadrature points
#   - "Np" is the number of particles
#   - "Ns" = N*Np is the total number of scatterers


# Some convenience functions
def mag(X):
    return np.sqrt((np.asarray(X)**2).sum(-1))

def dot(X, Y):
    return (np.asarray(X)*Y).sum(-1)

def mag1(X):
    return np.sqrt((np.asarray(X)**2).sum(-1))[..., np.newaxis]

def dot1(X, Y):
    return (np.asarray(X)*Y).sum(-1)[..., np.newaxis]

def norm(X):
    return X / mag1(X)

# Contact Force Function (Simple Power law scaling of the contact between two particles)
def PLC(F_0,X,alpha,n,a):
    d = 2*a
    return 5*F_0 *((alpha - X/d)/(alpha-1))**n

# Numba provides JIT compiled python functions.  These are written as
#   generlized vector functions, which means they are automatically evaluated
#   over an array of points in parallel!
if HAS_NUMBA:
    @numba.guvectorize(["f8[:], f8[:], f8[:], f8, c16[:]"], '(Nd), (Nd), (Nd), () -> ()', target='parallel')
    def numba_scattering_matrix(src, bdy, normal, k, A):
        # For Numba to give maximum acceleration you need to break out the dot
        #   product; sum is quite a bit slower!
        dx = bdy[0] - src[0]
        dy = bdy[1] - src[1]
        dz = bdy[2] - src[2]
        r2 = dx*dx + dy*dy + dz*dz
        r = np.sqrt(r2)
        G = np.exp(1j * k * r) / (4 * np.pi * r)

        dp = (dx*normal[0] + dy*normal[1] + dz*normal[2]) / r

        A[0] = (1j*k - 1/r) * G * dp

    @numba.njit
    def abs2(X):
        return X.real**2 + X.imag**2

    # We need to explicitly sum over the scatterers, so Ns is included in the
    #   vectorization.
    @numba.guvectorize(["f8[:, :], c16[:], c16[:], c16[:], f8[:], f8, f8, f8[:]"], '(Ns, Nd), (Ns), (N1), (Nd), (Nd), (), () -> ()', target='parallel')
    def numba_p2(src, c, inc, grad_inc, X, rho, k, p):
        phi = inc[0]
        vx = grad_inc[0]
        vy = grad_inc[1]
        vz = grad_inc[2]

        for i in range(src.shape[0]):
            dx = X[0] - src[i, 0]
            dy = X[1] - src[i, 1]
            dz = X[2] - src[i, 2]
            r2 = dx*dx + dy*dy + dz*dz
            r = np.sqrt(r2)

            G = c[i] * np.exp(1j*k*r) / (4*np.pi*r)

            phi += G

            dG = G * (1j*k - 1/r) / r
            vx += dx * dG
            vy += dy * dG
            vz += dz * dG

        # p[0] = 0.25*rho * (np.abs(vx)**2 + np.abs(vy)**2 + np.abs(vz)**2 - (k*np.abs(phi))**2)
        p[0] = 0.25*rho * (k*k*abs2(phi) - (abs2(vx) + abs2(vy) + abs2(vz)))


# Pretty time printing function, used for verbose output
def _tp(dt):
    if dt < 2E-3:
        return f'{dt*1e6:5f} µs'
    elif dt < 2:
        return f'{dt*1e3:5f} ms'
    else:
        return f'{dt:5f} s'


class Scatter:
    # For reference, indices of various arrays:
    # X (particle locations): [Np, Nd=3]
    # c (solved scattering intensity): [Np, N]
    # quad_[normal/weights]: [Nquad, Nd=3,1]
    # bdy1 (boundary locations for a single particle): [N, Nd=3]
    # src1 (source locations for a single particle): [N, Nd=3]

    def __init__(self, k=1, a=1, N=492, Nq=8, rho=1, phi_a=1, source_depth=0.5,
        lattice_type="icos", verbose=False, use_numba=HAS_NUMBA,
        solver=np.linalg.solve):
        '''
        Initialize a MFS acoustic scattering simulation.

        Keywords
        --------
        k : float (default: 1)
            The wave vector amplitude
        a : float (default: 1)
            The particle size
        N : int (default: 512)
            The number of source points.  For icosahedral or cubic lattices,
            the actual number of points will be slightly different (if you
            need the exact value, this is stored in the `N` attribute after
            initialization)
        Nq : int (default: 8)
            The number of quadrature points in θ
        phi_a : float (default: 1)
            The amplitude of the incoming velocity potential field
        source_depth : float (default: 0.5)
            The depth of the source points in fractions of a radius
        lattice_type : str (default: 'icos')
            The type of lattice used to construct the source/boundary points.
            Valid options are "fib", "cub", and "icos"
        verbose : bool (default: False)
            If true, print timing information as computations are performed
        use_numba : bool (default: True if numba installed)
            If true, uses parallel accelerated numba functions where available
        solver : function (default: numpy.linalg.solve)
            The linear algebra solver function used by `solve`.  Should be
            capable of solving a dense matrix linear equation, and take the
            same parameters as numpy.linalg.solve
        '''

        self.a = a
        self.k = k
        self.rho = rho
        self.phi_a = phi_a
        self.verbose = verbose
        self.use_numba = use_numba
        self.solver = solver

        # Define an incoming field function; the default throws an error
        # User needs to define an incoming wave before computing scattering!
        def dummy_inc(X, normal=None):
            raise ValueError('Incoming field undefined (call the "incoming_planewaves" method or equivalent first)')

        self._inc = dummy_inc

        self.N = N
        self.source_depth = source_depth

        lt = lattice_type.lower()
        if lt.startswith('fib'):
            self.build_fib_normal()
        elif lt.startswith('cub'):
            self.build_cube_normal()
        elif lt.startswith('icos'):
            self.build_icosahedral_normal()
        else:
            raise ValueError(f'Invalid lattice type: {lattice_type}')

        self.build_quadrature(Nq)


    def build_fib_normal(self):
        '''Build the normal of points used to compute scattering.  Normally
        this does not need to called by the user, unless you are reconfiguring
        an existing simulation.
        '''
        # Build the Fibonacci normal
        self.normal = np.empty((self.N, 3))
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        self.normal[:,2] = (1 - 1/self.N) * (1 - 2*np.arange(self.N)/(self.N-1))
        ρ = np.sqrt(1.0 - self.normal[:,2]**2)
        θ = golden_angle * np.arange(self.N)
        self.normal[:,0] = ρ * np.cos(θ)
        self.normal[:,1] = ρ * np.sin(θ)

        # Create boundary and source point arrays
        # Indices: [N, 3]
        self.bdy1 = self.normal
        self.src1 = (1-self.source_depth) * self.bdy1


    def build_cube_normal(self):
        '''
        Build the normal of points used to compute scattering.  Normally
        this does not need to called by the user, unless you are reconfiguring
        an existing simulation.

        Note: calling this function may change the number of source points, as
        it needs to be given by 6n², where n is the number of points per
        cubic face edge.
        '''
        # Enforce N to be 6n²
        n = int((self.N / 6)**.5 + 0.5) # Find closest points per edge
        N = 6 * n**2
        if N != self.N:
            print(f'Warning: specified number of points ({self.N}) not valid for cube lattice.\nUsed closest value: N={N}')
            self.N = N

        # Build a lattice on a cube surface
        # Build one edge first: the centers of a grid from -1 to 1 with n² points
        edge = (2*np.mgrid[:n, :n].T.reshape(-1, 2) + 1 - n) / n
        X = np.empty((n**2, 3))
        X[:, :2] = edge
        X[:, 2] = 1
        X = np.vstack([X, X * (1, 1, -1)]) # Mirror about z=0 plane
        X = np.vstack([X, np.roll(X, 1, axis=1), np.roll(X, 2, axis=1)]) # Permute the axes
        self.normal = norm(X)

        # Create boundary and source point arrays
        # Indices: [N, 3]
        self.bdy1 = self.normal
        self.src1 = (1-self.source_depth) * self.bdy1


    def build_icosahedral_normal(self):
        '''
        Build the normal of points used to compute scattering.  Normally
        this does not need to called by the user, unless you are reconfiguring
        an existing simulation.

        Note: calling this function may change the number of source points, as
        it needs to be given by 20n² - 10n + 12, where n is the number of
        points per icosahedral face edge.  The exact sizes that are possible
        are: N = 12, 42, 92, 162, 252, 362, 492, 642, 812, 1002, 1212, 1442...
        '''
        # Compute the number of points per edge
        n_edge = 1 + int(((self.N - 2) / 10)**.5 + 0.5)
        N = 10 * n_edge**2 - 20 * n_edge + 12
        if N != self.N:
            print(f'Warning: specified number of points ({self.N}) not valid for icosahedral lattice.\nUsed closest value: N={N}')
            self.N = N

        # Construct all the corners of a unit icosahedron
        # https://en.wikipedia.org/wiki/Regular_icosahedron#Cartesian_coordinates
        ϕ = (1 + 5**.5) / 2
        d1 = (1 + ϕ**2)**-0.5
        d2 = ϕ * d1
        X = np.array([(0, d1, d2), (0, -d1, d2), (0, d1, -d2), (0, -d1, -d2)])
        X = np.vstack([X, np.roll(X, -1, axis=-1), np.roll(X, -2, axis=-1)])

        # The corners of the faces of an icosahedron, determined by hand
        tris = [
            (0, 1, 8), (0,10, 1), (0, 8, 4), (0, 4, 5), (0, 5,10),
            (1,10, 7), (1, 7, 6), (1, 6, 8), (2, 3,11), (2,11, 5),
            (2, 5, 4), (2, 4, 9), (2, 9, 3), (3, 9, 6), (3, 6, 7),
            (3, 7,11), (4, 8, 9), (5,11,10), (6, 9, 8), (7,10,11),
        ]

        # Fractional displacement vectors along each edge for the splitting
        DV = np.array(np.triu_indices(n_edge)).T
        DV[:, 1] = (n_edge - 1) - DV[:, 1]
        DV = (DV / (n_edge - 1)).reshape(-1, 2, 1)

        # Build the points one face at a time
        Z = None
        for i1, i2, i3 in tris:
            V = np.vstack([X[i2] - X[i1], X[i3] - X[i1]])
            Y = X[i1] + (DV * V).sum(1)
            # If this isn't the first face, make sure we aren't repeating points
            if Z is not None:
                rmin = mag(Z - Y.reshape(-1, 1, 3)).min(1)
                Z = np.vstack([Z, Y[np.where(rmin > 0.5/n_edge)]])
            else:
                Z = Y

        self.normal = norm(Z)
        self.bdy1 = self.normal
        if len(self.bdy1) != self.N:
            raise ValueError(f'Error in computing icosahedral points; should have been {self.N} points total, but got {len(self.bdy1)}!\nThis should never happen, but it did!\nDid you request an absurd number of points?!')
        self.src1 = (1-self.source_depth) * self.bdy1


    def build_quadrature(self, Nq=8):
        '''Build the quadrature points for computing forces.  Normally this
        does not need to called by the user (unless you want to change the
        quadrature of an existing sim).

        Parameters
        ----------
        Nq : int (default: 8)
            The number of quadrature divisions in θ.  Total number of quadrature
            points is 2 Nq^2.
        '''
        # Indices for matrix construction is: [Nq, Nϕ, Nd]
        Nϕ = 2 * Nq
        self.Nquad = Nq * Nϕ

        # Compute the Gauss-Legendre quadrature rule points and weights
        z_normal, w = np.polynomial.legendre.leggauss(Nq)
        z_normal = z_normal.reshape(-1, 1)
        w = w.reshape(-1, 1)

        # Angles in ϕ
        ϕ = 2*np.pi / Nϕ * np.arange(Nϕ).reshape(1, -1)

        # Construct normal vectors for integration
        self.quad_normal = np.zeros((Nq, Nϕ, 3))
        self.quad_normal[..., 2] = z_normal
        ρ_normal = np.sqrt(1 - self.quad_normal[..., 2]**2)
        self.quad_normal[..., 0] = ρ_normal * np.cos(ϕ)
        self.quad_normal[..., 1] = ρ_normal * np.sin(ϕ)

        # Construct weights, adding a dimension for ϕ
        self.quad_weight = np.zeros((Nq, Nϕ, 1))
        self.quad_weight[..., 0] = (np.pi / Nq * w)

        # Reshape the quadrature points
        # Final indices: [Nquad, Nd]
        self.quad_normal = self.quad_normal.reshape(-1, 3)
        self.quad_weight = self.quad_weight.reshape(-1, 1)
        self.quad_wnormal = self.quad_normal * self.quad_weight


    def incoming_planewaves(self, A_inc, k_inc):
        '''Define the incoming field as a superposition of planewaves.

        The field is defined as: ϕinc = ∑ A_inc e^(i k_inc·X)

        Parameters
        ----------
        A_inc : 1D array
            The incoming plane wave amplitudes, can be complex
        k_inc : 2D array
            The incoming plane wave directions.  Note: this array will get
            normalized, as all waves must have the same magnitude of k
        '''
        A_inc = self.phi_a * np.array(A_inc, dtype='complex')
        k_inc = self.k * norm(np.array(k_inc))

        # Dynamically define a function for the incoming field.  The defaults
        #   are used to capture the values of these variables.
        # If gradient is requested, return that as well.
        def inc(X, grad=True, A_inc=A_inc, k_inc=k_inc, sim=self):
            f = np.zeros((X.shape[:-1] + (1,)), dtype='complex')
            if grad:
                g = np.zeros(X.shape, dtype='complex')

            # Iterate over incoming planewaves
            for A, k in zip(A_inc, k_inc):
                ik = 1j * k
                ff = A * np.exp(dot1(ik, X))
                f += ff

                if grad:
                    g += ik * ff

            if grad:
                return f, g
            else:
                return f

        self._inc = inc
        
    def incoming_gaussian_waves(self, A_inc, k_inc, sigma):
        '''Define the incoming field as a superposition of planewaves.

        The field is defined as: ϕinc = ∑ A_inc e^(i k_inc·X) * e^(-s^2 / sigma^2)
            where s is the cylindrical radius off the z axis. Note: this gaussian
            profile does not take into account beam divergence.

        Parameters
        ----------
        A_inc : 1D array
            The incoming plane wave amplitudes, can be complex
        k_inc : 2D array
            The incoming plane wave directions.  Note: this array will get
            normalized, as all waves must have the same magnitude of k
        '''
        A_inc = self.phi_a * np.array(A_inc, dtype='complex')
        k_inc = self.k * norm(np.array(k_inc))

        # Dynamically define a function for the incoming field.  The defaults
        #   are used to capture the values of these variables.
        # If gradient is requested, return that as well.
        def inc(X, grad=True, A_inc=A_inc, k_inc=k_inc, sim=self):
            f = np.zeros((X.shape[:-1] + (1,)), dtype='complex')
            if grad:
                g = np.zeros(X.shape, dtype='complex')

            #define cylindrical coordinates
            #s = np.sqrt((dot1((1,0,0),X)/mag1(X))**2 + (dot1((0,1,0),X)/mag1(X))**2)
            sv = np.zeros_like(X)
            sv[:,:,:2] = X[:,:,:2]
            s = np.sqrt((sv**2).sum(-1).sum(-1))
            
            # Iterate over incoming planewaves
            for A, k in zip(A_inc, k_inc):
                ik = 1j * k
                ff = A * np.exp(dot1(ik, X)) * np.exp(-s**2/sigma**2)
                f += ff

                if grad:
                    g += ik * ff
                    g -= (ff * 2 * sv / sigma**2)

            if grad:
                return f, g
            else:
                return f

        self._inc = inc


    def _G(self, Δ, grad=True):
        '''Compute the Green's function and it's gradient (if requested).
        Usually not called by the user.

        Parameters
        ----------
        Δ : numpy array with shape [..., 3]
            Displacement vectors from source to field location

        Keywords
        --------
        grad : bool (default: True)
            If True, return the gradient of the Green's function.

        Returns
        -------
        G : numpy array with shape [..., 1]
        grad_G : numpy array with shape [..., 3]
            (optional)
        '''

        r = mag1(Δ)
        G = np.exp(1j * self.k * r) / (4 * np.pi * r)

        if grad:
            return G, (1j*self.k - 1/r) * G * (Δ/r)
        else:
            return G

    # Internal functions used for printing timing info
    def _tick(self, msg=None):
        if self.verbose:
            self.start = time.time()
            if msg:
                print(msg)

    def _tock(self, msg='tock'):
        if self.verbose:
            t = time.time()
            print(f'{msg+":":>35s} {_tp(t-self.start)}')
            self.start = t


    def solve(self, X, solver=np.linalg.solve):
        '''
        Solve for the scattering coefficients of the source points.

        Parameters
        ----------
        X : 2D array (shape: [Np, 3])
            The positions of each of the spheres.
        '''
        self.Np = len(X)
        self.Ns = self.Np * self.N

        # Check if particle input is sensible.
        self.X = np.array(X)
        if len(self.X.shape) != 2 or self.X.shape[1] != 3:
            raise ValueError('X should be a 2D array, or equivalent, with shape [N, 3]')

        self._tick(f'=============== Solving for {self.Np} particles ===============')

        # Make the bdy and src arrays for the particles at the given locations
        # Array indices are right aligned, so the bdy1/src1 arrays are
        #   are applied to be the same for all particles (particle # = index 0)
        # Indices: [Np, N, Nd=3]
        self.bdy = self.a * self.bdy1 + self.X.reshape(-1, 1, 3)
        self.src = self.a * self.src1 + self.X.reshape(-1, 1, 3)

        # Incoming field at the bdy points
        # Indices: [Np, N, Nd]
        inc, grad_inc = self._inc(self.bdy)

        if self.use_numba:
            # Build scattering matrix in Numba
            A = numba_scattering_matrix(
                    self.src.reshape(1, 1, -1, 3),
                    self.bdy.reshape(-1, self.N, 1, 3),
                    self.normal.reshape(1, -1, 1, 3),
                    self.k
                ).reshape(self.Ns, self.Ns)
        else:
            # Compute the displacement vectors, radius, and angles
            # Indices: [Np*N, Np*N, Nd]
            Δ = self.bdy.reshape(-1, 1, 3) - self.src.reshape(1, -1, 3)
            G, grad_G = self._G(Δ)

            # Matrix elements are normal vectors dotted with the incoming
            #   field gradient
            A = (self.normal.reshape(1, self.N, 1, 3) * grad_G.reshape(self.Np, self.N, self.Ns, 3)).sum(-1).reshape(self.Ns, self.Ns)

        # The gradient of the field from the scatters must cancel the incoming
        # field
        b = -(self.normal * grad_inc).sum(-1).reshape(-1)

        self._tock(f'Matrix Building ({self.Ns} x {self.Ns})')

        # Final result has indices: [Np, N]
        self.c = self.solver(A, b).reshape(-1, self.N)

        self._tock(f'Linear Solver ({self.Ns} x {self.Ns})')

        return self.c


    def fields(self, X):
        '''Return the total fields at given location for a pre-solved system.

        Note: `solve` method must be called first!

        Parameters
        ----------
        X : numpy array with shape [..., 3]
            Locations at which to find fields (assumed to be outside spheres!)

        Returns
        -------
        ϕ1 : array with shape [...]
            Velocity potential (complex first order fluctuation term)
        v1 : array with shape [..., 3]
            Velocity (complex first order fluctuation term)
        '''

        # Check if we have solved for the scattering ampltudes
        if not hasattr(self, 'c'):
            raise ValueError('you must call the "solve" method before computing fields')

        # Reshape scattering intensity to: [Np*N, Nd=1]
        c_flat = self.c.reshape(-1, 1)

        # Compute Green's function and it's gradient
        # Indices: [NX=total number of points in X, Np*N, Nd]
        Δ = X.reshape(-1, 1, 3) - self.src.reshape(1, -1, 3)
        G, grad_G = self._G(Δ)

        # Sum over source points.
        # Result has indices: [NX, Nd]
        inc, grad_inc = self._inc(X)
        ϕ = (G * c_flat).sum(1).reshape(X.shape[:-1]) + inc.reshape(X.shape[:-1])
        v = (grad_G * c_flat).sum(1).reshape(X.shape) + grad_inc

        return ϕ, v


    def p2(self, X):
        '''Compute the time averaged second order pressure.

        Note: `solve` method must be called first!

        Parameters
        ----------
        X : numpy array with shape [..., 3]
            Locations at which to find fields (assumed to be outside spheres!)

        Returns
        -------
        p2 : numpy array with shape [...]
        '''

        self._tick()

        if self.use_numba:
            inc, grad_inc = self._inc(X)
            p2 = numba_p2(self.src.reshape(-1, 3), self.c.reshape(-1), inc, grad_inc, X, self.rho, self.k)

        else:
            ϕ, v = self.fields(X)

            # Force density is:
            #  p2 = (κ/2) <p1>² - (ρ/2) <v1>²
            #     = (ρ/4) (k² |ϕ1|² - |v1|²)
            p2 = (0.25*self.rho) * ((self.k**2) * abs(ϕ)**2 - dot(v, v.conjugate()).real)

        self._tock(f'Calculate p2 ({np.prod(self.c.shape)} -> {np.prod(X.shape[:-1])})')

        return p2


    def force(self):
        '''Compute the per particle force for a pre-solved system.

        Note: `solve` method must be called first!

        Returns
        -------
        F : array with shape (Np, 3)
            The force on each particle.
        '''

        # Define new boundary points at quadrature locations
        # Indices: [Np, Nquad, Nd=3]
        bdy = self.a * self.quad_normal + self.X.reshape(-1, 1, 3)

        # Get second order time averaged pressure on surface
        p2 = self.p2(bdy)

        # Result is pressure on boundary times surface normal, integrated over
        #   quadrature points with weighted normals
        # Reshape prior to multiplication so we can sum over quad points for
        #   each particle.
        # Indices: [Np, Nd=3]
        return  -self.a**2 * (p2.reshape(-1, self.Nquad, 1) * self.quad_wnormal).sum(1)
    
    def energy(self):
        '''Compute the per particle force for a pre-solved system.

        Note: `solve` method must be called first!

        Returns
        -------
        F : array with shape (Np, 3)
            The force on each particle.
        '''

        # Define new boundary points at quadrature locations
        # Indices: [Np, Nquad, Nd=3]
        bdy = self.a * self.quad_normal + self.X.reshape(-1, 1, 3)

        # Get second order time averaged pressure on surface
        p2 = self.p2(bdy)

        # Result is pressure on boundary times surface normal, integrated over
        #   quadrature points with weighted normals
        # Reshape prior to multiplication so we can sum over quad points for
        #   each particle.
        # Indices: [Np, Nd=3]
        return  (self.a**3 / 3) * (p2.reshape(-1, self.Nquad, 1) * self.quad_weight).sum(1)

    
    def contact(self, alpha=1.025, n=4):
        '''Compute the per particle contact force for a pre-solved system
        
        Note: `solve` method must be called first!
        
        Returns
        -------
        F: array with shape (Np, 3)
            The contact force on each particle.
        '''
        #Reference acoustic force, used to compute the strength of the contact force
        F_0 = np.pi * self.rho * self.phi_a**2 * self.k**2 * self.a**2
        # Dx: Cartesian separation Matrix, R: Radial separation matrix, 
        Dx = self.X.reshape(-1,1,3) - self.X.reshape(1,-1,3)
        R = mag(Dx)
        
        # Initialize force and direction matrices with zeros to preserve shape of final contact matrix
        F = np.zeros_like(R, dtype='float')
        rhat = np.zeros_like(Dx, dtype='float')
        
        # Inside: Indices where radial separation distance falls within the cutoff distance of the WCA potential, but must be greater than zero
        inside = np.where((R<2*self.a*alpha)*(R>self.a*1e-6))
        rhat[inside] += norm(Dx[inside])
        F[inside] += PLC(F_0,R[inside],alpha,n,self.a)
        
        # Reshaped to match shape of acoustical force matrix
        return (F[:,:,np.newaxis]*rhat).sum(1)
        
