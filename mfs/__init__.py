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
        p[0] = 0.25*rho * (abs2(vx) + abs2(vy) + abs2(vz) - k*k*abs2(phi))


# Pretty time printing function, used for verbose output
def _tp(dt):
    if dt < 1E-3:
        return f'{dt*1e6:.1f} µs'
    elif dt < 1:
        return f'{dt*1e3:.1f} ms'
    else:
        return f'{dt*1e3:.1f} s'


class Scatter:
    # For reference, indices of various arrays:
    # X (particle locations): [Np, Nd=3]
    # c (solved scattering intensity): [Np, N]
    # quad_[normal/weights]: [Nquad, Nd=3,1]
    # bdy1 (boundary locations for a single particle): [N, Nd=3]
    # src1 (source locations for a single particle): [N, Nd=3]

    def __init__(self, k=1, a=1, N=512, Nq=8, rho=1, source_depth=0.5, verbose=False, use_numba=HAS_NUMBA, solver=np.linalg.solve):
        '''
        Initialize a MSF acoustic scattering simulation.

        Keywords
        --------
        k : float (default: 1)
            The wave vector amplitude
        a : float (default: 1)
            The particle size
        N : int (default: 512)
            The number of source points
        Nq : int (default: 8)
            The number of quadrature points in θ
        source_depth : float (default: 0.5)
            The depth of the source points in fractions of a radius
        verbose : bool (default: False)
            If true, print timing information as computations are performed
        use_numba : bool (default: True if numba installed)
            If true, uses parallel accelerated numba functions where available
        solver : function (default: numpy.linalg.solve)
            The linear algebra solver function used by `solve`.  Should be
            capable of solving a dense matrix linear equation, and take the
            same parameters as numpy.linalg.solve
        '''

        self.k = k
        self.rho = rho
        self.verbose = verbose
        self.use_numba = use_numba
        self.solver = solver

        # Define an incoming field function; the default throws an error
        # User needs to define an incoming wave before computing scattering!
        def dummy_inc(X, normal=None):
            raise ValueError('Incoming field undefined (call the "incoming_planewaves" method or equivalent first)')

        self._inc = dummy_inc

        self.build_normal(a, N, source_depth)
        self.build_quadrature(Nq)


    def build_normal(self, a=None, N=None, source_depth=None):
        '''Build the normal of points used to compute scattering.  Normally
        this does not need to called by the user, unless you are reconfiguring
        an existing simulation.

        Keywords
        --------
        a : float
            The particle radius. If not specified, use existing value.
        N : int
            The number of normal points. If not specified, use existing value.
        source_depth : float
            The relative depth of the source points. If not specified, use
            existing value.
        '''
        # Redefine variabes if needed.
        self.a = self.a if a is None else a
        self.N = self.N if N is None else N
        self.source_depth = self.source_depth if source_depth is None else source_depth

        # Build the Fibonacci normal
        self.normal = np.empty((self.N, 3))
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        self.normal[:,2] = (1 - 1/self.N) * (1 - 2*np.arange(self.N)/(self.N-1))
        ρ = np.sqrt(1.0 - self.normal[:,2]**2)
        θ = golden_angle * np.arange(self.N)
        self.normal[:,0] = ρ * np.cos(θ)
        self.normal[:,1] = ρ * np.sin(θ)

        # Create boundary and source point arrays
        # Incdices: [N, 3]
        self.bdy1 = self.a * self.normal
        self.src1 = (1-self.source_depth) * self.bdy1


    def build_quadrature(self, Nq=8):
        '''Build the quadrature points for computing forces.  Normally this
        does not need to called by the user (unless you want to change the
        quadrature of an existing sim).

        Parameters
        ----------
        Nq : int (default: 8)
            The number of quadrature divisions in θ.  Total number of quadrature
            points is 2 θ^2.
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
        A_inc = np.array(A_inc, dtype='complex')
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
        self.bdy = self.bdy1 + self.X.reshape(-1, 1, 3)
        self.src = self.src1 + self.X.reshape(-1, 1, 3)

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
            Velcity (complex first order fluctuation term)
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
            #  p2 = -[(κ/2) <p1>^2 - (ρ/2) <v1>^2)
            #    = (ρ/4) * (|v1|^2 - k^2 |ϕ1|^2)
            p2 = (0.25*self.rho) * (dot(v, v.conjugate()).real - (self.k**2) * abs(ϕ)**2)

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
        # Rehape prior to multiplication so we can sum over quad points for
        #   each particle.
        # Indices: [Np, Nd=3]
        return  self.a**2 * (p2.reshape(-1, self.Nquad, 1) * self.quad_wnormal).sum(1)
