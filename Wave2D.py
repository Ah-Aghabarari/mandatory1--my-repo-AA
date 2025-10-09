import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm

x, y, t = sp.symbols('x,y,t')

class Wave2D:

    def create_mesh(self, N, sparse=False):
        self.N = int(N)
        self.L = 1.0
        self.h = self.L / self.N
        x1d = np.linspace(0.0, self.L, self.N + 1)
        y1d = np.linspace(0.0, self.L, self.N + 1)
        self.xij, self.yij = np.meshgrid(x1d, y1d, indexing="ij")

        """Create 2D mesh and store in self.xij and self.yij"""
        # self.xji, self.yij = ...
        #raise NotImplementedError

    def D2(self, N):
        m = self.N + 1
        D2 = sparse.diags([1, -2, 1], [-1, 0, 1], (m, m), 'lil')
        D2[0, :]  = 0.0
        D2[-1, :] = 0.0
        return (D2.tocsr())/(self.h**2)
        """Return second order differentiation matrix"""
        #raise NotImplementedError

    @property
    def w(self):
        return self.c * np.pi * np.sqrt(self.mx**2 + self.my**2)

        """Return the dispersion coefficient"""
        #raise NotImplementedError

    def ue(self, mx, my):
        return sp.sin(mx*sp.pi*x) * sp.sin(my*sp.pi*y) * sp.cos(self.w*t)

        """Return the exact standing wave"""
        #return sp.sin(mx*sp.pi*x)*sp.sin(my*sp.pi*y)*sp.cos(self.w*t)

    def initialize(self, N, mx, my):
        U0 = np.sin(mx * np.pi * self.xij) * np.sin(my * np.pi * self.yij)
        self.temp = U0.copy()
        self.apply_bcs()
        self.U = self.temp
        self.Um1 = None 
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        #raise NotImplementedError

    @property
    def dt(self):
        return self._dt
        """Return the time step"""
        #raise NotImplementedError

    def l2_error(self, u, t0):
        ue_fun = sp.lambdify((x, y, t), self.ue(self.mx, self.my), "numpy")
        Ue = ue_fun(self.xij, self.yij, t0)
        return np.sqrt((self.h**2) * np.sum((u - Ue) ** 2))

        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        #raise NotImplementedError

    def apply_bcs(self):
        U = self.temp
        U[0, :] = U[-1, :] = 0.0
        U[:, 0] = U[:, -1] = 0.0

        #raise NotImplementedError

    def __call__(self, N, Nt, cfl=0.5, c=1.0, mx=3, my=3, store_data=-1):
        self.mx, self.my, self.c = int(mx), int(my), float(c)
        self.create_mesh(N)
        self._dt = float(cfl) * self.h / self.c
        Nt = int(Nt)
        m = self.N + 1

        D = self.D2(self.N)
        I = sparse.identity(m, format="csr")
        L = sparse.kron(I, D, format="csr") + sparse.kron(D, I, format="csr")

        self.initialize(N, mx, my)

        u0 = self.U.reshape(m * m, order="C")
        Lu0 = L @ u0
        Um1 = self.U + 0.5 * (self.c * self.dt) ** 2 * Lu0.reshape((m, m), order="C")
        self.temp = Um1
        self.apply_bcs()
        self.Um1 = self.temp

        errs = [self.l2_error(self.U, 0.0)]
        out = None
        if store_data is not None and store_data > 0:
            out = {}
            out[0.0] = self.U.copy() 
        cdt = (self.c * self.dt) ** 2

        for n in range(1, Nt + 1):
            un = self.U.reshape(m * m, order="C")
            Lun = L @ un
            Unp1 = 2.0 * self.U - self.Um1 + cdt * Lun.reshape((m, m), order="C")

            self.temp = Unp1
            self.apply_bcs()
            Unp1 = self.temp

            self.Um1, self.U = self.U, Unp1

            tn = n * self.dt
            errs.append(self.l2_error(self.U, tn))
            if out is not None and (n % store_data == 0):
                out[tn] = self.U.copy()

        if out is not None:
            return out
        return self.h, np.array(errs)
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        #raise NotImplementedError

    def convergence_rates(self, m=4, cfl=0.1, Nt=10, mx=3, my=3):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for m in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

class Wave2D_Neumann(Wave2D):

    
    def D2(self, N):
        m = self.N + 1
        D2 = sparse.diags([1, -2, 1], [-1, 0, 1], (m, m), 'lil')
        D2[0, :]  = 0.0
        D2[0, 0]  = -2.0
        D2[0, 1]  =  2.0
        D2[-1, :] = 0.0
        D2[-1,-1] = -2.0
        D2[-1,-2] =  2.0
        return (D2.tocsr())/(self.h**2)
        #raise NotImplementedError

    def ue(self, mx, my):
        return sp.cos(mx*sp.pi*x) * sp.cos(my*sp.pi*y) * sp.cos(self.w*t)
        #raise NotImplementedError

    def initialize(self, N, mx, my):
        U0 = np.cos(mx * np.pi * self.xij) * np.cos(my * np.pi * self.yij)
        self.U = U0.copy()
        self.Um1 = None
        
    def apply_bcs(self):
        return
        #raise NotImplementedError

def test_convergence_wave2d():
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 1e-2

def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 0.05

def test_exact_wave2d():
    C = 1.0 / np.sqrt(2.0)
    mx = my = 2
    N = 64
    Nt = 128

    # Dirichlet
    sol = Wave2D()
    h, E = sol(N, Nt, cfl=C, c=1.0, mx=mx, my=my, store_data=-1)
    assert np.max(np.abs(E)) < 1e-12

    # Neumann
    solN = Wave2D_Neumann()
    h, E = solN(N, Nt, cfl=C, c=1.0, mx=mx, my=my, store_data=-1)
    assert np.max(np.abs(E)) < 1e-12
    #raise NotImplementedError
