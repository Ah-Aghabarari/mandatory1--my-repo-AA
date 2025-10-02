import numpy as np
import sympy as sp
import scipy.sparse as sparse

x, y = sp.symbols('x,y')

class Poisson2D:
    
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), in [0, L]^2

    where L is the length of the domain in both x and y directions.
    Dirichlet boundary conditions are used for the entire boundary.
    The Dirichlet values depend on the chosen manufactured solution.

    """

    def __init__(self, L, ue):
        """Initialize Poisson solver for the method of manufactured solutions

        Parameters
        ----------
        L : number
            The length of the domain in both x and y directions
        ue : Sympy function
            The analytical solution used with the method of manufactured solutions.
            ue is used to compute the right hand side function f.
        """
        self.L = L
        self.ue = ue
        self.f = sp.diff(self.ue, x, 2)+sp.diff(self.ue, y, 2)

    def create_mesh(self, N):
        self.N=N
        self.h = self.L / self.N 
        x = np.linspace(0, self.L, self.N+1)
        y = np.linspace(0, self.L, self.N+1)
        self.xij, self.yij = np.meshgrid(x, y, indexing='ij')
        """Create 2D mesh and store in self.xij and self.yij"""
        # self.xij, self.yij ...
        #raise NotImplementedError

    def D2(self):
        D2 = sparse.diags([1, -2, 1], np.array([-1, 0, 1]), (self.N+1, self.N+1), 'lil')
        D2[0, :4] = 2, -5, 4, -1
        D2[-1, -4:] = -1, 4, -5, 2
        D2 *= (1/self.h**2)
        return D2
        """Return second order differentiation matrix"""
        #raise NotImplementedError

    def laplace(self):
        N = self.N
        D = self.D2()
        I = sparse.identity(N + 1, format="csr")
        L = sparse.kron(I, D, format="csr") + sparse.kron(D, I, format="csr")
        return L
        """Return vectorized Laplace operator"""
        #raise NotImplementedError

    def get_boundary_indices(self):
        id = []
        m = self.N + 1
        for i in range(m):
            for j in range(m):
                if i == 0 or i == self.N or j == 0 or j == self.N:
                    k = i * m + j  
                    id.append(k)
        return np.array(id)
        """Return indices of vectorized matrix that belongs to the boundary"""
        #raise NotImplementedError

    def assemble(self):
        self._ue = sp.lambdify((x, y), self.ue, "numpy")
        self._f = sp.lambdify((x, y), self.f, "numpy")
        A = self.laplace().tolil()


        b = self._f(self.xij, self.yij).astype(float).ravel()
        g = self._ue(self.xij, self.yij).astype(float).ravel()


        for k in self.get_boundary_indices():
            A.rows[k] = [k]
            A.data[k] = [1.0]
            b[k] = g[k]

        return A.tocsr(), b.reshape(self.N + 1, self.N + 1)
        """Return assembled matrix A and right hand side vector b"""
        # return A, b
        #raise NotImplementedError

    def l2_error(self, u):
        Ue = self._ue(self.xij, self.yij).astype(float)
        err = u - Ue
        return np.sqrt(np.sum(err ** 2) * self.h ** 2)
        """Return l2-error norm"""
        #raise NotImplementedError

    def __call__(self, N):
        """Solve Poisson's equation.

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction

        Returns
        -------
        The solution as a Numpy array

        """
        self.create_mesh(N)
        A, b = self.assemble()
        self.U = sparse.linalg.spsolve(A, b.flatten()).reshape((N+1, N+1))
        return self.U

    def convergence_rates(self, m=6):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretization levels to use

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
            u = self(N0)
            E.append(self.l2_error(u))
            h.append(self.h)
            N0 *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

    def eval(self, x, y):
        """Return u(x, y)

        Parameters
        ----------
        x, y : numbers
            The coordinates for evaluation

        Returns
        -------
        The value of u(x, y)

        """
        i = min(int(np.floor(x / self.h)), self.N - 1)
        j = min(int(np.floor(y / self.h)), self.N - 1)
        tx = (x - i * self.h) / self.h
        ty = (y - j * self.h) / self.h
        U = self.U

        u_left = U[i, j]
        u_right = U[i + 1, j]
        u_top_left = U[i, j + 1]
        u_top_right = U[i + 1, j + 1]

        return (1 - tx) * (1 - ty) * u_left + tx * (1 - ty) * u_right + (1 - tx) * ty * u_top_left + tx * ty * u_top_right
        #raise NotImplementedError

def test_convergence_poisson2d():
    # This exact solution is NOT zero on the entire boundary
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    r, E, h = sol.convergence_rates()
    assert abs(r[-1]-2) < 1e-2

def test_interpolation():
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    U = sol(100)
    assert abs(sol.eval(0.52, 0.63) - ue.subs({x: 0.52, y: 0.63}).n()) < 1e-3
    assert abs(sol.eval(sol.h/2, 1-sol.h/2) - ue.subs({x: sol.h/2, y: 1-sol.h/2}).n()) < 1e-3

