# %% [markdown]
# .. index:: Solvers; Saddle Point (Uzawa)
#
# # Saddle Point Solver (using Scipy)
# %%
import matplotlib
matplotlib.rc( 'image', cmap='jet' )
from matplotlib import pyplot
import numpy
from scipy.sparse import linalg
from dune.grid import cartesianDomain
from dune.alugrid import aluCubeGrid
from ufl import SpatialCoordinate, CellVolume, TrialFunction, TestFunction,\
                inner, dot, div, nabla_grad, grad, dx, as_vector, transpose, Identity, sym, \
                FacetNormal, ds, dS, avg, jump, CellVolume, FacetArea, conditional
from dune.ufl import Constant, DirichletBC
import dune.fem

import dune.vem
from dune.fem.operator import galerkin as galerkinOperator
from dune.vem import vemScheme as vemScheme
from dune.fem.scheme import galerkin as galerkinScheme

def plot(target):
    fig = pyplot.figure(figsize=(10,10))
    target[0].plot(colorbar="vertical", figure=(fig, 211))
    target[1].plot(colorbar="vertical", figure=(fig, 212))
    pyplot.show()

def dgLaplace(beta, p,q, spc, p_bnd, precondDbnd):
    n             = FacetNormal(spc)
    he            = avg( CellVolume(spc) ) / FacetArea(spc)
    hbnd          = CellVolume(spc) / FacetArea(spc)
    aInternal     = dot(grad(p), grad(q)) * dx
    diffSkeleton  = beta/he*jump(p)*jump(q)*dS -\
                    dot(avg(grad(p)),n('+'))*jump(q)*dS -\
                    jump(p)*dot(avg(grad(q)),n('+'))*dS
    if type(precondDbnd) == bool:
        diffSkeleton  += ( beta/hbnd * (p-p_bnd) -
                           dot(grad(p),n) ) * q * ds
    else:
        for cond in precondDbnd:
            diffSkeleton  += ( beta/hbnd * (p-p_bnd) -
                               dot(grad(p),n) ) * q * conditional(cond,1,0) * ds
    return aInternal + diffSkeleton
class Uzawa:
    def __init__(self, grid,
                 spcU, spcP, dbc_u,
                 mainModel, gradModel, divModel,
                 mu, nu,
                 tolerance=1e-9, verbose=False,
                 lagrange=False, precondDbnd=None):
        self.dimension = grid.dimension
        self.verbose = verbose
        self.tolerance2 = tolerance**2
        self.dbc_u = dbc_u
        self.mu = mu
        self.nu = nu
        p = TrialFunction(spcP)
        q = TestFunction(spcP)
        massModel   = p*q * dx
        preconModel = inner(grad(p),grad(q)) * dx

        if not lagrange:
            self.mainOp = vemScheme( ( mainModel==0, *dbc_u ),
                                       gradStabilization=as_vector([self.mu*2,self.mu*2]),
                                       massStabilization=as_vector([self.nu,self.nu])
                                     )
        else:
            self.mainOp = galerkinScheme( ( mainModel==0, *dbc_u ) )
        gradOp    = galerkinOperator( [gradModel, *dbc_u] )
        divOp     = galerkinOperator( divModel )
        massOp    = galerkinOperator( massModel )

        self.mainLinOp = self.mainOp.linear()
        self.G    = gradOp.linear().as_numpy
        self.D    = divOp.linear().as_numpy
        self.M    = massOp.linear().as_numpy
        self.Minv = lambda rhs: linalg.spsolve(self.M,rhs)

        if self.mainOp.model.nu > 0:
            print("adding outer preconditioner",
                  "using nu=",self.mainOp.model.nu,
                  "and mu=",self.mainOp.model.mu,flush=True)
            if not lagrange:
                o = spcP.order
                preconModel = dgLaplace(10*(o+1)**2, p,q, spcP, 0, precondDbnd)
                preconOp    = galerkinOperator(preconModel, spcP)
            else:
                x = SpatialCoordinate(spcP)
                preconModel = inner(grad(p),grad(q)) * dx
                if type(precondDbnd) == bool:
                    pdbcs = [ DirichletBC(spcP,0) ]
                else:
                    pdbcs = [ DirichletBC(spcP,0,cond) for cond in precondDbnd ]
                preconOp    = galerkinOperator([ preconModel, *pdbcs ])
            self.P    = preconOp.linear().as_numpy.tocsc()
            self.Pinv = linalg.splu(self.P)
        else:
            self.Pinv = None
            print("No outer preconditioner",
                  "with nu=",self.mainOp.model.nu,
                  "and mu=",self.mainOp.model.mu,flush=True)

        self.rhsVelo  = spcU.interpolate(spcU.dimRange*[0], name="vel")
        self.rhsPress = spcP.interpolate(0, name="pres")
        self.rhs_u  = self.rhsVelo.as_numpy
        self.rhs_p  = self.rhsPress.as_numpy
        self.r      = numpy.copy(self.rhs_p)
        self.d      = numpy.copy(self.rhs_p)
        self.precon = numpy.copy(self.rhs_p)
        self.xi     = numpy.copy(self.rhs_u)

    def solve(self,target):
        velocity = target[0]
        pressure = target[1]
        info = {"uzawa.outer.iterations":0,
                "uzawa.converged":False}
        # problem is linear but coefficients depend on previous time step so need to reassemble
        self.mainOp.jacobian(velocity, self.mainLinOp)
        A = self.mainLinOp.as_numpy.tocsc()
        Ainv = linalg.splu(A)
        sol_u = velocity.as_numpy
        sol_p = pressure.as_numpy
        # right hand side for Shur complement problem
        velocity.clear()
        self.mainOp(velocity,self.rhsVelo)
        self.rhs_u *= -1
        self.xi[:]  = self.G*sol_p
        self.rhs_u -= self.xi
        # self.mainOp.setConstraints(self.rhsVelo)
        sol_u[:]      = Ainv.solve(self.rhs_u[:])
        self.rhs_p[:] = self.D*sol_u
        self.r[:]     = self.Minv(self.rhs_p[:])
        if self.Pinv:
            self.precon.fill(0)
            self.precon[:] = self.Pinv.solve(self.rhs_p[:])
            self.r *= self.mu.value
            self.r += self.nu.value*self.precon
        self.d[:] = self.r[:]
        delta = numpy.dot(self.r,self.rhs_p)
        if self.verbose:
            print(0,delta,self.tolerance2)
        # cg type iteration
        for m in range(1,1000):
            self.xi.fill(0)
            self.rhs_u[:] = self.G*self.d
            # self.mainOp.setConstraints([0,]*self.dimension, self.rhsVelo)
            self.xi[:] = Ainv.solve(self.rhs_u[:])
            self.rhs_p[:] = self.D*self.xi
            rho = delta / numpy.dot(self.d,self.rhs_p)
            sol_p += rho*self.d
            sol_u -= rho*self.xi
            self.rhs_p[:] = self.D*sol_u
            self.r[:] = self.Minv(self.rhs_p[:])
            if self.Pinv:
                self.precon.fill(0)
                self.precon[:] = self.Pinv.solve(self.rhs_p[:])
                self.r *= self.mu.value
                self.r += self.nu.value*self.precon
            oldDelta = delta
            delta = numpy.dot(self.r,self.rhs_p)
            if self.verbose:
                print(m,delta,self.tolerance2)
            if delta < self.tolerance2:
                info["uzawa.converged"] = True
                break
            gamma = delta/oldDelta
            self.d *= gamma
            self.d += self.r
        info["uzawa.outer.iterations"] = m
        return info
