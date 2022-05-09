# %% [markdown]
# .. index:: Solvers; Saddle Point (Uzawa)
#
# # Saddle Point Solver (using Scipy)
# %%
import matplotlib
matplotlib.rc( 'image', cmap='jet' )
from matplotlib import pyplot
import numpy
from scipy.sparse import bmat, linalg
from dune.grid import cartesianDomain
from dune.alugrid import aluCubeGrid
from ufl import SpatialCoordinate, CellVolume, TrialFunction, TestFunction,\
                inner, dot, div, nabla_grad, grad, dx, as_vector, transpose, Identity, sym, \
                FacetNormal, ds, dS, avg, jump, CellVolume, FacetArea
from dune.ufl import Constant, DirichletBC
import dune.fem
from dune.fem.operator import linear as linearOperator

import dune.vem
# from dune.vem import vemOperator as galerkinOperator
from dune.fem.operator import galerkin as galerkinOperator
from dune.vem import vemScheme as galerkinScheme

def dgLaplace(beta, p,q, spc, p_bnd, dD):
    n             = FacetNormal(spc)
    he            = avg( CellVolume(spc) ) / FacetArea(spc)
    hbnd          = CellVolume(spc) / FacetArea(spc)
    aInternal     = dot(grad(p), grad(q)) * dx
    diffSkeleton  = beta/he*jump(p)*jump(q)*dS -\
                    dot(avg(grad(p)),n('+'))*jump(q)*dS -\
                    jump(p)*dot(avg(grad(q)),n('+'))*dS
    diffSkeleton -= ( dot(grad(p),n)*q*dD +\
                      p_bnd*dot(grad(q),n)*dD ) * ds
    if p_bnd is not None:
        diffSkeleton += beta/hbnd*(p-p_bnd)*dD*q*ds
    return aInternal + diffSkeleton
class Uzawa:
    def __init__(self, grid, spcU, spcP, dbc_u, forcing,
                 mu, tau, u_h_n=None,
                 tolerance=1e-9, precondition=True, verbose=False):
        self.dimension = grid.dimension
        self.verbose = verbose
        self.tolerance2 = tolerance**2
        self.dbc_u = dbc_u
        self.mu = Constant(mu, "mu")
        if tau is not None:
            assert tau>0
            self.nu  = Constant(1/tau, "nu")
        else:
            self.nu  = Constant(0, "nu")
        u = TrialFunction(spcU)
        v = TestFunction(spcU)
        p = TrialFunction(spcP)
        q = TestFunction(spcP)
        def epsilon(u):
            return sym(nabla_grad(u))
        if u_h_n is not None:
            forcing += self.nu*u_h_n
            forcing -= dot(u_h_n, nabla_grad(u))/2
            forcing -= dot(u, nabla_grad(u_h_n))/2
        mainModel   = ( self.nu*dot(u,v) - dot(forcing,v) + 2*self.mu*inner(epsilon(u), epsilon(v)) ) * dx
        gradModel   = -p*div(v) * dx
        divModel    = -div(u)*q * dx
        massModel   = p*q * dx
        preconModel = inner(grad(p),grad(q)) * dx

        self.mainOp = galerkinScheme( ( mainModel==0, *dbc_u ),
                                        gradStabilization=as_vector([self.mu*2,self.mu*2]),
                                        massStabilization=as_vector([self.nu,self.nu])
                                    )
        gradOp    = galerkinOperator( gradModel, spcP,spcU)
        divOp     = galerkinOperator( divModel, spcU,spcP)
        massOp    = galerkinOperator( massModel, spcP)

        self.A    = linearOperator(self.mainOp).as_numpy
        self.Ainv = lambda rhs: linalg.spsolve(self.A,rhs)
        self.G    = linearOperator(gradOp).as_numpy
        self.D    = linearOperator(divOp).as_numpy
        self.M    = linearOperator(massOp).as_numpy
        self.Minv = lambda rhs: linalg.spsolve(self.M,rhs)

        if precondition and self.mainOp.model.nu > 0:
            preconModel = dgLaplace(10, p,q, spcP, 0, 1)
            preconOp    = galerkinOperator( preconModel, spcP)
            self.P    = linearOperator(preconOp).as_numpy
            self.Pinv = lambda rhs: linalg.spsolve(self.P,rhs)
        else:
            self.Pinv = None

        self.rhsVelo  = spcU.interpolate(spcU.dimRange*[0], name="vel")
        self.rhsPress = spcP.interpolate(0, name="pres")
        self.rhs_u  = self.rhsVelo.as_numpy
        self.rhs_p  = self.rhsPress.as_numpy
        self.r      = numpy.copy(self.rhs_p)
        self.d      = numpy.copy(self.rhs_p)
        self.precon = numpy.copy(self.rhs_p)
        self.xi     = numpy.copy(self.rhs_u)

    def solve(self,target):
        info = {"uzawa.outer.iterations":0,
                "uzawa.converged":False}
        self.A    = linearOperator(self.mainOp).as_numpy
        self.Ainv = lambda rhs: linalg.spsolve(self.A,rhs)
        velocity = target[0]
        pressure = target[1]
        sol_u = velocity.as_numpy
        sol_p = pressure.as_numpy
        # right hand side for Shur complement problem
        velocity.clear()
        self.mainOp(velocity,self.rhsVelo)
        self.rhs_u *= -1
        self.xi[:]  = self.G*sol_p
        self.rhs_u -= self.xi
        self.mainOp.setConstraints(self.rhsVelo)
        sol_u[:]      = self.Ainv(self.rhs_u[:])
        self.rhs_p[:] = self.D*sol_u
        self.r[:]     = self.Minv(self.rhs_p[:])
        if self.Pinv:
            self.precon.fill(0)
            self.precon[:] = self.Pinv(self.rhs_p[:])
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
            self.mainOp.setConstraints([0,]*self.dimension, self.rhsVelo)
            self.xi[:] = self.Ainv(self.rhs_u[:])
            self.rhs_p[:] = self.D*self.xi
            rho = delta / numpy.dot(self.d,self.rhs_p)
            sol_p += rho*self.d
            sol_u -= rho*self.xi
            self.rhs_p[:] = self.D*sol_u
            self.r[:] = self.Minv(self.rhs_p[:])
            if self.Pinv:
                self.precon.fill(0)
                self.precon[:] = self.Pinv(self.rhs_p[:])
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

def main():
    muValue = 0.1
    tauValue = 1e-2
    order = 4
    grid = dune.vem.polyGrid(cartesianDomain([0,0],[3,1],[30,10]),cubes=True)
    spcU = dune.vem.divFreeSpace( grid, order=order)
    # spcP = dune.vem.bbdgSpace( grid, order=0)
    spcP = dune.fem.space.finiteVolume( grid )
    # spcP = dune.fem.space.lagrange( grid, order=1 )

    x       = SpatialCoordinate(spcU)
    exact_u = as_vector( [x[1] * (1.-x[1]), 0] ) # dy u_x = 1-2y, -dy^2 u_x = 2
    exact_p = (-2*x[0] + 2)*muValue              # dx p   = -2mu
    f       = as_vector( [0,]*grid.dimension )
    f      += exact_u/tauValue

    # discrete functions
    velocity = spcU.interpolate(spcU.dimRange*[0], name="velocity")
    pressure = spcP.interpolate(0, name="pressure")

    dbc = [ DirichletBC(velocity.space,exact_u,None) ]
    uzawa = Uzawa(grid, spcU, spcP, dbc, f,
                  muValue, tauValue,
                  tolerance=1e-9, precondition=True, verbose=True)
    uzawa.solve([velocity,pressure])
    print("Retry")
    uzawa.solve([velocity,pressure])

    fig = pyplot.figure(figsize=(10,10))
    velocity.plot(colorbar="vertical", figure=(fig, 211))
    pressure.plot(colorbar="vertical", figure=(fig, 212))
    pyplot.show()
    fig = pyplot.figure(figsize=(10,10))
    dune.fem.plotting.plotPointData(grad(velocity[0])[0], grid=grid,colorbar="vertical", figure=(fig, 211))
    dune.fem.plotting.plotPointData(grad(velocity[0])[1], grid=grid,colorbar="vertical", figure=(fig, 212))
    pyplot.show()

if __name__ == "__main__":
    main()
