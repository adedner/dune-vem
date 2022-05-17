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
                FacetNormal, ds, dS, avg, jump, CellVolume, FacetArea
from dune.ufl import Constant, DirichletBC
import dune.fem
from dune.fem.operator import linear as linearOperator

import dune.vem
from dune.fem.operator import galerkin as galerkinOperator
from dune.vem import vemScheme as vemScheme
from dune.fem.scheme import galerkin as galerkinScheme

def plot(target):
    fig = pyplot.figure(figsize=(10,10))
    target[0].plot(colorbar="vertical", figure=(fig, 211))
    target[1].plot(colorbar="vertical", figure=(fig, 212))
    pyplot.show()

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
                 mu, tau, u_h_n=None, explicit=False,
                 tolerance=1e-9, precondition=True, verbose=False,
                 lagrange=False):
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
        forcing = dot(forcing,v)
        if u_h_n is not None:
            forcing += dot(self.nu*u_h_n,v)
            b = lambda u1,u2,phi: dot( dot(u1, nabla_grad(u2)), phi)
            symb = lambda u1,u2,phi: ( b(u1,u2,phi) - b(u1,phi,u2) ) / 2
            if explicit:
                forcing -= b( u_h_n, u_h_n, v )
            else:
                # forcing -= b(u_h_n, u, v)
                # forcing -= b(u,u_h_n,v)
                # ustable forcing -= symb(u_h_n, u, v)
                # default
                forcing -= ( b(u_h_n, u, v) + b(u,u_h_n,v) ) / 2
                # unstable at outflow forcing -= ( symb(u_h_n, u, v) + symb(u,u_h_n,v) ) / 2
        epsilon = lambda u: sym(nabla_grad(u))
        mainModel   = ( self.nu*dot(u,v) - forcing + 2*self.mu*inner(epsilon(u), epsilon(v)) ) * dx
        gradModel   = -p*div(v) * dx
        divModel    = -div(u)*q * dx
        massModel   = p*q * dx
        preconModel = inner(grad(p),grad(q)) * dx

        if not lagrange:
            self.mainOp = vemScheme( ( mainModel==0, *dbc_u ),
                                       gradStabilization=as_vector([self.mu*2,self.mu*2]),
                                       massStabilization=as_vector([self.nu,self.nu])
                                     )
        else:
            self.mainOp = galerkinScheme( ( mainModel==0, *dbc_u ) )
        gradOp    = galerkinOperator( gradModel, spcP,spcU)
        divOp     = galerkinOperator( divModel, spcU,spcP)
        massOp    = galerkinOperator( massModel, spcP)

        self.mainLinOp = linearOperator(self.mainOp)
        self.G    = linearOperator(gradOp).as_numpy
        self.D    = linearOperator(divOp).as_numpy
        self.M    = linearOperator(massOp).as_numpy
        self.Minv = lambda rhs: linalg.spsolve(self.M,rhs)

        if precondition and self.mainOp.model.nu > 0:
            if not lagrange:
                preconModel = dgLaplace(10, p,q, spcP, 0, 1)
                preconOp    = galerkinOperator( preconModel, spcP)
            else:
                preconModel = inner(grad(p),grad(q)) * dx
                preconOp    = galerkinOperator( (preconModel,DirichletBC(spcP,0)), spcP)
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
        velocity = target[0]
        pressure = target[1]
        info = {"uzawa.outer.iterations":0,
                "uzawa.converged":False}
        # problem is linear but coefficients depend on previous time step so need to reassemble
        self.mainOp.jacobian(velocity, self.mainLinOp)
        A = self.mainLinOp.as_numpy
        Ainv = lambda rhs: linalg.spsolve(A,rhs)
        sol_u = velocity.as_numpy
        sol_p = pressure.as_numpy
        # right hand side for Shur complement problem
        velocity.clear()
        self.mainOp(velocity,self.rhsVelo)
        self.rhs_u *= -1
        self.xi[:]  = self.G*sol_p
        self.rhs_u -= self.xi
        self.mainOp.setConstraints(self.rhsVelo)
        sol_u[:]      = Ainv(self.rhs_u[:])
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
            self.xi[:] = Ainv(self.rhs_u[:])
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
    useVem   = False
    muValue  = 0.1
    tauValue = 1e-2
    order    = 2
    grid = dune.vem.polyGrid(cartesianDomain([0,0],[3,1],[30*4,10*4]),cubes=False)
    if useVem:
        spcU = dune.vem.divFreeSpace( grid, order=order)
        spcP = dune.fem.space.finiteVolume( grid )
        # spcP = dune.fem.space.lagrange( grid, order=1 )
    else:
        spcU = dune.fem.space.lagrange( grid, order=order, dimRange=2 )
        spcP = dune.fem.space.lagrange( grid, order=order-1 )

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
                  muValue, tauValue, u_h_n=None,
                  tolerance=1e-9, precondition=True, verbose=True, # tolerance 2e-5 (TH p=2)
                  lagrange=not useVem )
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
