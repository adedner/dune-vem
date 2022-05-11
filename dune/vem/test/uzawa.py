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

# this will be available in dune.common soon
# from dune.generator.threadedgenerator import ThreadedGenerator
from threadedgenerator import ThreadedGenerator
from dune.grid import cartesianDomain
from dune.alugrid import aluCubeGrid
from ufl import SpatialCoordinate, CellVolume, TrialFunction, TestFunction,\
                inner, dot, div, nabla_grad, grad, dx, as_vector, transpose, Identity, sym, \
                FacetNormal, ds, dS, avg, jump, CellVolume, FacetArea
from dune.ufl import Constant, DirichletBC
import dune.fem
from dune.fem.function import integrate, discreteFunction
from dune.fem.operator import linear as linearOperator

import dune.vem
from dune.vem import vemScheme as vemScheme
from dune.fem.operator import galerkin as galerkinOperator
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
        if u_h_n is not None:
            forcing += self.nu*u_h_n
            if explicit:
                forcing -= dot(u_h_n, nabla_grad(u_h_n))
            else:
                forcing -= ( dot(u_h_n, nabla_grad(u)) + dot(u, nabla_grad(u_h_n)) ) / 2
        epsilon = lambda u: sym(nabla_grad(u))
        mainModel   = ( self.nu*dot(u,v) - dot(forcing,v) + 2*self.mu*inner(epsilon(u), epsilon(v)) ) * dx
        gradModel   = -p*div(v) * dx
        divModel    = -div(u)*q * dx
        massModel   = p*q * dx
        preconModel = inner(grad(p),grad(q)) * dx

        generator = ThreadedGenerator(4)
        if not lagrange:
            generator.add( vemScheme, ( mainModel==0, *dbc_u ),
                                            gradStabilization=as_vector([self.mu*2,self.mu*2]),
                                            massStabilization=as_vector([self.nu,self.nu])
                         )
        else:
            generator.add( galerkinScheme, ( mainModel==0, *dbc_u ) )

        # gradOp    = galerkinOperator( gradModel, spcP,spcU)
        # divOp     = galerkinOperator( divModel, spcU,spcP)
        # massOp    = galerkinOperator( massModel, spcP)
        # self.A    = linearOperator(self.mainOp).as_numpy
        # self.G    = linearOperator(gradOp).as_numpy
        # self.D    = linearOperator(divOp).as_numpy
        # self.M    = linearOperator(massOp).as_numpy
        generator.add(galerkinOperator, gradModel, spcP,spcU)
        generator.add(galerkinOperator, divModel, spcU,spcP)
        generator.add(galerkinOperator, massModel, spcP)
        if precondition and tau is not None:
            if not lagrange:
                preconModel = dgLaplace(10, p,q, spcP, 0, 1)
                generator.add(galerkinOperator, preconModel, spcP)
            else:
                preconModel = inner(grad(p),grad(q)) * dx
                generator.add(galerkinOperator, (preconModel,DirichletBC(spcP,[0])), spcP)
        self.mainOp, gradOp, divOp, massOp, preconOp = generator.execute()

        generator.add(linearOperator,self.mainOp)
        generator.add(linearOperator,gradOp)
        generator.add(linearOperator,divOp)
        generator.add(linearOperator,massOp)
        if preconOp:
            generator.add(linearOperator,preconOp)
        else:
            generator.add(None)
        self.A, self.G, self.D, self.M, self.P = generator.execute()
        self.A, self.G, self.D, self.M = \
          self.A.as_numpy, self.G.as_numpy, self.D.as_numpy, self.M.as_numpy

        self.Ainv = lambda rhs: linalg.spsolve(self.A,rhs)
        self.Minv = lambda rhs: linalg.spsolve(self.M,rhs)

        if self.P:
            self.P = self.P.as_numpy
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

def main(level = 0):
    generator = ThreadedGenerator(4)
    Lx = 3
    Ly = 1
    N = 2**(level)
    muValue = 0.01
    tauValue = 1e-3
    order = 3
    # grid = dune.vem.polyGrid(cartesianDomain([0,0],[3,1],[30*N,10*N]),cubes=False)
    grid = dune.vem.polyGrid(
          dune.vem.voronoiCells([[0,0],[Lx,Ly]], 50*N*N, lloyd=200, fileName="voronoiseeds", load=True)
        #   cartesianDomain([0.,0.],[Lx,Ly],[N,N]), cubes=False
        #   cartesianDomain([0.,0.],[Lx,Ly],[2*N,2*N]), cubes=True
       )

    generator.add(dune.vem.divFreeSpace, grid, order=order)
    generator.add(dune.vem.bbdgSpace, grid)
    generator.add(dune.vem.vemSpace, grid, order=order, testSpaces=[-1,order-1,order-2])
    spcU, spcP, spcPsmooth = generator.execute()

    x       = SpatialCoordinate(spcU)
    exact_u = as_vector( [x[1] * (1.-x[1]), 0] ) # dy u_x = 1-2y, -dy^2 u_x = 2
    exact_p = (-2*x[0] + 3)*muValue              # dx p   = -2mu
    f       = as_vector( [0,]*grid.dimension )

    # discrete functions
    velocity = spcU.interpolate(spcU.dimRange*[0], name="velocity")
    pressure = spcP.interpolate(0, name="pressure")

    dbc = [ DirichletBC(velocity.space,exact_u,None) ]
    exact_uh = dune.fem.function.uflFunction(grid, order=2, name="exact_uh", ufl=exact_u)
    uzawa = Uzawa(grid, spcU, spcP, dbc, f,
                  muValue, tauValue, u_h_n=exact_uh,
                  tolerance=1e-12, precondition=True, verbose=False)
    uzawa.solve([velocity,pressure])

    p,q = TrialFunction(spcPsmooth), TestFunction(spcPsmooth)
    scheme = vemScheme( [ ( inner(grad(p),grad(q)) -
                            div( dot(velocity, nabla_grad(velocity)) - f)*q ) * dx == 0,
                        DirichletBC(spcPsmooth,exact_p) ],
                        gradStabilization=1, massStabilization=None,
                        parameters={"newton.linear.tolerance":1e-14}
                      )
    pressure = spcPsmooth.interpolate(0,name="pRecon")
    scheme.solve(pressure)
    average_p = Constant( pressure.integrate(), "aver_p" )

    edf = as_vector( [v1-v2 for v1,v2 in zip(velocity,exact_u)] +
                     [(pressure-average_p)-exact_p] )
    err = [ e*e for e in edf ]
    errors  = [ numpy.sqrt(e) for e in integrate(grid, err, order=8) ]

    # print(errors, average_p.value)
    # plot([velocity,pressure])
    return errors

if __name__ == "__main__":
    err1 = main(0)
    err2 = main(1)
    assert all( [ e<1e-9 for e in err1 ] )
    assert all( [ e<1e-9 for e in err2 ] )
