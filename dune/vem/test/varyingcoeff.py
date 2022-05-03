from dune import create
from dune.grid import cartesianDomain, gridFunction
from dune.fem.plotting import plotPointData as plot
from dune.fem.function import integrate, discreteFunction
from dune.fem import parameter
from dune.fem.operator import linear as linearOperator

import ufl.algorithms
from ufl import *
import dune.ufl

from script import runTest, checkEOC, interpolate
from interpolate import interpolate_fourthorder

dimR = 1

parameters = {"newton.linear.tolerance": 1e-12,
              "newton.linear.preconditioning.method": "jacobi",
              "penalty": 40,  # for the bbdg scheme
              "newton.linear.verbose": False,
              "newton.verbose": True
              }

def varyingcoeff(space, exact):
    laplace = lambda w: div(grad(w))
    H = lambda w: grad(grad(w))
    u = TrialFunction(space)
    v = TestFunction(space)
    x = SpatialCoordinate(space)

    # epsilon      = 1
    laplaceCoeff = 1
    mu           = 1

    kappa = 1./(1+dot(x,x))
    beta  = exp(-x[0]*x[1])
    gamma = sin(dot(x,x))**2

    a = ( kappa*inner(H(u[0]),H(v[0])) +\
          laplaceCoeff*beta*inner(grad(u),grad(v)) +\
          mu*gamma*inner(u,v)
        ) * dx

    # right hand side and the boundary conditions
    q = sum([ H(kappa*H(exact[0])[i,j])[i,j] for i in range(2) for j in range(2) ])
    b = ( q -\
          laplaceCoeff*div(beta*grad(exact[0])) +\
          mu*gamma*exact[0] ) * v[0] * dx
    dbc = [dune.ufl.DirichletBC(space, [0], i+1) for i in range(4)]

    biLaplaceCoeff = kappa
    diffCoeff      = laplaceCoeff*beta
    massCoeff      = mu*gamma

    scheme = dune.vem.vemScheme([a==b, *dbc], space,
                            solver="cg",
                            hessStabilization=biLaplaceCoeff,
                            gradStabilization=diffCoeff,
                            massStabilization=massCoeff,
                            parameters=parameters)

    df = discreteFunction(space, name="solution")
    info = scheme.solve(target=df)

    edf = exact-df
    err = [inner(edf,edf),
            inner(grad(edf),grad(edf)),
            inner(grad(grad(edf)),grad(grad(edf)))]

    return err

def runTestVaryingcoeff(testSpaces, order):
      x = SpatialCoordinate(triangle)
      exact = as_vector( dimR*[sin(2*pi*x[0])**2*sin(2*pi*x[1])**2] )
      spaceConstructor = lambda grid, r: dune.vem.vemSpace( grid,
                                                            order=order,
                                                            dimRange=r,
                                                            testSpaces=testSpaces )

      if (order==2):
        expected_eoc = [order, order, order-1]
      else:
        expected_eoc = [order+1, order, order-1]

      if (interpolate()):
            eoc = runTest(exact, spaceConstructor, interpolate_fourthorder)
      else:
            eoc = runTest(exact, spaceConstructor, varyingcoeff)

      return eoc, expected_eoc

def main():
      orders = [2,4]
      for order in orders:
            print("order: ", order)
            C1NCtestSpaces = [ [0], [order-3,order-2], [order-4] ]
            print("C1 non conforming test spaces: ", C1NCtestSpaces)

            eoc, expected_eoc = runTestVaryingcoeff( C1NCtestSpaces, order )

            checkEOC(eoc, expected_eoc)

main()