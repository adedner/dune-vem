from dune import create
from dune.grid import cartesianDomain, gridFunction
from dune.fem.plotting import plotPointData as plot
from dune.fem.function import integrate, discreteFunction
from dune.fem import parameter
from dune.fem.operator import linear as linearOperator

import ufl.algorithms
from ufl import *
import dune.ufl

from script import runTest, checkEOC
from interpolate import interpolate_fourthorder

parameters = {"newton.linear.tolerance": 1e-12,
              "newton.linear.preconditioning.method": "jacobi",
              "penalty": 40,  # for the bbdg scheme
              "newton.linear.verbose": False,
              "newton.verbose": True
              }

def biharmonic(space, exact):
    laplace = lambda w: div(grad(w))
    H = lambda w: grad(grad(w))
    u = TrialFunction(space)
    v = TestFunction(space)

    epsilon      = 1
    laplaceCoeff = 1
    mu           = 1

    a = ( epsilon*inner(H(u[0]),H(v[0])) +\
          laplaceCoeff*inner(grad(u),grad(v)) +\
          mu*inner(u,v)
        ) * dx
    b = ( epsilon*laplace(laplace(exact[0])) -\
          laplaceCoeff*laplace(exact[0]) +\
          mu*exact[0] ) * v[0] * dx
    dbc = [dune.ufl.DirichletBC(space, [0], i+1) for i in range(4)]

    biLaplaceCoeff = epsilon
    diffCoeff      = laplaceCoeff
    massCoeff      = mu

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

def runTestBiharmonic(testSpaces, order):
    x = SpatialCoordinate(triangle)
    exact = as_vector( [sin(2*pi*x[0])**2*sin(2*pi*x[1])**2] )
    spaceConstructor = lambda grid, r: dune.vem.vemSpace( grid,
                                                          order=order,
                                                          dimRange=r,
                                                          testSpaces=testSpaces )

    expected_eoc = [order+1, order, order-1]
    eoc_interpolation = runTest(exact, spaceConstructor, interpolate_fourthorder)
    eoc_solve = runTest(exact, spaceConstructor, biharmonic)

    return eoc_interpolation, eoc_solve, expected_eoc

def main():
      orderslist_fourthorder = [3]
      for order in orderslist_fourthorder:
            print("order: ", order)
            C1NCtestSpaces = [ [0], [order-3,order-2], [order-4] ]
            print("C1 non conforming test spaces: ", C1NCtestSpaces)
            eoc_interpolation, eoc_solve, expected_eoc = runTestBiharmonic( C1NCtestSpaces, order )

            checkEOC(eoc_interpolation, expected_eoc)

main()