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
from interpolate import interpolate_secondorder

dimR = 1

parameters = {"newton.linear.tolerance": 1e-12,
              "newton.linear.preconditioning.method": "jacobi",
              "penalty": 40,  # for the bbdg scheme
              "newton.linear.verbose": False,
              "newton.verbose": True
              }

def hk(space, exact):
    laplace = lambda w: div(grad(w))
    H = lambda w: grad(grad(w))
    u = TrialFunction(space)
    v = TestFunction(space)

    diffCoeff = 1
    massCoeff = 1

    a = (diffCoeff*inner(grad(u),grad(v)) + massCoeff*dot(u,v) ) * dx
    b = sum( [ ( -div(diffCoeff*grad(exact[i])) + massCoeff*exact[i] ) * v[i]
             for i in range(dimR) ] ) * dx
    dbc = [dune.ufl.DirichletBC(space, exact, i+1) for i in range(4)]

    scheme = dune.vem.vemScheme([a==b, *dbc], space,
                            solver="cg",
                            gradStabilization=diffCoeff,
                            massStabilization=massCoeff,
                            parameters=parameters)

    df = discreteFunction(space, name="solution")
    info = scheme.solve(target=df)

    edf = exact-df
    err = [inner(edf,edf),
            inner(grad(edf),grad(edf))]

    return err

def runTesthk(testSpaces, order, vectorSpace, reduced):
      x = SpatialCoordinate(triangle)
      exact = as_vector( dimR*[x[0]*x[1] * cos(pi*x[0]*x[1])] )
      spaceConstructor = lambda grid, r: dune.vem.vemSpace( grid,
                                                            order=order,
                                                            dimRange=r,
                                                            testSpaces=testSpaces,
                                                            vectorSpace=vectorSpace,
                                                            reduced=reduced)

      expected_eoc = [order+1, order]
      if (interpolate()):
            eoc = runTest(exact, spaceConstructor, interpolate_secondorder)
      else:
            eoc = runTest(exact, spaceConstructor, hk)

      return eoc, expected_eoc

def main():
      orders = [1,3]
      for order in orders:
            print("order: ", order)
            # C0 non conforming VEM
            C0NCtestSpaces = [-1, order-1, order-2 ]
            print("C0 non conforming test spaces: ", C0NCtestSpaces)

            # vectorSpace=False, reduced=True
            eoc, expected_eoc = runTesthk( C0NCtestSpaces, order, vectorSpace=False, reduced=True )
            checkEOC(eoc, expected_eoc)

            # vectorSpace=True, reduced=True
            eoc, expected_eoc = runTesthk( C0NCtestSpaces, order, vectorSpace=True, reduced=True )
            checkEOC(eoc, expected_eoc)

main()