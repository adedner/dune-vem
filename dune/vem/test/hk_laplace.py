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

dimR = 3

parameters = {"newton.linear.tolerance": 1e-12,
              "newton.linear.preconditioning.method": "jacobi",
              "penalty": 40,  # for the bbdg scheme
              "newton.linear.verbose": False,
              "newton.verbose": True
              }

def hk(space, exact):
    u      = TrialFunction(space)
    v      = TestFunction(space)
    normal = FacetNormal(space)

    diffCoeff = 0.1
    massCoeff = 2

    a = (diffCoeff*inner(grad(u),grad(v)) + massCoeff*dot(u,v) ) * dx
    b = sum( [ ( -div(diffCoeff*grad(exact[i])) + massCoeff*exact[i] ) * v[i]
             for i in range(dimR) ] ) * dx +\
        sum( [ diffCoeff*( dot(grad(exact[i]),normal) ) * v[i]
             for i in range(dimR) ] ) * ds

    scheme = dune.vem.vemScheme([a==b], space,
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
    exact = as_vector( dimR*[x[0]**dimR*x[1] * cos(pi*x[0]*x[1]**dimR)] )
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
        # C0 conforming VEM
        C0testSpaces = [0,order-2,order-2]
        print("C0 conforming test spaces: ", C0testSpaces)

        # vectorSpace=False, reduced=True
        eoc, expected_eoc = runTesthk( C0testSpaces, order, vectorSpace=False, reduced=True )
        checkEOC(eoc, expected_eoc)

main()
