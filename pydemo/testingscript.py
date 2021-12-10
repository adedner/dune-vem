import dune.vem
import math, random
from dune import create
from dune.grid import cartesianDomain, gridFunction
from dune.fem.plotting import plotPointData as plot
from dune.fem.function import integrate, discreteFunction
from dune.fem import parameter
from dune.vem import voronoiCells, vemScheme, vemSpace, polyGrid

import ufl.algorithms
from ufl import *
import dune.ufl

dune.fem.parameter.append({"fem.verboserank": -1})

parameters = {"newton.linear.tolerance": 1e-12,
              "newton.linear.preconditioning.method": "jacobi",
              "penalty": 10*order*order,  # for the dg schemes
              "newton.linear.verbose": False,
              "newton.verbose": False
              }

def test(exact, spaceConstructor, get_df):
    Nval = 5
    N_values = [2**i*Nval for i in range(4)]
    results = []
    for N in N_values:
        # set up grid for testing
        constructor = cartesianDomain([0,0],[1,1],[N,N])
        cells = voronoiCells(constructor,50,"voronoiseeds",load=True,show=False,lloyd=10)
        grid = create.grid("agglomerate", cells, convex=True)

        # get dimension of range
        dimRange = exact.ufl_shape[0]

        # construct space to use
        space = spaceConstructor(grid,r)

        # get sizes for eoc calculation
        sizes = [grid.hierarchicalGrid.agglomerate.size, space.size, *space.diameters()]

        # solve
        df = get_df(space)
        df.plot(level=3)
        plot(df-exact, grid=polyGrid, gridLines=None, level=3)

        # calculate errors
        edf = exact-df
        err = [inner(edf,edf), inner(grad(edf),grad(edf))]
        errors = [ numpy.sqrt(e) for e in integrate(grid, err, order=8) ]
        print(errors)

        results += [ [sizes] + errors ]

    # calculate eocs
    h = lambda sizes: sizes[3]
    eoc = len(results[0][1])*[-1]
    for level in range(0,len(N_values)):
        print(*results[level], *eoc)
        if level < len(N_values) -1:
            eoc = [math.log(results[level+1][1][j]/results[level][1][j])/
            math.log(h(results[level+1][0])/h(results[level][0]))
            for j in range(len(eoc))]

def runTest(exact, spaceConstructor):
    # interpolation test
    test( exact, spaceConstructor, lambda space: space.interpolate([0],name="solution") )

    # elliptic test
    def testElliptic(space):
        # set up scheme, df, and solve

        u = TrialFunction(space)
        v = TestFunction(space)

        massCoeff = 1+sin(dot(x,x))       # factor for mass term
        diffCoeff = 11-0.9*cos(dot(x,x))

        a = (diffCoeff*inner(grad(u),grad(v)) + massCoeff*dot(u,v) ) * dx

        # finally the right hand side and the boundary conditions
        b = (-div(diffCoeff*grad(exact[0])) + massCoeff*exact[0] ) * v[0] * dx
        dbc = [dune.ufl.DirichletBC(space, exact, i+1) for i in range(4)]

        df = space.interpolate([0],name="solution")
        scheme = vemScheme( [a==b, *dbc], space, solver="cg",
                             gradStabilization=diffCoeff,
                             massStabilization=massCoeff,
                             parameters=parameters )


        info = scheme.solve(target=df)
        return df

    test( exact, spaceConstructor, testElliptic )

def main():
    # test with conforming second order VEM space
    conformingSpaceConstructor = lambda grid, r: vemSpace(grid, order=3, dimRange=r, testSpaces = [-1,order-1,order-2])
    # define exact solution
    x = ufl.Coordinate(ufl.triangle)
    exact = as_vector( [x[0]*x[1] * cos(pi*x[0]*x[1])] )

    runTest(exact, conformingSpaceConstructor)

main():