# import dune.vem
# import math, random
# from dune import create
# from dune.grid import cartesianDomain, gridFunction
# from dune.fem.plotting import plotPointData as plot
# from dune.fem.function import integrate, discreteFunction
# from dune.fem import parameter
# from dune.vem import voronoiCells, vemScheme, vemSpace, polyGrid
# import numpy

# from ufl import *
# import dune.ufl

import matplotlib
matplotlib.rc( 'image', cmap='jet' )
from matplotlib import pyplot
import math
from dune import create
from dune.grid import cartesianDomain, gridFunction
from dune.fem.plotting import plotPointData as plot
from dune.fem.function import integrate, discreteFunction
from dune.fem import parameter
from dune.vem import voronoiCells
from dune.fem.operator import linear as linearOperator
from scipy.sparse.linalg import spsolve
import numpy

import ufl.algorithms
from ufl import *
import dune.ufl

dune.fem.parameter.append({"fem.verboserank": -1})

parameters = {"newton.linear.tolerance": 1e-12,
              "newton.linear.preconditioning.method": "jacobi",
              "penalty": 40,  # for the dg schemes
              "newton.linear.verbose": False,
              "newton.verbose": False
              }

def runTest(exact, spaceConstructor, get_df):
    # Nval = 1
    # N_values = [2**i*Nval for i in range(4)]
    # results = []
    # for N in N_values:
        # N = 50
        # print('grid size N:', N)
        # set up grid for testing
        # constructor = cartesianDomain([0,0],[1,1],[N,N])
        # cells = voronoiCells(constructor,N*N,"voronoiseeds",load=True,show=False,lloyd=10)
        # grid = create.grid("agglomerate", cells, convex=True)

    grid = dune.vem.polyGrid( dune.vem.voronoiCells([[0,0],[1,1]], 50, lloyd=100) )

    # grid = dune.vem.polyGrid( dune.vem.voronoiCells([[-0.5,-0.5],[1,1]], 50, lloyd=100) )

    # get dimension of range
    dimRange = exact.ufl_shape[0]
    print('dim Range:', dimRange)

    # construct space to use
    space = spaceConstructor(grid,dimRange)

    # get sizes for eoc calculation
    sizes = [grid.hierarchicalGrid.agglomerate.size, space.size, *space.diameters()]

    laplace = lambda w: div(grad(w))
    H = lambda w: grad(grad(w))
    u = TrialFunction(space)
    v = TestFunction(space)

    a = ( inner(H(u[0]),H(v[0])) ) * dx
    b = ( laplace(laplace(exact[0]) ) )* v[0] * dx
    dbcs = [dune.ufl.DirichletBC(space, [0], i+1) for i in range(4)]

    scheme = dune.vem.vemScheme( [a==b, *dbcs], space, hessStabilization=1, solver="cg", parameters=parameters )
    df = discreteFunction(space, name="solution")
    info = scheme.solve(target=df)

    # solve
    # df = get_df(space)
    edf = exact-df
    errors = [ numpy.sqrt(e) for e in
            integrate(grid, [inner(edf,edf),inner(grad(edf),grad(edf))], order=5) ]
    print("bi-laplace errors:", errors )
    df.plot(gridLines=None)

    #     df.plot(level=3)
    #     plot(df-exact, grid=grid, gridLines=None, level=3)

    #     # calculate errors
    #     edf = exact-df
    #     err = [inner(edf,edf), inner(grad(edf),grad(edf))]
    #     errors = [ numpy.sqrt(e) for e in integrate(grid, err, order=8) ]
    #     # print(errors)

    #     results += [[sizes] + [errors]]

    # # calculate eocs
    # h = lambda sizes: sizes[3]
    # eoc = len(results[0][1])*[-1]
    # for level in range(0,len(N_values)):
    #     print(*results[level], *eoc)
    #     if level < len(N_values) -1:
    #         eoc = [math.log(results[level+1][1][j]/results[level][1][j])/
    #         math.log(h(results[level+1][0])/h(results[level][0]))
    #         for j in range(len(eoc))]

# def runTest(exact, spaceConstructor):
#     # interpolation test
#     # test( exact, spaceConstructor, lambda space: space.interpolate([0],name="solution") )

#     # elliptic test


#     test( exact, spaceConstructor, biharmonic )

def testElliptic(space):
    # set up scheme, df, and solve
    u = TrialFunction(space)
    v = TestFunction(space)
    x = SpatialCoordinate(space)

    massCoeff = 1+sin(dot(x,x))       # factor for mass term
    diffCoeff = 11-0.9*cos(dot(x,x))

    a = (diffCoeff*inner(grad(u),grad(v)) + massCoeff*dot(u,v) ) * dx

    # finally the right hand side and the boundary conditions
    b = (-div(diffCoeff*grad(exact[0])) + massCoeff*exact[0] ) * v[0] * dx
    dbc = [dune.ufl.DirichletBC(space, exact, i+1) for i in range(4)]

    df = space.interpolate([0],name="solution")
    scheme = dune.vem.vemScheme( [a==b, *dbc], space, solver="cg",
                            gradStabilization=diffCoeff,
                            massStabilization=massCoeff,
                            parameters=parameters )

    info = scheme.solve(target=df)

    return df

def biharmonic(space):
    H = lambda w: grad(grad(w))
    u = TrialFunction(space)
    v = TestFunction(space)

    a = ( inner(H(u[0]),H(v[0])) ) * dx
    b = ( laplace(laplace(exact[0]) )) * v[0] * dx
    dbcs = [dune.ufl.DirichletBC(space, [0], i+1) for i in range(4)]

    scheme = dune.vem.vemScheme( [a==b, *dbcs], space, hessStabilization=1, solver="cg", parameters=parameters )
    df = discreteFunction(space, name="solution")
    info = scheme.solve(target=df)

    return df

def main():
    # test with conforming second order VEM space
    order=3
    conformingSpaceConstructor = lambda grid, r: dune.vem.vemSpace( grid, order=order, dimRange=r, testSpaces = [-1,order-1,order-2] )
    # define exact solution
    x = ufl.SpatialCoordinate(ufl.triangle)
    exact = as_vector( [x[0]*x[1] * cos(pi*x[0]*x[1])] )

    # run tests for second order case
    # runTest(exact, conformingSpaceConstructor)

    # exact solution for biharmonic problem
    exact_biharmonic = as_vector( [sin(2*pi*x[0])**2*sin(2*pi*x[1])**2] )
    ncC1testSpaces = [ [0], [order-3,order-2], [order-4] ]
    nonconformingSpaceConstructor = lambda grid, r: dune.vem.vemSpace( grid, order=order, dimRange=r, storage="istl", testSpaces = ncC1testSpaces )

    # run tests for fourth order problems
    runTest(exact_biharmonic, nonconformingSpaceConstructor, biharmonic)

main()