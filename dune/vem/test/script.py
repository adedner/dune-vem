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

from elliptic import elliptic
from perturbation import perturbation
from interpolate import interpolate
from hk import hk
from curlfree import curlfree
from divfree import divfree

dune.fem.parameter.append({"fem.verboserank": -1})

maxLevel = 4

parameters = {"newton.linear.tolerance": 1e-12,
              "newton.linear.preconditioning.method": "jacobi",
              "penalty": 40,  # for the dg schemes
              "newton.linear.verbose": False,
              "newton.verbose": False
              }

def runTest(exact, spaceConstructor, get_df):
    results = []
    for level in range(1,maxLevel):
        # set up grid for testing
        N = 2**(level)
        grid = dune.vem.polyGrid( dune.vem.voronoiCells([[-0.5,-0.5],[1,1]], 50, lloyd=100) )

        # get dimension of range
        dimRange = exact.ufl_shape[0]
        print('dim Range:', dimRange)

        res = []

        # construct space to use using spaceConstructor passed in
        space = spaceConstructor(grid,dimRange)

        err = get_df(space,exact)
        errors  = [ math.sqrt(e) for e in integrate(grid, err, order=8) ]
        length = len(errors)

        res += [ [[grid.hierarchicalGrid.agglomerate.size,space.size,*space.diameters()],*errors] ]
        results += [res]

    return calculateEOC(results,length)


def calculateEOC(results,length):
    eoc = length*[-1]

    for level in range(len(results)):
        print(*results[level][0],*eoc)
        if level<len(results)-1:
            eoc = [math.log(results[level+1][0][1+j]/results[level][0][1+j])/
                   math.log(results[level+1][0][0][3]/results[level][0][0][3])
                   for j in range(len(eoc))]

    return eoc


def runTestElliptic(testSpaces, order):
    x = SpatialCoordinate(triangle)
    exact = as_vector( [x[0]*x[1] * cos(pi*x[0]*x[1])] )
    spaceConstructor = lambda grid, r: dune.vem.vemSpace( grid,
                                                          order=order,
                                                          dimRange=r,
                                                          testSpaces=testSpaces )

    eoc = runTest(exact, spaceConstructor, elliptic)
    expected_eoc = [order]

    return eoc, expected_eoc

def runTestBiharmonic(testSpaces):
    x = SpatialCoordinate(triangle)
    exact = as_vector( [sin(2*pi*x[0])**2*sin(2*pi*x[1])**2] )
    spaceConstructor = lambda grid, r: dune.vem.vemSpace( grid, order=order, dimRange=r, testSpaces = testSpaces )

    eoc = runTest(exact, spaceConstructor, biharmonic)

    return eoc

def runTestPerturbation(testSpaces):
    x = ufl.SpatialCoordinate(ufl.triangle)
    exact = as_vector( [sin(2*pi*x[0])**2*sin(2*pi*x[1])**2] )
    spaceConstructor = lambda grid, r: dune.vem.vemSpace( grid, order=order, dimRange=r, testSpaces = testSpaces )

    # eoc = runTest(exact, spaceConstructor, interpolate)

    eoc = runTest(exact, spaceConstructor, perturbation)

    if order == 2:
        expected_eoc = [order, order, order-1, order-1]
    else:
        expected_eoc = [order+1, order, order-1, order-1]

    return eoc, expected_eoc

def runTesthk(testSpaces, vectorSpace, reduced, dimRange):
    x = SpatialCoordinate(triangle)

    exact = as_vector( dimRange*[x[0]*x[1] * cos(pi*x[0]*x[1])] )
    spaceConstructor = lambda grid, r: dune.vem.vemSpace( grid, order=order,
                                                          dimRange=r,
                                                          testSpaces=testSpaces,
                                                          vectorSpace=vectorSpace,
                                                          reduced=reduced)
    eoc = runTest(exact, spaceConstructor, hk)
    expected_eoc = [order+1,order]

    return eoc, expected_eoc

def runTestCurlfree():
    ln,lm, Lx,Ly = 1,0, 1,1.1
    x = SpatialCoordinate(triangle)

    exact = -grad( (cos(ln/Lx*pi*x[0])*cos(lm/Ly*pi*x[1])) )
    spaceConstructor = lambda grid, r: dune.vem.curlFreeSpace( grid, order=order )
    eoc = runTest(exact, spaceConstructor, curlfree)

    expected_eoc = [order+1,order+1]

    return eoc, expected_eoc

def runTestDivFree():
    x = SpatialCoordinate(triangle)

    exact = as_vector([-x[1]+x[0]**2*x[1],
                            x[0]-x[1]**2*x[0]])

    spaceConstructor = lambda grid, r: dune.vem.divFreeSpace( grid, order=order )
    # eoc = runTest(exact, spaceConstructor, interpolate)

    eoc = runTest(exact, spaceConstructor, divfree)
    expected_eoc = [order+1]

    return eoc, expected_eoc

def main():
    # test elliptic with conforming second order VEM space
    orderslist_secondorder = [3]
    # for order in orderslist_secondorder:
    #     # C0testSpaces = [0,order-2,order-2]

    #     # eoc, expected eoc = runTestElliptic( C0testSpaces )
    #     eoc, expected_eoc = runTestElliptic( C0NCtestSpaces, order )

    order = 3
    C0NCtestSpaces = [-1,order-1,order-2]
    eoc, expected_eoc = runTestElliptic( C0NCtestSpaces, order )

    # # test biharmonic with non conforming C1 space
    # C1NCtestSpaces = [ [0], [order-3,order-2], [order-4] ]
    # C1C0testSpaces = [ [0], [order-2,order-2], [order-4] ]

    # # eoc, expected_eoc = runTestPerturbation( C1C0testSpaces, False, False, dimRange=3 )

    # # eoc, expected_eoc = runTestCurlfree()

    # eoc, expected_eoc = runTestDivFree()

    i = 0
    for k in expected_eoc:
        assert(0.8*k <= eoc[i] <= 1.2*k), "eoc out of expected range"
        i += 1


main()