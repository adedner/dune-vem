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

from interpolate import interpolate_secondorder, interpolate_fourthorder
from elliptic import elliptic
from biharmonic import biharmonic
from perturbation import perturbation
from hk import hk
from curlfree import curlfree
from divfree import divfree

dune.fem.parameter.append({"fem.verboserank": -1})

maxLevel = 4

def runTest(exact, spaceConstructor, get_df):
    results = []
    for level in range(1,maxLevel):
        # set up grid for testing
        N = 2**(level)
        Lx,Ly = 1,1
        grid = dune.vem.polyGrid(
          dune.vem.voronoiCells([[0,0],[Lx,Ly]], 10*N*N, lloyd=200, fileName="voronoiseeds", load=True)
        #   cartesianDomain([0.,0.],[Lx,Ly],[N,N]), cubes=False
        #   cartesianDomain([0.,0.],[Lx,Ly],[2*N,2*N]), cubes=True
        )

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

    expected_eoc = [order+1,order]
    eoc_interpolation = runTest(exact, spaceConstructor, interpolate_secondorder)
    eoc_solve = runTest(exact, spaceConstructor, elliptic)

    return eoc_interpolation, eoc_solve, expected_eoc

def runTestBiharmonic(testSpaces, order):
    x = SpatialCoordinate(triangle)
    exact = as_vector( [sin(2*pi*x[0])**2*sin(2*pi*x[1])**2] )
    spaceConstructor = lambda grid, r: dune.vem.vemSpace( grid,
                                                          order=order,
                                                          dimRange=r,
                                                          testSpaces=testSpaces )

    expected_eoc = [order+1, order, order-1]
    eoc_interpolation = runTest(exact, spaceConstructor, biharmonic)

    return eoc_interpolation, expected_eoc

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

def runTestCurlfree(order):
    ln,lm, Lx,Ly = 1,0, 1,1.1
    x = SpatialCoordinate(triangle)

    exact = -grad( (cos(ln/Lx*pi*x[0])*cos(lm/Ly*pi*x[1])) )
    spaceConstructor = lambda grid, r: dune.vem.curlFreeSpace( grid,
                                                               order=order )
    eoc_interpolation = runTest(exact, spaceConstructor, curlfree)

    expected_eoc = [order+1,order+1]

    return eoc_interpolation, expected_eoc

def runTestDivFree(order):
    x = SpatialCoordinate(triangle)

    exact = as_vector([-x[1]+x[0]**2*x[1],
                            x[0]-x[1]**2*x[0]])

    spaceConstructor = lambda grid, r: dune.vem.divFreeSpace( grid, order=order )
    expected_eoc = [order+1]

    # eoc = runTest(exact, spaceConstructor, interpolate)
    eoc_interpolation = runTest(exact, spaceConstructor, divfree)

    return eoc_interpolation, expected_eoc

def checkEOC(eoc, expected_eoc):
    i = 0
    for k in expected_eoc:
        assert(0.8*k <= eoc[i] <= 1.2*k), "eoc out of expected range"
        i += 1

    return

def main():
    # test elliptic with conforming and nonconforming second order VEM space
    orderslist_secondorder = [2]
    # for order in orderslist_secondorder:
    #     print("order: ", order)
    #     C0NCtestSpaces = [-1,order-1,order-2]
    #     print("C0 non conforming test spaces: ", C0NCtestSpaces)
    #     eoc_interpolation, eoc_solve, expected_eoc = runTestElliptic( C0NCtestSpaces, order )

    #     checkEOC(eoc_interpolation, expected_eoc)
    #     checkEOC(eoc_solve, expected_eoc)

    #     C0testSpaces = [0,order-2,order-2]
    #     print("C0 test spaces: ", C0testSpaces)
    #     eoc_interpolation, eoc_solve, expected_eoc = runTestElliptic( C0testSpaces, order )

    #     checkEOC(eoc_interpolation, expected_eoc)
    #     checkEOC(eoc_solve, expected_eoc)

    # test biharmonic fourth order with nonconforming VEM space
    # orderslist_fourthorder = [3]
    # for order in orderslist_fourthorder:
    #     print("order: ", order)
    #     C1NCtestSpaces = [ [0], [order-3,order-2], [order-4] ]
    #     print("C1 non conforming test spaces: ", C1NCtestSpaces)
    #     eoc_interpolation, expected_eoc = runTestBiharmonic( C1NCtestSpaces, order )

    #     checkEOC(eoc_interpolation, expected_eoc)
    #     checkEOC(eoc_solve, expected_eoc)

    # test curl free interpolation
    for order in orderslist_secondorder:
        print("order: ", order)
        eoc_interpolation, expected_eoc = runTestCurlfree( order )
        checkEOC(eoc_interpolation, expected_eoc)

    #     eoc_interpolation, expected_eoc = runTestDivFree( order )
    #     checkEOC(eoc_interpolation, expected_eoc)

    # hk tests


main()