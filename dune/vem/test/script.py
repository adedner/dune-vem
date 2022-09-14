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

maxLevel = 3

def interpolate():
    return False

def getParameters():
    ln, lm, Lx, Ly = 1,0, 1,1
    return ln, lm, Lx, Ly

def runTest(exact, spaceConstructor, get_df):
    results = []
    for level in range(1,maxLevel):
        ln, lm, Lx, Ly = getParameters()
        # set up grid for testing
        N = 2**(level)
        grid = dune.vem.polyGrid(
          dune.vem.voronoiCells([[0,0],[Lx,Ly]], 13*N*N, lloyd=150, fileName="voronoiseeds", load=True)
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

def checkEOC(eoc, expected_eoc):
    i = 0
    for k in expected_eoc:
        assert(0.8*k <= eoc[i], "eoc out of expected range")
        if eoc[i] > 1.2*k:
            print("WARNING: the eoc seems too large (",eoc[i],"expected",k)
        i += 1
