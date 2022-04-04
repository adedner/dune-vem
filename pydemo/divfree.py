import matplotlib
matplotlib.rc( 'image', cmap='jet' )
from matplotlib import pyplot
import math
from dune import create
from dune.grid import cartesianDomain, gridFunction
from dune.fem.plotting import plotPointData as plot
from dune.fem.function import integrate, discreteFunction
from dune.fem import parameter
from dune.vem import voronoiCells, divFreeSpace, vemScheme
from dune.fem.operator import linear as linearOperator
from scipy.sparse.linalg import spsolve
from pprint import pprint

from ufl import *
import dune.ufl

maxLevel     = 4
order        = 2
level        = 5
constructor = cartesianDomain([0,0],[1,1],[2**level,2**level])
polyGrid = dune.vem.polyGrid( constructor, cubes=False )
space = divFreeSpace(polyGrid, order=order, storage="numpy")
x = SpatialCoordinate(space)
exact = [x[1],-x[0]]
uh = space.interpolate(exact,name="solution")
uh.plot()

errors = []
edf = exact - vh
err = [inner(edf,edf)]
errors += [ numpy.sqrt(e) for e in integrate(polyGrid,err,order=8) ]
print(errors, "#", space.diameters())
