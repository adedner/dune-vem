import dune.vem
import matplotlib
matplotlib.rc( 'image', cmap='jet' )
from matplotlib import pyplot
import math, numpy
from dune.grid import cartesianDomain, gridFunction
from dune.fem.plotting import plotPointData as plot
from dune.fem.function import integrate, discreteFunction
import dune.fem

from ufl import *
import dune.ufl

dune.fem.parameter.append({"fem.verboserank": 0})
order = 3
testSpaces = [0,order-2,order-2]

x = SpatialCoordinate(triangle)
massCoeff = 1 # +sin(dot(x,x))       # factor for mass term
diffCoeff = 1 # -0.9*cos(dot(x,x))   # factor for diffusion term

def model(space):
    dimR = space.dimRange
    exact = x[0]*x[1] * cos(pi*x[0]*x[1])

    u = TrialFunction(space)
    v = TestFunction(space)
    a = (diffCoeff*inner(grad(u),grad(v)) + massCoeff*dot(u,v) ) * dx
    b = sum( [ ( -div(diffCoeff*grad(exact[i])) + massCoeff*exact[i] ) * v[i]
             for i in range(dimR) ] ) * dx
    dbc = [dune.ufl.DirichletBC(space, exact, i+1) for i in range(4)]
    return a,b,dbc,exact

def calc(polyGrid):
    orders = (l,l,l-1)
    space = dune.vem.vemSpace( polyGrid, order=orders, storage="numpy",
                               testSpaces=testSpaces)
    a,b,dbc,exact = model(space)
    if False:
        df = space.interpolate(exact,name="solution")
    else:
        df = space.interpolate(dimR*[0],name="solution")
        scheme = dune.vem.vemScheme(
                  [a==b, *dbc], space, solver="cg",
                  gradStabilization=diffCoeff,
                  massStabilization=massCoeff)
        info = scheme.solve(target=df)
    # df.plot()
    edf = exact-df
    err = [inner(edf,edf),
           inner(grad(edf),grad(edf))]
    return [ numpy.sqrt(e) for e in integrate(polyGrid, err, order=8) ]

oldErrors = []
for i in range(1,4):
    errors = []
    N = 2*2**i
    """
    polyGrid = dune.vem.polyGrid(
          dune.vem.voronoiCells([[-0.5,-0.5],[1,1]], N*N, lloyd=100, fileName="test", load=True)
          # cartesianDomain([-0.5,-0.5],[1,1],[N,N]),
      )
    """
    polyGrid = dune.vem.polyGrid("agglomerate",
                                 cartesianDomain([-0.5,-0.5],[1,1],[N,N]),
                                 cubes=True )
    errors += calc(polyGrid)

    print("errors:",errors)
    if len(oldErrors)>0:
        print(i,"eocs:", [ math.log(oe/e)/math.log(2.)
                for oe,e in zip(oldErrors,errors) ])
    oldErrors = errors
