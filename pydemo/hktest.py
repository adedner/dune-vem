import dune.vem
import matplotlib
matplotlib.rc( 'image', cmap='jet' )
from matplotlib import pyplot
import math, numpy
from dune.grid import cartesianDomain, gridFunction
from dune.fem.plotting import plotPointData as plot
from dune.fem import integrate
from dune.fem.function import discreteFunction
import dune.fem
from concave import ncGrid

from ufl import *
import dune.ufl

dune.fem.parameter.append({"fem.verboserank": 0})
order = 3
testSpaces = [0,order-2,order-2]

x = SpatialCoordinate(triangle)
massCoeff = 1+sin(dot(x,x))       # factor for mass term
diffCoeff = 1-0.9*cos(dot(x,x))   # factor for diffusion term

def model(space):
    dimR = space.dimRange
    exact = as_vector(dimR*[x[0]*x[1] * cos(pi*x[0]*x[1])])

    u = TrialFunction(space)
    v = TestFunction(space)
    a = (diffCoeff*inner(grad(u),grad(v)) + massCoeff*dot(u,v) ) * dx
    b = sum( [ ( -div(diffCoeff*grad(exact[i])) + massCoeff*exact[i] ) * v[i]
             for i in range(dimR) ] ) * dx
    dbc = [dune.ufl.DirichletBC(space, exact, i+1) for i in range(4)]
    return a,b,dbc,exact

def calc(polyGrid):
    dimR = 1
    orders = (order,order,order-1)
    space = dune.vem.vemSpace( polyGrid, order=orders, storage="numpy",
                               dimRange=dimR,testSpaces=testSpaces)
    a,b,dbc,exact = model(space)
    if False:
        df = space.interpolate(exact,name="solution")
    else:
        df = space.interpolate(dimR*[0],name="solution")
        scheme = dune.vem.vemScheme(
                  [a==b, *dbc], space, solver="cg",
                  gradStabilization=None, # diffCoeff,
                  massStabilization=massCoeff)
        info = scheme.solve(target=df)
    # df.plot()
    edf = exact-df
    err = [inner(edf,edf),
           inner(grad(edf),grad(edf))]
    return [ numpy.sqrt(e) for e in integrate(err) ]

oldErrors = []
for i in range(1,4):
    errors = []
    N = 2*2**i
    """
    polyGrid = dune.vem.polyGrid(
          dune.vem.voronoiCells([[-0.5,-0.5],[1,1]], N*N, lloyd=100, fileName="test", load=True)
      )
    """
    # polyGrid = dune.vem.polyGrid(cartesianDomain([-0.5,-0.5],[1,1],[N,N]), cubes=False )
    cells = ncGrid(N)
    polyGrid = dune.vem.polyGrid( cells )
    dune.vem.writePolygons(f"concave_{i}",cells)

    errors += calc(polyGrid)

    print(i,polyGrid.hierarchicalGrid.agglomerate.size,"errors:",errors)
    if len(oldErrors)>0:
        eocs = [ math.log(oe/e)/math.log(2.) for oe,e in zip(oldErrors,errors) ]
        print(i,"eocs:", eocs)
        assert eocs[0] > order+0.5
        assert eocs[1] > order-0.5
    oldErrors = errors
