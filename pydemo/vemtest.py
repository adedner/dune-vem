try:
    import dune.vem
except:
    print("This example needs 'dune.vem'")
    import sys
    sys.exit(0)
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
dimR = 3
# testSpaces = [ [0], [order-3,order-2], [order-4] ]
testSpaces = [-1,order-1,order-2]
# testSpaces = [0,order-2,order-2]

x = SpatialCoordinate(triangle)
massCoeff = 1 # +sin(dot(x,x))       # factor for mass term
diffCoeff = 1 # -0.9*cos(dot(x,x))   # factor for diffusion term

H = lambda u: grad(grad(u))
Hplot = lambda u: [ H(u)[0,0], H(u)[1,0], H(u)[0,1], H(u)[1,1] ]

def model(space):
    dimR = space.dimRange
    u = TrialFunction(space)
    v = TestFunction(space)

    exact = as_vector( [x[0]*x[1] * cos(pi*x[0]*x[1])]*dimR )

    a = (diffCoeff*inner(grad(u),grad(v)) + massCoeff*dot(u,v) ) * dx

    # finally the right hand side and the boundary conditions
    b = sum( [ ( -div(diffCoeff*grad(exact[i])) + massCoeff*exact[i] ) * v[i]
             for i in range(dimR) ] ) * dx
    dbc = [dune.ufl.DirichletBC(space, exact, i+1) for i in range(4)]
    return a,b,dbc,exact

def writeVTK(name,df):
  polyGrid.writeVTK(name,
       pointdata={"df0":[df[0],grad(df[0])[0],grad(df[0])[1]]+
                         Hplot(df[0]),
                  "df1":[df[1],grad(df[1])[0],grad(df[1])[1]]+
                         Hplot(df[1]),
                  "df2":[df[2],grad(df[2])[0],grad(df[2])[1]]+
                         Hplot(df[2]),
                 },
       subsampling=3)

oldErrors = []
for i in range(1,6):
    errors = []
    N = 2*2**i
    polyGrid = dune.vem.polyGrid(
          dune.vem.voronoiCells([[-0.5,-0.5],[1,1]], N*N, lloyd=100, fileName="test", load=True)
          # cartesianDomain([-0.5,-0.5],[1,1],[N,N]),
      )
    space = dune.vem.vemSpace( polyGrid, order=order, dimRange=dimR, storage="numpy",
                               testSpaces=testSpaces,
                               vectorSpace=True)
    a,b,dbc,exact = model(space)
    df = space.interpolate(exact,name="solution")
    edf = exact-df
    err = [inner(edf,edf),
           inner(grad(edf),grad(edf))]
    errors += [ numpy.sqrt(e) for e in integrate(polyGrid, err, order=8) ]

    space = dune.vem.vemSpace( polyGrid, order=order, dimRange=dimR, storage="numpy",
                               testSpaces=testSpaces,
                               vectorSpace=False)
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
    edf = exact-df
    err = [inner(edf,edf),
           inner(grad(edf),grad(edf))]
    errors += [ numpy.sqrt(e) for e in integrate(polyGrid, err, order=8) ]
    print(errors)
    if len(oldErrors)>0:
        print(i,"eocs:", [ math.log(oe/e)/math.log(2.)
                for oe,e in zip(oldErrors,errors) ])
    oldErrors = errors
