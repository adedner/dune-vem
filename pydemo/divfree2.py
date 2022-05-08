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
parameters = {"newton.linear.tolerance": 1e-8,
              "newton.tolerance": 5e-6,
              "newton.lineSearch": "simple",
              "newton.linear.verbose": False,
              "newton.verbose": True
              }
order, Lx,Ly = 3,  3,1
# order, Lx,Ly = 5,  1,1.1
mass = dune.ufl.Constant(1, "mu")
D = dune.ufl.Constant(1, "mu")

x = SpatialCoordinate(triangle)
def model(space):
    u = TrialFunction(space)
    v = TestFunction(space)

    if False:
        # ( -y+x^2y , x-y^2x)
        # div: 2xy - 2yx = 0
        exact = as_vector([-x[1]+x[0]**2*x[1],
                            x[0]-x[1]**2*x[0]])
    elif True:
        # div: (p+1)x^py^p - (p+1)y^px^p = 0
        p = 1
        exact = as_vector([-x[1]+x[0]**(p+1)*x[1]**p,
                            x[0]-x[1]**(p+1)*x[0]**p])
    else:
        exact = as_vector( [x[1] * (1.-x[1]), 0] )
        # exact = as_vector( [10.,0.] )

    a = (D*inner(grad(u),grad(v)) + mass*dot(u,v) ) * dx

    # b = dot( -D*div(grad(exact)) + mass*exact, v) * dx
    b = dot( mass*exact, v) * dx
    dbc = [dune.ufl.DirichletBC(space, [0,0], i+1) for i in range(4)]
    return a,b,dbc,exact

oldErrors = []
oldDiams = None
for i in range(0,6):
    errors = []
    N = 2**i # 2**(i+1)
    polyGrid = dune.vem.polyGrid(
          # dune.vem.voronoiCells([[0,0],[Lx,Ly]], 10*N*N, lloyd=200, fileName="test", load=True)
          cartesianDomain([0.,0.],[3,1],[3*N,2*N]), cubes=False
          # cartesianDomain([0.,0.],[3,1],[3*N,3*N]), cubes=True
      )

    indexSet = polyGrid.indexSet
    @gridFunction(polyGrid, name="cells")
    def polygons(en,x):
        return polyGrid.hierarchicalGrid.agglomerate(indexSet.index(en))
    # polygons.plot()
    space = dune.vem.divFreeSpace( polyGrid, order=order)
    a,b,dbc,exact = model(space)
    dfI = space.interpolate(exact,name="interpol")
    if True:
        df = space.interpolate(exact,name="solution")
        df.plot()
    else:
        df = space.interpolate(exact,name="solution")
        scheme = dune.vem.vemScheme(
                  [a==b, *dbc], space, solver=("suitesparse","umfpack"),
                  parameters=parameters,
                  gradStabilization=D,
                  massStabilization=mass)
        info = scheme.solve(target=df)
        df.plot()

    edf = exact-df
    err = [inner(edf,edf), inner(grad(edf),grad(edf))]
    errors += [ numpy.sqrt(e) for e in integrate(polyGrid, err, order=8) ]

    print(errors,"#",space.diameters(),flush=True)
    if len(oldErrors)>0:
        factor = oldDiams[0] / space.diameters()[0]
        print(i,"eocs:", [ math.log(oe/e)/math.log(factor)
                for oe,e in zip(oldErrors,errors) ],flush=True)
    oldErrors = errors
    oldDiams = space.diameters()
    vtk = space.gridView.writeVTK("cos"+"_"+str(i),subsampling=3,
      celldata={"solution":df,"error":edf,
                "cells":polygons,
               })
