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
order = 1
ln,lm, Lx,Ly = 1,0, 1,1.1
mass = dune.ufl.Constant(1, "mu")

x = SpatialCoordinate(triangle)
def model(space):
    u = TrialFunction(space)
    v = TestFunction(space)

    # exact = -grad( (sin(ln/Lx*pi*x[0])*sin(lm/Ly*pi*x[1])) )
    exact = -grad( (cos(ln/Lx*pi*x[0])*cos(lm/Ly*pi*x[1])) )
    # exact = grad( x[0]*x[1] );
    # exact = grad( x[0]*x[1]**2 ) # (y^2,2xy)

    a = (div(u)*div(v) + dot(u,v) ) * dx

    # finally the right hand side and the boundary conditions
    # sum_ij (d_i u_i d_j v_j) = -sum_ij (d_ij u_i v_j)
    b = dot( -grad(div(exact)) + exact, v) * dx
    dbc = [dune.ufl.DirichletBC(space, [0,0], i+1) for i in range(4)]
    return a,b,dbc,exact

oldErrors = []
oldDiams = None
for i in range(1,6):
    errors = []
    N = 2**(i+1)
    polyGrid = dune.vem.polyGrid(
          dune.vem.voronoiCells([[0,0],[Lx,Ly]], N*N, lloyd=200, fileName="test", load=True)
          # cartesianDomain([0.,0.],[Lx,Ly],[N,N]), cubes=False
          # cartesianDomain([0.,0.],[Lx,Ly],[N,N]), cubes=True
      )
    space = dune.vem.curlFreeSpace( polyGrid, order=order)
    a,b,dbc,exact = model(space)
    dfI = space.interpolate(exact,name="interpol")
    if False:
        df = space.interpolate(exact,name="solution")
    else:
        df = space.interpolate(exact,name="solution")
        scheme = dune.vem.vemScheme(
                  [a==b, *dbc], space, solver=("suitesparse","umfpack"),
                  parameters=parameters,
                  gradStabilization=0,
                  massStabilization=mass)
        info = scheme.solve(target=df)

    if False:
        # plot(df[0])
        # plot(df[1])
        # plot(grad(df[0])[0], grid=polyGrid)
        # plot(grad(df[1])[1], grid=polyGrid)
        plot(div(df), grid=polyGrid)
        plot(curl(df), grid=polyGrid)

    edf = exact-df
    err = [inner(edf,edf),
           inner(div(edf),div(edf))]
    errors += [ numpy.sqrt(e) for e in integrate(polyGrid, err, order=8) ]

    print(errors,"#",space.diameters())
    if len(oldErrors)>0:
        factor = oldDiams[0] / space.diameters()[0]
        print(i,"eocs:", [ math.log(oe/e)/math.log(factor)
                for oe,e in zip(oldErrors,errors) ])
    oldErrors = errors
    oldDiams = space.diameters()
    vtk = space.gridView.writeVTK("cos"+"_"+str(i),
      celldata={"solution":df,"div":div(df),
                "interpol":dfI,"divI":div(dfI),
               })
