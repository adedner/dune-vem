import dune.vem
import matplotlib
matplotlib.rc( 'image', cmap='jet' )
from matplotlib import pyplot
import math, numpy
import scipy
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix, bmat
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
              "newton.linear.verbose": True,
              "newton.verbose": True
              }
order, ln,lm, Lx,Ly = 1,  2,3, 1,1.1
mass = dune.ufl.Constant(1, "mu")

def model(spaceU):
    x = SpatialCoordinate(triangle)
    exact   = (x[0]-Lx)*(x[1]-Ly)*x[0]*x[1]
    forcing = -div(grad(exact))
    dbc = [dune.ufl.DirichletBC(spaceU, exact, i+1) for i in range(4)]
    return forcing,dbc,exact
def mixed(polyGrid,level):
    spaceU     = dune.vem.bbdgSpace( polyGrid, order=order)
    spaceSigma = dune.vem.curlFreeSpace( polyGrid, order=order)
    df = spaceU.interpolate(0,name="solution")

    forcing,dbc,exact = model(spaceU)

    u     = TrialFunction(spaceU)
    v     = TestFunction(spaceU)
    sigma = TrialFunction(spaceSigma)
    psi   = TestFunction(spaceSigma)
    massOp = inner(sigma,psi) * dx
    gradOp = +inner(u,div(psi)) * dx # - inner(dot(sigma,n),exact)
    divOp  = -inner(div(sigma),v) * dx - inner(forcing,v) * dx

    tmp = spaceSigma.interpolate([0,0],name="rhs")
    rhs = spaceU.interpolate(0,name="rhs")
    schemeM = dune.vem.vemScheme(
                  massOp==0, spaceSigma, solver=("suitesparse","umfpack"),
                  parameters=parameters,
                  gradStabilization=0, massStabilization=1)
    schemeG = dune.fem.operator.galerkin( gradOp, spaceU, spaceSigma )
    schemeD = dune.fem.operator.galerkin( divOp, spaceSigma,spaceU )

    M = dune.fem.operator.linear(schemeM).as_numpy
    G = dune.fem.operator.linear(schemeG).as_numpy
    D = dune.fem.operator.linear(schemeD).as_numpy
    # (  M   G  ) ( u     )   ( g ) # g: Dirichlet BC - Neuman BC are natural
    # ( -D   0  ) ( sigma ) = ( f )
    # sigma = -M^{-1}Gu + M^{-1}g
    # f = -Dsigma = DM^{-1}Gu - DM^{-1}g
    # DM^{-1}Gu = f + DM^{-1}g
    # Au = b with A = DM^{-1}G and b = f + DM^{-1}g
    class mixedOp(scipy.sparse.linalg.LinearOperator):
        def __init__(self):
            self.shape = (spaceU.size, spaceU.size)
            self.dtype = df.as_numpy[:].dtype
            self.s1 = spaceSigma.interpolate([0,0],name="tmp").as_numpy[:]
            self.s0 = spaceSigma.interpolate([0,0],name="tmp").as_numpy[:]
            self.y  = spaceU.interpolate(0,name="tmp").as_numpy[:]
        def update(self, x_coeff, f):
            pass
        def _matvec(self, x_coeff):
            x = spaceU.function("tmp", dofVector=x_coeff)
            self.s0[:] = G@x_coeff[:]
            self.s1[:] = spsolve(M,self.s0[:])
            self.y[:] = D@self.s1[:]
            return self.y

    schemeD(tmp,rhs)
    df.as_numpy[:] = scipy.sparse.linalg.cg(mixedOp(),rhs.as_numpy)[0]
    vtk = spaceU.gridView.writeVTK("mixed"+"_"+str(level),subsampling=3,
      celldata={"solution":df, "cells":polygons, })
    return exact,df,spaceSigma.diameters()

def primal(polyGrid,level):
    spaceU     = dune.vem.vemSpace( polyGrid, order=order,
                                    testSpaces=[0,[order-2,-1],order-2])
    df = spaceU.interpolate(0,name="solution")

    forcing,dbc,exact = model(spaceU)

    u      = TrialFunction(spaceU)
    v      = TestFunction(spaceU)
    stiff  = inner(grad(u),grad(v)) * dx - inner(forcing,v) * dx
    scheme = dune.vem.vemScheme(
                  [stiff==0,*dbc], spaceU, solver=("suitesparse","umfpack"),
                  parameters=parameters, gradStabilization=1)

    scheme.solve(target=df)
    vtk = spaceU.gridView.writeVTK("primal"+"_"+str(level),subsampling=3,
      celldata={"solution":df, "cells":polygons, })
    df.plot()
    return exact,df,spaceU.diameters()

oldErrors = []
oldDiams = None
for i in range(1,6):
    errors = []
    N = 2**(i+1)
    polyGrid = dune.vem.polyGrid(
          dune.vem.voronoiCells([[0,0],[Lx,Ly]], N*N, lloyd=200, fileName="test", load=True)
          # cartesianDomain([0.,0.],[Lx,Ly],[N,N]), cubes=True
          # cartesianDomain([0.,0.],[Lx,Ly],[N,N]), cubes=True
      )
    indexSet = polyGrid.indexSet
    @gridFunction(polyGrid, name="cells")
    def polygons(en,x):
        return polyGrid.hierarchicalGrid.agglomerate(indexSet.index(en))

    #######################################################
    exact,df,diam = mixed(polyGrid,i)
    edf = exact-df
    err = [edf*edf,inner(grad(edf),grad(edf))]
    errors += [ numpy.sqrt(e) for e in integrate(polyGrid, err, order=8) ]
    #######################################################
    exact,df,diam = primal(polyGrid,i)
    edf = exact-df
    err = [edf*edf,inner(grad(edf),grad(edf))]
    errors += [ numpy.sqrt(e) for e in integrate(polyGrid, err, order=8) ]
    #######################################################

    print(errors,"#",diam,flush=True)
    if len(oldErrors)>0:
        factor = oldDiams[0] / diam[0]
        print(i,"eocs:", [ math.log(oe/e)/math.log(factor)
                for oe,e in zip(oldErrors,errors) ],flush=True)
    oldErrors = errors
    oldDiams = diam
