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
              "newton.linear.verbose": False,
              "newton.verbose": False
              }
Lx,Ly = 1,1.1
orders = [0,1,2]
useVem = True
gridType = "voronoi"

def model(spaceU):
    x = SpatialCoordinate(triangle)
    # exact   = (x[0]-Lx)*(x[1]-Ly)*x[0]*x[1]
    exact = sin(2*pi*x[0]/Lx)*sin(3*pi*x[1]/Ly)
    forcing = -div(grad(exact))
    dbc = [dune.ufl.DirichletBC(spaceU, exact, i+1) for i in range(4)]
    return forcing,dbc,exact

def mixed(polyGrid,level,oder, useVem):
    if useVem:
        # unstable spaceU     = dune.fem.space.dglegendre( polyGrid, order=order)
        spaceU     = dune.vem.bbdgSpace( polyGrid, order=order)
        spaceSigma = dune.vem.curlFreeSpace( polyGrid, order=order)
        scheme     = lambda eq: dune.vem.vemScheme(
                       eq, spaceSigma, solver=("suitesparse","umfpack"),
                       parameters=parameters,
                       gradStabilization=None, massStabilization=1)
        diams      = spaceSigma.diameters()
    else:
        # supotimal on cubes
        if gridType.lower() == "cube":
            spaceU     = dune.fem.space.dglegendre( polyGrid, order=order)
        else:
            spaceU     = dune.fem.space.dgonb( polyGrid, order=order)
        spaceSigma = dune.fem.space.raviartThomas( polyGrid, order=order)
        # spaceSigma = dune.fem.space.bdm( polyGrid, order=order+1)
        scheme     = lambda eq: dune.fem.scheme.galerkin(
                       eq, spaceSigma, solver=("suitesparse","umfpack"),
                       parameters=parameters)
        diams      = 2*[2**(-level)]

    forcing,dbc,exact = model(spaceU)
    df  = spaceU.interpolate(exact,name="solution")
    sig = spaceSigma.interpolate(grad(exact),name="sigma")
    # return exact,df,sig,diams

    u     = TrialFunction(spaceU)
    v     = TestFunction(spaceU)
    sigma = TrialFunction(spaceSigma)
    psi   = TestFunction(spaceSigma)
    massOp = inner(sigma,psi) * dx
    gradOp =  u*div(psi) * dx # - inner(dot(sigma,n),exact)
    divOp  = -div(sigma)*v * dx - inner(forcing,v) * dx

    schemeM = scheme(massOp==0)
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

    tmp = spaceSigma.interpolate([0,0],name="rhs")
    rhs = spaceU.interpolate(0,name="rhs")
    schemeD(tmp,rhs)
    df.as_numpy[:] = scipy.sparse.linalg.cg(mixedOp(),rhs.as_numpy)[0]
    tmp.as_numpy[:] = -G@df.as_numpy[:]
    sig.as_numpy[:] = spsolve(M,tmp.as_numpy[:])

    vtk = spaceU.gridView.writeVTK("mixed"+"_"+str(level),subsampling=3,
      celldata={"solution":df, "cells":polygons, })
    return exact,df,sig,diams

def primal(polyGrid,level,order, useVem):
    if useVem:
        spaceU = dune.vem.vemSpace( polyGrid, order=order,
                                        testSpaces=[0,order-2,order-2])
        scheme     = lambda eq: dune.vem.vemScheme(
                       eq, spaceU, solver=("suitesparse","umfpack"),
                       parameters=parameters,
                       gradStabilization=1, massStabilization=None)
        diams      = spaceU.diameters()
    else:
        spaceU = dune.fem.space.lagrange( polyGrid, order=order)
        scheme     = lambda eq: dune.fem.scheme.galerkin(
                       eq, spaceU, solver=("suitesparse","umfpack"),
                       parameters=parameters)
        diams      = 2*[2**(-level)]

    forcing,dbc,exact = model(spaceU)
    df = spaceU.interpolate(exact,name="solution")
    # return exact,df,diams

    u      = TrialFunction(spaceU)
    v      = TestFunction(spaceU)
    stiff  = inner(grad(u),grad(v)) * dx - inner(forcing,v) * dx
    scheme = scheme([stiff==0,*dbc])

    scheme.solve(target=df)
    vtk = spaceU.gridView.writeVTK("primal"+"_"+str(level),subsampling=3,
      celldata={"solution":df, "cells":polygons, })
    return exact,df,diams

for order in orders:
    oldErrors = []
    oldDiams = None
    for i in range(1,6):
        errors = []
        sizes = []
        N = 2**(i+1)
        if gridType.lower() == "cube":
            polyGrid = dune.vem.polyGrid(
                  cartesianDomain([0.,0.],[Lx,Ly],[N,N]), cubes=True
              )
        elif gridType.lower() == "simplex":
            assert useVem, "RT not working for simplex grids"
            polyGrid = dune.vem.polyGrid(
                  cartesianDomain([0.,0.],[Lx,Ly],[N,N]), cubes=False
              )
        elif gridType.lower() == "voronoi":
            assert useVem
            polyGrid = dune.vem.polyGrid(
                  dune.vem.voronoiCells([[0,0],[Lx,Ly]], N*N, lloyd=200, fileName="test", load=True)
              )
        indexSet = polyGrid.indexSet
        @gridFunction(polyGrid, name="cells")
        def polygons(en,x):
            return polyGrid.hierarchicalGrid.agglomerate(indexSet.index(en))

        #######################################################
        exact,df,sigma,diam = mixed(polyGrid,i,order, useVem)
        e  = exact-df
        de = grad(exact)-sigma
        err = [e**2,inner(de,de),div(de)**2]
        errors += [ numpy.sqrt(e) for e in integrate(polyGrid, err, order=8) ]
        sizes += [ df.size, sigma.size ]
        #######################################################
        exact,df,diam = primal(polyGrid,i,order+1, useVem)
        e = exact-df
        de = grad(e)
        d2e = grad(grad(e))
        err = [e**2,inner(de,de),div(de)**2,inner(d2e,d2e)]
        errors += [ numpy.sqrt(e) for e in integrate(polyGrid, err, order=8) ]
        sizes += [ df.size ]
        #######################################################

        print(order,i,*sizes,*errors,"# order=",order,diam,flush=True)
        if len(oldErrors)>0:
            factor = oldDiams[0] / diam[0]
            print(order,i,*[ math.log(oe/e)/math.log(factor)
                    for oe,e in zip(oldErrors,errors) ],
                    "#eoc order=",order, flush=True)
        oldErrors = errors
        oldDiams = diam
