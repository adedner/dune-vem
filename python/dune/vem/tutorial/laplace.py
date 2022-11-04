import matplotlib
matplotlib.rc( 'image', cmap='jet' )
import matplotlib.pyplot as plt
from ufl import *
import dune.ufl, dune.fem, dune.vem
from mixedSolver import MixedSolver # solver for mixed system

Lx,Ly = 1,1.1
x = SpatialCoordinate(triangle)
exact = sin(2*pi*x[0]/Lx)*sin(3*pi*x[1]/Ly)
forcing = -div(grad(exact))

def primal(polyGrid, order):
    spaceU = dune.vem.vemSpace( polyGrid, order=order, testSpaces=[0,order-2,order-2])
    diams  = spaceU.diameters()
    df     = spaceU.interpolate(0,name="solution")
    u,v    = TrialFunction(spaceU), TestFunction(spaceU)
    stiff  = inner(grad(u),grad(v)) * dx - inner(forcing,v) * dx
    dbc    = [dune.ufl.DirichletBC(spaceU,exact,i+1) for i in range(4)]
    scheme = dune.vem.vemScheme([stiff==0,*dbc], gradStabilization=1)
    scheme.solve(target=df)
    return df,diams

def mixed(polyGrid, order):
    spaceU     = dune.vem.bbdgSpace( polyGrid, order=order)
    spaceSigma = dune.vem.curlFreeSpace( polyGrid, order=order)
    diams      = spaceSigma.diameters()
    df         = spaceU.interpolate(0,name="solution")
    sig        = spaceSigma.interpolate([0,0],name="sigma")
    u,v        = TrialFunction(spaceU), TestFunction(spaceU)
    sigma,psi  = TrialFunction(spaceSigma), TestFunction(spaceSigma)
    schemeM = dune.vem.vemScheme(inner(sigma,psi)*dx==0,
                                 massStabilization=1)
    schemeG = dune.fem.operator.galerkin( u*div(psi)*dx )
    schemeD = dune.fem.operator.galerkin( (-div(sigma)-forcing)*v*dx )
    mixedSolver = MixedSolver(schemeM,schemeG,schemeD)
    mixedSolver.solve(target = [df,sig])
    return df,sig,diams

for order in [0,1]:
    for i in range(2,3):
        cells = dune.vem.voronoiCells([[0,0],[Lx,Ly]],2**(2*i+1),
                                      lloyd=250, load="test")
        polyGrid = dune.vem.polyGrid(cells)
        iset = polyGrid.indexSet
        @dune.fem.function.gridFunction(polyGrid, name="cells", order=0)
        def polygons(en,x):
            return polyGrid.hierarchicalGrid.agglomerate(iset.index(en))
        fig = plt.figure(figsize=(15,15))
        dune.fem.plotting.plotPointData(polygons,figure=fig,
                                        linewidth=2,colorbar=None)
        fig.savefig(f"laplace_{i}.png", dpi=300)

        dfP,diam = primal(polyGrid,order+1)
        dfM,sigma,diam = mixed(polyGrid,order)
        fig = plt.figure(figsize=(25,15))
        dfP.plot(figure=(fig,121),
                        gridLines=None,level=order,colorbar=None)
        dfM.plot(figure=(fig,122),
                        gridLines=None,level=order,colorbar=None)
        fig.savefig(f"laplace_{i}_{order}.png", dpi=300)
