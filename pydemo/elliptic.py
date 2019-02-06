from __future__ import print_function

import math, sys
import numpy
from scipy.spatial import Voronoi, voronoi_plot_2d, cKDTree
import numpy.linalg
import scipy.sparse.linalg

from ufl import *

from dune.grid import cartesianDomain
import dune.ufl
import dune.fem
import dune.fem.function as gf

import dune.create as create

from voronoi import triangulated_voronoi
try:
    import graph
except:
    graph = False
try:
    import holes
except:
    holes = False

dimRange = 1
polOrder = 2

dune.fem.parameter.append({"fem.verboserank": 0})

def plot(grid, solution):
    try:
        from matplotlib import pyplot
        from numpy import amin, amax, linspace

        triangulation = grid.triangulation(4)
        data = solution.pointData(4)

        levels = linspace(amin(data[:,0]), amax(data[:,0]), 256)

        pyplot.gca().set_aspect('equal')
        pyplot.triplot(grid.triangulation(), antialiased=True, linewidth=0.2, color='black')
        pyplot.tricontourf(triangulation, data[:,0], cmap=pyplot.cm.rainbow, levels=levels)
        pyplot.show()
    except ImportError:
        pass

def error(grid, df, interp, exact):
    edf = exact-df
    ein = exact-interp
    errors = create.function("ufl", grid, "error", 5,
            [ dot(edf,edf), dot(ein,ein),
              inner(grad(edf),grad(edf)), inner(grad(ein),grad(ein))
            ] ).integrate()
    return [ math.sqrt(e) for e in errors ]

# http://zderadicka.eu/voronoi-diagrams/
class Agglomerate:
    def __init__(self,N,constructor=None,version="voronoi"):
        self.N = N
        self.suffix = str(self.N)
        self.version = version
        if version == "metis" or version == "metisVoronoi":
            assert graph
            self.suffix = "m" + self.suffix
            self.grid = create.grid("ALUSimplex", constructor=constructor, dimgrid=2)
            self.parts = graph.partition(self.grid, nparts=self.N,
                    contig=True, iptype="node", objtype="vol")
            if version == "metisVoronoi":
                voronoi_points = numpy.zeros((self.N,2))
                weights = numpy.zeros(self.N)
                for en in self.grid.elements:
                    index = self.parts[ self.grid.indexSet.index(en) ]
                    voronoi_points[index] += en.geometry.center
                    weights[index] += 1
                for p,w in zip(voronoi_points,weights):
                    p /= w
                voronoi_kdtree = cKDTree(voronoi_points)
                for en in self.grid.elements:
                    p = en.geometry.center
                    test_point_dist, test_point_regions = voronoi_kdtree.query([p], k=1)
                    index = test_point_regions[0]
                    self.parts[ self.grid.indexSet.index(en) ] = index
                vor = Voronoi(voronoi_points)
                voronoi_plot_2d(vor).savefig("metis_voronoi"+self.suffix+".pdf", bbox_inches='tight')
        elif version == "voronoi":
            self.suffix = "v" + self.suffix
            numpy.random.seed(1234)
            self.voronoi_points = numpy.random.rand(self.N, 2)
            self.voronoi_kdtree = cKDTree(self.voronoi_points)
            vor = Voronoi(self.voronoi_points)
            voronoi_plot_2d(vor).savefig("agglomerate_voronoi"+self.suffix+".pdf", bbox_inches='tight')
            if constructor == None:
                bounding_box = numpy.array([0., 1., 0., 1.]) # [x_min, x_max, y_min, y_max]
                points, triangles = triangulated_voronoi(self.voronoi_points, bounding_box)
                self.grid = create.grid("ALUSimplex", {'vertices':points, 'simplices':triangles}, dimgrid=2)
            else:
                self.grid = create.grid("ALUSimplex", constructor=constructor, dimgrid=2)
        else:
            self.suffix = "c" + self.suffix
            # self.grid = create.grid("ALUSimplex", constructor=constructor)
            self.grid = create.grid("yasp", constructor=constructor)
            self.division = constructor.division
        self.ind = set()

    def __call__(self,en):
        if self.version == "metis" or self.version == "metisVoronoi":
            index = self.parts[ self.grid.indexSet.index(en) ]
        elif self.version == "voronoi":
            p = en.geometry.center
            test_point_dist, test_point_regions = self.voronoi_kdtree.query([p], k=1)
            index = test_point_regions[0]
        else:
            idx = self.grid.indexSet.index(en)
            nx = int(idx / self.division[1])
            ny = int(idx % self.division[1])
            # print("nx,ny",nx,ny,flush=True,end="\t")
            nx = int(nx/self.division[0]*self.N[0])
            ny = int(ny/self.division[1]*self.N[1])
            index = nx*self.N[1]+ny
            # print("    idx",idx,"Dx",self.division[0],"Dy",self.division[1],"nx,ny",nx,ny,flush=True, end="\t")
            # print("     index",index,flush=True)
        self.ind.add(index)
        return index
    def check(self):
        return len(self.ind)==self.N[0]*self.N[1]

parameters = {"newton.inear.absolutetol": 1e-8, "newton.linear.reductiontol": 1e-8,
              "newton.linear.verbose": "true",
              "newton.tolerance": 3e-5,
              "newton.maxiterations": 1,
              "newton.maxlinesearchiterations":50,
              "newton.verbose": "true"
              }

h1errors = []
l2errors = []

def solve(grid,agglomerate,model,exact,name,space,scheme,order=1,penalty=None):
    print("SOLVING: ",name,space,scheme,penalty,flush=True)
    gf_exact = create.function("ufl",grid,"exact",4,exact)
    if agglomerate:
        spc = create.space(space, grid, agglomerate, dimrange=dimRange,
                order=order, storage="fem")
        assert agglomerate.check(), "missing or too many indices provided by agglomoration object. Should be "+str(agglomerate.N)+" was "+str(len(agglomerate.ind))
    else:
        spc = create.space(space, grid, dimrange=dimRange,
                order=order, storage="fem")
    interpol = spc.interpolate( gf_exact, "interpol_"+name )
    scheme = create.scheme(scheme, model, spc,
                        solver=("suitesparse","umfpack"),
                parameters=parameters)
    df = spc.interpolate([0],name=name)
    info = scheme.solve(target=df)
    errors = error(grid,df,interpol,exact)
    print("Computed",name+" size:",spc.size,
          "L^2 (s,i):", [errors[0],errors[1]],
          "H^1 (s,i):", [errors[2],errors[3]],
          "linear and Newton iterations:",
          info["linear_iterations"], info["iterations"],flush=True)
    global h1errors, l2errors
    l2errors += [errors[0],errors[1]]
    h1errors += [errors[2],errors[3]]
    return interpol, df

def compute(agglomerate,filename):
    grid = agglomerate.grid
    uflSpace = dune.ufl.Space((grid.dimGrid, grid.dimWorld), dimRange, field="double")
    u = TrialFunction(uflSpace)
    v = TestFunction(uflSpace)
    x = SpatialCoordinate(uflSpace.cell())

    ### trivial Neuman bc
    exact  = as_vector( [cos(pi*x[0])*cos(pi*x[1])] )
    ### zero boundary conditions
    exact *= x[0]*x[1]*(2-x[0])*(2-x[1])
    ### non zero and non trivial Neuman boundary conditions
    exact += as_vector( [sin(x[0]*x[1])] )

    H = lambda w: grad(grad(w))
    a = (inner(grad(u), grad(v)) + inner(u,v)) * dx
    b = ( -(H(exact[0])[0,0]+H(exact[0])[1,1]) + exact[0] ) * v[0] * dx
    model = create.model("elliptic", grid, a==b,
            *[dune.ufl.DirichletBC(uflSpace, exact, i+1) for i in range(4)]
            )

    interpol_vem, df_vem = solve(grid,agglomerate,model,exact,
                                     "vem","AgglomeratedVEM","vem",order=polOrder)
    interpol_fem, df_fem = solve(grid,None,model,exact,
                                     "fem","lagrange","h1",order=polOrder)

    grid.writeVTK(filename+agglomerate.suffix,subsampling=polOrder-1,
        pointdata=[ df_vem, interpol_vem, df_fem, interpol_fem ],
        celldata =[ create.function("local",grid,"cells",1,lambda en,x: [agglomerate(en)]) ])


start = 4
end   = 8
for i in range(end-start):
    print("*******************************************************")
    n = 2**(i+start)
    N = 2*n
    print("Test: ",n,N)
    constructor = cartesianDomain([0,0],[2,2],[N,N])
    compute(Agglomerate([n,n],version="cartesian",constructor=constructor), "cartesian")
    if i>0:
        for j in range(4):
            l2eoc = math.log( l2errors[4*i+j]/l2errors[4*(i-1)+j] ) / math.log(0.5)
            h1eoc = math.log( h1errors[4*i+j]/h1errors[4*(i-1)+j] ) / math.log(0.5)
            print("eoc:",j,l2eoc,h1eoc)
    print("*******************************************************")
