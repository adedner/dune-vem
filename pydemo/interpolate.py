from __future__ import print_function

import math
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

dune.fem.parameter.append({"fem.verboserank": -1})

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
            self.ind.add(index)
        elif self.version == "voronoi":
            p = en.geometry.center
            test_point_dist, test_point_regions = self.voronoi_kdtree.query([p], k=1)
            index = test_point_regions[0]
            self.ind.add(index)
        else:
            idx = self.grid.indexSet.index(en) # /2
            nx = int(idx / self.division[1])
            ny = int(idx % self.division[1])
            # print("nx,ny",nx,ny)
            nx = int(nx/self.division[0]*self.N[0])
            ny = int(ny/self.division[1]*self.N[1])
            index = nx*self.N[1]+ny
            # print("index",index)
        return index
    def check(self):
        return len(self.ind)==self.N

parameters = {"newton.inear.absolutetol": 1e-13, "newton.linear.reductiontol": 1e-13,
              "newton.linear.verbose": "true",
              "newton.tolerance": 3e-12,
              "newton.maxiterations": 50,
              "newton.maxiterations": 2500,
              "newton.maxlinesearchiterations":50,
              "newton.verbose": "true"
              }

# test of higher order vem spaces (only interpolation)
def test(agglomerate, polys):
    dfs,err = [],[]
    x = SpatialCoordinate(triangle)
    # f = as_vector( [2] )
    # f = as_vector( [2*x[0]] )
    # f = as_vector( [2*(x[0]*x[1])**2] )
    f = as_vector( [cos(x[0])*cos(x[1]) ] )
    for p in polys:
        space = create.space("agglomeratedvem", agglomerate.grid, agglomerate,
                dimrange=1, order=p, storage="fem")
        dfs += [space.interpolate(f,name="df"+str(p))]
        err += [error(agglomerate.grid,dfs[-1],dfs[-1],f)]
        dfs += [dune.create.function("ufl",space.grid,name="gdf"+str(p),
            ufl=grad(dfs[-1][0]),order=p-1)]
        print("error:",p,err[-1])
    return polys, dfs, err

N = 16
constructor = cartesianDomain([0,0],[1,1],[N,N])
err = []
n = 1
polys = [1,2,3]
while n <= N:
    print("n/N=",n,N)
    agglomerate = Agglomerate([n,n],version="cartesian",constructor=constructor)
    polys,dfs,e = test(agglomerate,polys)
    err += [e]
    dfs[0].space.grid.writeVTK("test"+str(n),
        pointdata=dfs, celldata = [ create.function("local",agglomerate.grid,"cells",1,lambda en,x: [agglomerate(en)]) ])
    n = n*2

eoc = lambda E,e: math.log(E/e)/math.log(2.)
print("degree","\t","step","\t\t","eocL2","\t\t\t","eocH1")
for p,poly in enumerate(polys):
    for i in range(1,len(err)):
        print(poly,"\t",i,"\t\t",eoc(err[i-1][p][0],err[i][p][0]), "\t", eoc(err[i-1][p][2],err[i][p][2]))
