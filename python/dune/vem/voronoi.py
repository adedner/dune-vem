# http://stackoverflow.com/questions/28665491/getting-a-bounded-polygon-coordinates-from-voronoi-cells

import matplotlib.pyplot as pl
import numpy as np
import scipy as sp
import scipy.spatial
import sys, os, pickle
from scipy.spatial import Voronoi, voronoi_plot_2d, cKDTree
import numpy

eps = sys.float_info.epsilon

def in_box(towers, bounding_box):
    return np.logical_and(np.logical_and(bounding_box[0] <= towers[:, 0],
                                         towers[:, 0] <= bounding_box[1]),
                          np.logical_and(bounding_box[2] <= towers[:, 1],
                                         towers[:, 1] <= bounding_box[3]))

def voronoiCells(constructor, towers, fileName=None, load=False):
    lowerleft  = numpy.array(constructor.lower)
    upperright = numpy.array(constructor.upper)
    bounding_box = numpy.array(
            [lowerleft[0],upperright[0],lowerleft[1],upperright[1]] )

    if isinstance(towers,int):
        fileName = fileName + str(towers) + '.pickle'
        if not load or not os.path.exists(fileName):
            print("generating new seeds for voronoi grid")
            numpy.random.seed(1234)
            towers = numpy.array(
                    [ p*(upperright-lowerleft) + lowerleft
                        for p in numpy.random.rand(towers, 2) ])
            if fileName is not None:
                with open(fileName, 'wb') as f:
                    pickle.dump(towers, f)
        else:
            assert fileName is not None
            print("loading seeds for voronoi grid")
            with open(fileName, 'rb') as f:
                towers = pickle.load(f)

    # Select towers inside the bounding box
    i = in_box(towers, bounding_box)

    vor = sp.spatial.Voronoi(towers[i,:],incremental=True)
    if fileName is not None:
        voronoi_plot_2d(vor,show_points=False,show_vertices=False).\
            savefig(fileName+"inc"+str(len(towers))+".pdf", bbox_inches='tight')

    # Mirror points
    points_center = towers[i, :]
    points_left = np.copy(points_center)
    points_left[:, 0] = bounding_box[0] - (points_left[:, 0] - bounding_box[0])
    points_right = np.copy(points_center)
    points_right[:, 0] = bounding_box[1] + (bounding_box[1] - points_right[:, 0])
    points_down = np.copy(points_center)
    points_down[:, 1] = bounding_box[2] - (points_down[:, 1] - bounding_box[2])
    points_up = np.copy(points_center)
    points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_up[:, 1])
    points = np.append(points_center,
                       np.append(np.append(points_left,
                                           points_right,
                                           axis=0),
                                 np.append(points_down,
                                           points_up,
                                           axis=0),
                                 axis=0),
                       axis=0)
    # Compute Voronoi
    vor = sp.spatial.Voronoi(points)

    # Filter regions
    regions = []
    for region in vor.regions:
        flag = True
        for index in region:
            if index == -1:
                flag = False
                break
            else:
                x = vor.vertices[index, 0]
                y = vor.vertices[index, 1]
                if not(bounding_box[0] - eps <= x and x <= bounding_box[1] + eps and
                       bounding_box[2] - eps <= y and y <= bounding_box[3] + eps):
                    flag = False
                    break
        if region != [] and flag:
            regions.append(region)

    # polys = [ r+[r[0]] for r in regions ]

    indices = set()
    for poly in regions:
        for r in poly:
            indices.add( r )
    indices = np.array(list(indices))
    vorVertices = vor.vertices[indices,:]
    newind = np.zeros(len(vor.vertices),int)
    for i in range(len(indices)):
        newind[indices[i]] = i

    return {"vertices":vorVertices, "polygons":[newind[r] for r in regions]}

def triangulated_voronoi(constructor, towers):
    voronoi = voronoiCells(constructor, towers)
    vorVertices, polys = voronoi["vertices"], voronoi["polygons"]
    indices = set()
    triangles = np.zeros([0,3],int)
    for poly in polys:
        p = numpy.append(poly,[poly[0]])
        vert = vorVertices[p, :]
        for r in poly:
            indices.add( r )
        tri = sp.spatial.Delaunay(vert).simplices
        triangles = np.concatenate(
                (triangles, p[tri] ),
                axis=0 )

    indices = np.array(list(indices))
    vertices = vorVertices[indices,:]
    newind = np.zeros(len(vorVertices),int)
    for i in range(len(indices)):
        newind[indices[i]] = i
    return vertices, newind[triangles]



# n_towers = 100
# towers = np.random.rand(n_towers, 2)
# bounding_box = np.array([0., 1., 0., 1.]) # [x_min, x_max, y_min, y_max]

# points, triangles = triangulated_voronoi(towers, bounding_box)
# pl.triplot(points[:,0], points[:,1], triangles.copy())
# pl.plot(points[:,0], points[:,1], 'o')
# pl.show()
