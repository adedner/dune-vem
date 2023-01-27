import numpy as np
from matplotlib import pyplot as plt
from dune.grid import gridFunction
from dune.vem import voronoiCells, polyGrid

def hexaGrid(N,Lx,Ly):
    # assert N%3 == 1
    ratio = np.sqrt(3)/2 # cos(60°)
    xv, yv = np.meshgrid(np.arange(N), np.arange(N), sparse=False, indexing='xy')
    xv = xv * ratio
    xv[::2, :] += ratio/2
    px = []
    py = []
    cx = []
    cy = []
    for i, (rx, ry) in enumerate(zip(xv,yv)): # rows
        for j, (x, y) in enumerate(zip(rx,ry)): # coordinates
            if i%2==0 and j%3==0:
                cx += [x]
                cy += [y]
            elif i%2==1 and j%3==2:
                cx += [x]
                cy += [y]
            else:
                if i%2==1 and j==0: x += ratio/2
                px += [x]
                py += [y]
        if i%2 == 1:
            px += [px[-1]+ratio/2]
            py += [py[-1]]
    px += [ratio/2,ratio/2,   ratio*(N-0.5), ratio*(N-0.5)]
    py += [0,N-1, 0, N-1]

    px = np.array(px, dtype=float)
    py = np.array(py, dtype=float)

    cx = np.array(cx, dtype=float)
    cy = np.array(cy, dtype=float)

    cells = {}
    cells["vertices"] = np.array([py,px]).transpose()
    cells["polygons"] = []
    for i, (x,y) in enumerate(zip(cy,cx)):
        cells["polygons"] += np.where( np.linalg.norm( cells["vertices"]-np.array([x,y]),
                               ord=np.inf, axis=1 ) < 1.2 )

    for i,c in enumerate(cells["polygons"]):
        if len(c) == 6:
            c[0],c[1],c[2],c[3],c[4],c[5] = c[0],c[1],c[3],c[5],c[4],c[2]

    def trans(p):
        p[:,0] -= min(p[:,0])
        p[:,0] *= Lx/max(p[:,0])
        p[:,1] -= min(p[:,1])
        p[:,1] *= Ly/max(p[:,1])
    trans(cells["vertices"])

    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(px, py, s=40)
    ax.scatter(cx, cy, s=10)
    ax.axis('equal')
    plt.show()
    """
    return cells
    """
    grid = polyGrid(cells, convex=True)
    indexSet = grid.indexSet
    @gridFunction(grid, name="cells")
    def polygons(en,x):
        return grid.hierarchicalGrid.agglomerate(indexSet.index(en))
    # polygons.plot()
    return grid
    """

"""
for i in range(1,6,2):
    grid = hexaGrid(6*i+1, 1.2,1)
"""
