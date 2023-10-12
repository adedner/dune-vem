import numpy as np
import random
import dune.vem
from dune.grid import gridFunction
def reference(h,offset):
    eps = 0.1
    rand = lambda l,r: random.random()*(r-l)+l
    x, y = [], []
    r = random.random()
    if r < 0.2:
        r = 2
    elif r < 0.5:
        r = 3
    elif r < 0.8:
        r = 4
    else:
        r = 5
    if r == 1:
        x += [ h*rand(0.3,0.4) ]
        y += [ h*rand(0.3,0.4) ]
    else:
        for i in range(r):
            x += [ h*rand(i/r+eps,(i+1)/r-eps) ]
            if i%2==0:
                y += [ h*rand(0.2,0.4) ]
            else:
                y += [ h*rand(0.6,0.8) ]
    inner = []
    for i in range(r):
        inner += [ [x[i]+offset[0],y[i]+offset[1]] ]
    return inner

def ncGrid(N):
    h = 1/N
    vert =  []
    for i in range(N+1):
        for j in range(N+1):
            vert += [ [j*h,i*h] ]
    c = 0
    polys = []
    for i in range(N+1):
        for j in range(N+1):
            if i>0 and j>0:
                L = len(vert)
                inner = reference(h, vert[c-N-2])
                vert += inner
                l = len(inner)
                p1 = [c-N-2,c-N-1,c] + [L+k for k in range(l-1,-1,-1)] + [c-1]
                p2 = [c-1] + [L+k for k in range(0,l,1)] + [c]
                polys += [p1]
                polys += [p2]
                # polys += [ [c-N-2,c-N-1,c,L+2,L+1,L,c-1] ]
                # polys += [ [c-1,L,L+1,L+2,c] ]
            c += 1

    cells = {"vertices":np.array(vert), "polygons":polys}
    return cells
"""
grid = dune.vem.polyGrid(cells)
print(grid.size(0),grid.hierarchicalGrid.agglomerate.size)

dune.vem.writePolygons("concave",cells)

indexSet = grid.indexSet
@gridFunction(grid, name="cells")
def polygons(en,x):
    return grid.hierarchicalGrid.agglomerate(indexSet.index(en))
# polygons.plot()
"""
