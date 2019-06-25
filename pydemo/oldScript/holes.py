import math
import numpy
import triangle
import matplotlib.pyplot as plt
import triangle.plot as trplot


def ellipse(x0,y0,r1,r2,N):
    p = [ [r1*math.cos(t*2*math.pi/N)+x0,r2*math.sin(t*2*math.pi/N)+y0] for t in range(N) ]
    e = [ [i,(i+1)%N] for i in range(N) ]
    return p,e

def plot(tr):
    trplot.plot(plt.axes(), **tr)
    plt.show()

def get(area,plotDomain=False):
    N=50
    pts,segs = ellipse(0,0,10,8,2*N)

    holes =  [ [[1,1],[1.5,3.5], N] ]
    holes += [ [[-3.5,-1.8],[3,1.5], N] ]
    holes += [ [[3.5,-1.8],[1,2.5], N] ]
    holes += [ [[4.5,0],[3,0.5], N] ]
    holes += [ [[-4,2],[5,2], N] ]
    holes += [ [[0.5,2]] ]
    # holes += [ [[0,3.5],[5,2], N] ]

    for h in holes:
        if len(h) == 1: continue
        n = len(pts)
        p,s = ellipse( *h[0], *h[1], h[2] )
        pts  += p
        segs += [ [e[0]+n,e[1]+n] for e in s ]
    domain = {"vertices":numpy.array(pts),
              "segments":numpy.array(segs),
              "holes":numpy.array([ h[0] for h in holes ]) }

    tr = triangle.triangulate(domain, opts="pqa"+str(area))
    if plotDomain:
        plot(tr)
    tr['simplices'] = tr.pop('triangles')
    tr.pop('segments')
    return tr

get(0.1,True)
