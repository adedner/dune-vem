from dune import create
from dune.grid import cartesianDomain, gridFunction
from dune.fem.plotting import plotPointData as plot
from dune.fem.function import integrate, discreteFunction
from dune.fem import parameter
from dune.vem import voronoiCells
from dune.fem.operator import linear as linearOperator

import ufl.algorithms
from ufl import *
import dune.ufl

def interpolate(space, exact):
    df = discreteFunction(space, name="solution")
    df.interpolate(exact)

    edf = exact-df
    err = [inner(edf,edf),
           inner(grad(edf),grad(edf)),
           inner(grad(grad(edf)),grad(grad(edf)))]

    return df, err