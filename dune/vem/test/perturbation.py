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

def perturbation(space, exact):
    epsilon = 1

    laplace = lambda w: div(grad(w))
    H = lambda w: grad(grad(w))
    u = TrialFunction(space)
    v = TestFunction(space)

    kappa = epsilon*epsilon
    beta  = 1
    gamma = 0
    laplaceCoeff  = 1
    mu            = 1

    # bilinear form
    a = ( kappa*inner(H(u[0]),H(v[0])) +\
      laplaceCoeff*beta*inner(grad(u),grad(v)) +\
      mu*gamma*inner(u,v)
    ) * dx

    # right hand side and the boundary conditions
    q = sum([ H(kappa*H(exact[0])[i,j])[i,j] for i in range(2) for j in range(2) ])
    b = ( q -\
        laplaceCoeff*div(beta*grad(exact[0])) +\
        mu*gamma*exact[0] ) * v[0] * dx
    dbc = [dune.ufl.DirichletBC(space, [0], i+1) for i in range(4)]

    # stability coefficients
    biLaplaceCoeff = kappa
    diffCoeff      = laplaceCoeff*beta
    massCoeff      = mu*gamma

    df = discreteFunction(space, name="solution")
    info = {"linear_iterations":1}

    scheme = dune.vem.vemScheme( [a==b, *dbc], space,
                            hessStabilization=biLaplaceCoeff,
                            gradStabilization=diffCoeff,
                            massStabilization=massCoeff,
                            # solver=("cg")
                          )

    scheme.solve(target=df)

    edf = exact-df
    energy  = replace(a,{u:edf,v:edf}).integrals()[0].integrand()
    err = [inner(edf,edf),
           inner(grad(edf),grad(edf)),
           inner(grad(grad(edf)),grad(grad(edf))),
           energy]

    return df, err