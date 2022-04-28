from ufl import *
import dune.ufl

parameters = {"newton.linear.tolerance": 1e-8,
              "newton.tolerance": 5e-6,
              "newton.lineSearch": "simple",
              "newton.linear.verbose": False,
              "newton.verbose": True
              }

mass = dune.ufl.Constant(1, "mu")

def curlfree(space, exact):
    u = TrialFunction(space)
    v = TestFunction(space)

    a = (div(u)*div(v) + dot(u,v) ) * dx
    b = dot( -grad(div(exact)) + exact, v) * dx
    dbc = [dune.ufl.DirichletBC(space, [0,0], i+1) for i in range(4)]

#     df = space.interpolate(exact,name="solution")
    scheme = dune.vem.vemScheme(
                  [a==b, *dbc], space,
                  #solver=("suitesparse","umfpack"),
                  parameters=parameters,
                  gradStabilization=0,
                  massStabilization=mass)
    info = scheme.solve(target=df)

    edf = exact-df
    err = [inner(edf,edf),
           inner(div(edf),div(edf))]

    return err