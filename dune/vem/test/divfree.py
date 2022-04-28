from ufl import *
import dune.ufl

mass = dune.ufl.Constant(1, "mu")

def divfree(space, exact):
    u = TrialFunction(space)
    v = TestFunction(space)

    a = (inner(grad(u),grad(v)) + dot(u,v) ) * dx

    b = dot( -div(grad(exact)) + exact, v) * dx
    dbc = [dune.ufl.DirichletBC(space, [0,0], i+1) for i in range(4)]

    df = space.interpolate(exact,name="solution")
    # scheme = dune.vem.vemScheme(
    #               [a==b, *dbc], space,
    #             #   solver=("suitesparse","umfpack"),
    #               parameters=parameters,
    #               gradStabilization=0,
    #               massStabilization=mass)
    # info = scheme.solve(target=df)

    edf = exact-df
    err = [inner(edf,edf)]

    return err