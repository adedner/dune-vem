from ufl import *
import dune.ufl

def hk(space, exact):
    massCoeff = 1
    diffCoeff = 1

    dimR = exact.ufl_shape[0]
    u = TrialFunction(space)
    v = TestFunction(space)

    a = (inner(grad(u),grad(v)) + dot(u,v) ) * dx
    b = sum( [ ( -div(diffCoeff*grad(exact[i])) + massCoeff*exact[i] ) * v[i]
             for i in range(dimR) ] ) * dx
    dbc = [dune.ufl.DirichletBC(space, exact, i+1) for i in range(4)]

    df = space.interpolate(dimR*[0],name="solution")
    scheme = dune.vem.vemScheme(
                  [a==b, *dbc], space, solver="cg",
                  gradStabilization=diffCoeff,
                  massStabilization=massCoeff)
    info = scheme.solve(target=df)

    edf = exact-df
    err = [inner(edf,edf),
           inner(grad(edf),grad(edf))]

    return df, err