def biharmonic(space, exact):
    laplace = lambda w: div(grad(w))
    H = lambda w: grad(grad(w))
    u = TrialFunction(space)
    v = TestFunction(space)

    a = ( inner(H(u[0]),H(v[0])) ) * dx
    b = ( laplace(laplace(exact[0])) )* v[0] * dx
    dbcs = [dune.ufl.DirichletBC(space, [0], i+1) for i in range(4)]

    scheme = dune.vem.vemScheme( [a==b, *dbcs], space, hessStabilization=1, solver="cg", parameters=parameters )
    df = discreteFunction(space, name="solution")
    info = scheme.solve(target=df)

    return df