# these two versions should work - but the second doesn't:
# useVem, useTaylorHood  = False, True
useVem, useTaylorHood  = True,  False

import matplotlib
matplotlib.rc( 'image', cmap='jet' )
from matplotlib import pyplot

from ufl import SpatialCoordinate, CellVolume, TrialFunction, TestFunction,\
                inner, dot, div, grad, dx, as_vector, transpose, Identity
from dune.ufl import Constant, DirichletBC

from dune.fem.operator import linear as linearOperator
from dune.fem import parameter
parameter.append({"fem.verboserank": 0})

# grid construction
from dune.grid import structuredGrid, cartesianDomain
if not useVem:
    from dune.alugrid import aluCubeGrid as leafGridView
else:
    from dune.vem import polyGrid        as leafGridView
grid = leafGridView( cartesianDomain([0,0],[3,1],[30,10]) )

# spaces
order = 2

if not useVem:
    from dune.fem.space import lagrange     as velocitySpace
    if useTaylorHood:
        from dune.fem.space import lagrange as pressureSpace
    else:
        from dune.fem.space import dgonb    as pressureSpace
    from dune.fem.scheme   import galerkin  as velocityScheme
    from dune.fem.scheme   import dg        as pressureScheme
    from dune.fem.operator import galerkin  as galerkinOperator
else:
    from dune.vem import vemSpace           as velocitySpace
    if useTaylorHood:
        from dune.vem import vemSpace       as pressureSpace
    else:
        from dune.vem import bbdgSpace      as pressureSpace
    from dune.vem          import vemScheme as velocityScheme
    from dune.fem.scheme   import dg        as pressureScheme
    from dune.fem.operator import galerkin  as galerkinOperator

if useTaylorHood:
    spcU = velocitySpace(grid, dimRange=grid.dimension, order=order,
              storage="petsc")
    spcP = pressureSpace(grid, dimRange=1, order=order-1,
              storage="petsc")
else:
    spcU = velocitySpace(grid, dimRange=grid.dimension, order=order,
              conforming=False, storage="petsc")
    spcP = pressureSpace(grid, dimRange=1, order=order-1,
              storage="petsc")

mu_   = 0.1
nu_   = 0.0
cell  = spcU.cell()
x     = SpatialCoordinate(cell)
mu    = Constant(mu_,  "mu")
nu    = Constant(nu_,  "nu")
u     = TrialFunction(spcU)
v     = TestFunction(spcU)
p     = TrialFunction(spcP)
q     = TestFunction(spcP)
exact_u     = as_vector( [x[1] * (1.-x[1]), 0] )
exact_p     = as_vector( [ (-2*x[0] + 2)*mu ] )
f           = as_vector( [0,]*grid.dimension )
f          += nu*exact_u
mainModel   = (nu*dot(u,v) + mu*inner(grad(u)+grad(u).T, grad(v)) - dot(f,v)) * dx
gradModel   = -inner( p[0]*Identity(grid.dimension), grad(v) ) * dx
divModel    = -div(u)*q[0] * dx
massModel   = inner(p,q) * dx
preconModel = inner(grad(p),grad(q)) * dx

if not useVem:
    mainOp      = velocityScheme([mainModel==0,DirichletBC(spcU,exact_u,1)])
else:
    mainOp      = velocityScheme([mainModel==0,DirichletBC(spcU,exact_u,1)],
                         gradStabilization=as_vector([2*mu_,2*mu_]),
                         massStabilization=as_vector([nu_,nu_])
                  )
gradOp      = galerkinOperator(gradModel)
divOp       = galerkinOperator(divModel)
massOp      = pressureScheme(massModel==0,
              #       gradStabilization=0,
              #       massStabilization=1,
              )
preconOp    = pressureScheme(preconModel==0,
              #       gradStabilization=1,
              #       massStabilization=0,
                    penalty=1,
              )

velocity = spcU.interpolate([0,]*spcU.dimRange, name="velocity")
pressure = spcP.interpolate([0], name="pressure")
rhsVelo  = velocity.copy()
rhsPress = pressure.copy()

r      = rhsPress.copy()
d      = rhsPress.copy()
precon = rhsPress.copy()
xi     = rhsVelo.copy()

A = linearOperator(mainOp)
G = linearOperator(gradOp)
D = linearOperator(divOp)
M = linearOperator(massOp)
P = linearOperator(preconOp)

solver = {"method":"cg","tolerance":1e-10, "verbose":False}
Ainv   = mainOp.inverseLinearOperator(A,parameters=solver)
Minv   = massOp.inverseLinearOperator(M,parameters=solver)
Pinv   = preconOp.inverseLinearOperator(P,solver)

mainOp(velocity,rhsVelo)
rhsVelo *= -1
G(pressure,xi)
rhsVelo -= xi
mainOp.setConstraints(rhsVelo)
mainOp.setConstraints(velocity)

Ainv(rhsVelo, velocity)
D(velocity,rhsPress)
Minv(rhsPress, r)

if mainOp.model.nu > 0:
    precon.clear()
    Pinv(rhsPress, precon)
    r *= mainOp.model.mu
    r.axpy(mainOp.model.nu,precon)
d.assign(r)
delta = r.scalarProductDofs(rhsPress)
print("delta:",delta,flush=True)
assert delta >= 0
for m in range(1000):
    xi.clear()
    G(d,rhsVelo)
    mainOp.setConstraints(\
        [0,]*grid.dimension, rhsVelo)
    Ainv(rhsVelo, xi)
    D(xi,rhsPress)
    rho = delta /\
       d.scalarProductDofs(rhsPress)
    pressure.axpy(rho,d)
    velocity.axpy(-rho,xi)
    D(velocity, rhsPress)
    Minv(rhsPress,r)
    if mainOp.model.nu > 0:
        precon.clear()
        Pinv(rhsPress,precon)
        r *= mainOp.model.mu
        r.axpy(mainOp.model.nu,precon)
    oldDelta = delta
    delta = r.scalarProductDofs(rhsPress)
    print("delta:",delta,flush=True)
    if delta < 1e-9: break
    gamma = delta/oldDelta
    d *= gamma
    d += r
velocity.plot(colorbar="horizontal")
pressure.plot(colorbar="horizontal")
