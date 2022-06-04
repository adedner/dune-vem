# %% [markdown]
# # 2D navier stokes around a cylinder
#
# https://hplgit.github.io/fenics-tutorial/pub/sphinx1/._ftut1004.html#flow-past-a-cylinder
# OMP_NUM_THREADS=4 python cylinderSplit.py

# %%
import matplotlib

matplotlib.rc("image", cmap="jet")
from scipy.sparse import linalg
from math import sqrt
import matplotlib.pyplot as plt
import pygmsh
from dune.ufl import Constant, DirichletBC
import dune.fem.space
import dune.fem.scheme
import dune.vem
from uzawa import dgLaplace, plot

useVem     = True
pressureFV = True
order      = 2

T = 5.0                              # final time
# tau = Constant(T / 10000, "tau")     # fine
tau = Constant(T / 4000, "tau")      # coarse (works with p=2 fails with p=4)
# tau = Constant(T / 5000, "tau")      # coarse (works with p=4)

mu  = Constant(0.001, "mu")          # kinematic viscosity
rho = Constant(1, "rho")             # density

with pygmsh.occ.Geometry() as geom:
    # geom.characteristic_length_max = 0.02
    geom.set_mesh_size_callback( lambda dim, tag, x, y, z, lc:
          min(0.04+(x-0.2)**2+(y-0.2)**2 - 0.1**2, 0.1) )      # coarse
          # min(0.01+(x-0.2)**2+(y-0.2)**2 - 0.1**2, 0.1) )      # fine
    rectangle = geom.add_rectangle([0, 0, 0], 2.2, 0.41)
    cylinder = geom.add_disk([0.2, 0.2, 0.0], 0.05)
    geom.boolean_difference(rectangle, cylinder)

    mesh = geom.generate_mesh()
    points, cells = mesh.points, mesh.cells_dict
    domain = {
        "vertices": points[:, :2].astype(float),
        "simplices": cells["triangle"].astype(int),
    }
    print(len(domain["simplices"]))

gridView = dune.vem.polyGrid(domain)
# gridView.plot()
# plt.show()

if useVem:
    path="splitVemNC_p" + str(order)
    space_u = dune.vem.divFreeSpace( gridView, order=order, conforming=False)
    if pressureFV:
        # space_p = dune.fem.space.finiteVolume( gridView )
        # FIX ERROR:
        space_p = dune.vem.bbdgSpace( gridView, order=0 )
    else:
        space_p = dune.fem.space.lagrange( gridView, order=1 )
    scheme = lambda model, gStab, mStab: dune.vem.vemScheme(
                  model,
                  solver=("suitesparse","umfpack"),
                  # parameters={"newton.verbose":True, "newton.linear.verbose":True},
                  gradStabilization=gStab,
                  massStabilization=mStab )
else:
    path="splitTH_p" + str(order)
    space_u = dune.fem.space.lagrange( gridView, order=order, dimRange=2)
    space_p = dune.fem.space.lagrange( gridView, order=order-1)
    pressureFV = False
    scheme = lambda model, gStab, mStab: dune.fem.scheme.galerkin(
                  model,
                  solver=("suitesparse","umfpack"),
                  # parameters={"newton.verbose":True, "newton.linear.verbose":True}
                  )


# %%
from ufl import (
    FacetNormal,
    Identity,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    conditional,
    div,
    dot,
    ds,
    dx,
    inner,
    nabla_grad,
    sqrt,
    sym,
    triangle,
)

u = TrialFunction(space_u)
v = TestFunction(space_u)
p = TrialFunction(space_p)
q = TestFunction(space_p)

u_h = space_u.interpolate([0, 0], name="u_h")
u_h_n = u_h.copy(name="u_h_n")

p_h = space_p.interpolate(0, name="p_h")
p_h_n = p_h.copy(name="p_h_n")

# %%
x = SpatialCoordinate(triangle)

inflow = [6 * x[1] * (0.41 - x[1]) / 0.41**2, 0] # 6->18
dbc_u_in = DirichletBC(space_u, inflow, x[0] <= 1e-8)
dbc_u_noflow = DirichletBC(space_u, [0, 0], None)
dbc_u_out = DirichletBC(space_u, [None,None], 2.2 - x[0] <= 1e-8)

dbc_p_out = DirichletBC(space_p, 0, 2.2 - x[0] <= 1e-8)

# %%
# Define strain-rate tensor
def epsilon(u):
    return sym(nabla_grad(u))

# Define stress tensor
def sigma(u, p):
    return 2 * mu * epsilon(u) - p * Identity(len(u))

# Define expressions used in variational forms
U = 0.5 * (u_h_n + u)
n = FacetNormal(space_u)
f = Constant((0, 0), "f")

# %%

E1 = (
    rho * dot((u - u_h_n) / tau, v) * dx
    + rho * dot(dot(u_h_n, nabla_grad(u_h_n)), v) * dx
    + inner(sigma(U, p_h_n), epsilon(v)) * dx
    # - dot(( mu * nabla_grad(U)-p_h_n ) * n, v) * ds
    - rho * dot(f, v) * dx
) == 0

scheme1 = scheme( [E1, dbc_u_in,dbc_u_out,dbc_u_noflow],
                  gStab=2*mu, mStab=rho/tau )

if pressureFV:
    # dbc_p_out = DirichletBC(space_p, 0, 2.2 - x[0] <= 1e-8)
    laplace_p   = dgLaplace(10, p-p_h_n,q, space_p, 0, conditional(2.2-x[0]<=1e-8,1,0))
    E2 = laplace_p + (1 / tau) * div(u_h_n) * q * dx == 0
    scheme2 = dune.fem.scheme.galerkin(E2,
                # solver="cg",
                solver=("suitesparse","umfpack"),
                # parameters={"newton.linear.verbose":True}
                )
    E3 = (
        dot(u, v) * dx
        - dot(u_h_n, v) * dx
        - tau * (p_h - p_h_n) * div(v) * dx
    ) == 0
    # add bnd term after moving grad(p-p_n)v to (p-p_n)div(v)
    scheme3 = scheme( [E3, dbc_u_in,dbc_u_out,dbc_u_noflow],
                      gStab=None, mStab=1)
else:
    E2 = (
        dot(nabla_grad(p-p_h_n), nabla_grad(q)) * dx
        + (1 / tau) * div(u_h_n) * q * dx
    ) == 0
    scheme2 = dune.fem.scheme.galerkin([E2, dbc_p_out],
                # solver="cg",
                solver=("suitesparse","umfpack"),
                # parameters={"newton.linear.verbose":False}
                )
    E3 = (
        dot(u, v) * dx
        - dot(u_h_n, v) * dx
        + tau * dot(nabla_grad(p_h - p_h_n), v) * dx # can we move to p div(v)
    ) == 0
    scheme3 = scheme( [E3, dbc_u_in,dbc_u_out,dbc_u_noflow],
                      gStab=None, mStab=1)

solve = lambda A,b: linalg.spsolve(A,b)
A1    = dune.fem.operator.linear(scheme1).as_numpy
A1inv = lambda rhs: solve(A1,rhs.as_numpy[:])
A2    = dune.fem.operator.linear(scheme2).as_numpy
A2inv = lambda rhs: solve(A2,rhs.as_numpy[:])
A3    = dune.fem.operator.linear(scheme3).as_numpy
A3inv = lambda rhs: solve(A3,rhs.as_numpy[:])
zero_u = space_u.interpolate([0,0],name="zero")
zero_p = space_p.interpolate(0,name="zero")
rhs_u  = zero_u.copy(name="rhs")
rhs_p  = zero_p.copy(name="rhs")

# %%
from pathlib import Path
p = Path(path)
p.mkdir(exist_ok=True)
vtk = gridView.sequencedVTK(path+f"/2DCylinder",
              subsampling=3,
              pointdata=[p_h], pointvector=[u_h])
vtk()

time = 0
saveStep = 0.05
nextSave = saveStep

while time < T:
    u_h_n.assign(u_h)
    p_h_n.assign(p_h)
    scheme1(zero_u,rhs_u)
    rhs_u.as_numpy[:] *= -1
    scheme1.setConstraints(rhs_u)
    u_h.as_numpy[:] = A1inv(rhs_u)
    u_h_n.assign(u_h)

    scheme2(zero_p,rhs_p)
    rhs_p.as_numpy[:] *= -1
    try:
        scheme2.setConstraints(rhs_p)
    except AttributeError:
        pass
    p_h.as_numpy[:] = A2inv(rhs_p)

    scheme3(zero_u,rhs_u)
    rhs_u.as_numpy[:] *= -1
    scheme3.setConstraints(rhs_u)
    u_h.as_numpy[:] = A3inv(rhs_u)

    time += tau.value
    if time > nextSave:
        print(time, flush=True)
        vtk()
        nextSave += saveStep

# %%
