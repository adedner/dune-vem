from pathlib import Path
import pygmsh
import argparse
from pathlib import Path
import dune.fem, dune.vem
from uzawa import Uzawa # saddle point solver
from dune.ufl import Constant, DirichletBC
from ufl import SpatialCoordinate, TrialFunction, TestFunction,\
                inner, dot, div, nabla_grad, dx, as_vector, sym

dune.fem.threading.use = 8
parser = argparse.ArgumentParser(description='set order and spaces')
parser.add_argument('-l', type=int, required=True, help="velocity order >= 2")
parser.add_argument('-s', type=str, required=True, help="use vem or th")
parser.add_argument('-g', type=int, required=True, help="0: coarse grid, 1: fine grid")

args = parser.parse_args()
useVem = not args.s == "th"
coarse = args.g == 0
order  = args.l

T      = 5.0
tau    = T / 8000
mu     = 0.001

# use pygmsh to define domain with cylinder
with pygmsh.occ.Geometry() as geom:
    geom.set_mesh_size_callback( lambda dim, tag, x, y, z, lc:
        min(0.02+1.5*( (x-0.2)**2+(y-0.2)**2), 0.2 if coarse else 0.04)
    )
    rectangle = geom.add_rectangle([0, 0, 0], 2.2, 0.41)
    cylinder = geom.add_disk([0.2, 0.2, 0.0], 0.05)
    geom.boolean_difference(rectangle, cylinder)
    mesh = geom.generate_mesh()
    points, cells = mesh.points, mesh.cells_dict
    # extract dictionary for grid construction
    domain = {
        "vertices": points[:, :2].astype(float),
        "simplices": cells["triangle"].astype(int),
    }
gridView = dune.vem.polyGrid(domain)
print("grid size:", gridView.size(0),flush=True)

# set up spaces for velocity and pressure
if useVem:
    spcU = dune.vem.divFreeSpace( gridView, order=order )
    spcP = dune.fem.space.finiteVolume( gridView )
else:
    spcU = dune.fem.space.lagrange( gridView, order=order, dimRange=2 )
    spcP = dune.fem.space.lagrange( gridView, order=order-1 )

u_h   = spcU.interpolate([0, 0], name="u_h")
u_h_n = spcU.interpolate([0, 0], name="u_h")
p_h   = spcP.interpolate(0, name="p_h")

# set up models for saddle point problem
x            = SpatialCoordinate(spcU)
mu,nu        = Constant(mu, "mu"), Constant(1/tau, "nu")
inflow       = [6 * x[1] * (0.41 - x[1]) / 0.41**2, 0] # 6->18
dbc_u_in     = DirichletBC(spcU, inflow, x[0] <= 1e-8)
dbc_u_noflow = DirichletBC(spcU, [0, 0], None)
dbc_u_out    = DirichletBC(spcU, [None,None], 2.2 - x[0] <= 1e-8)

u,v   = TrialFunction(spcU), TestFunction(spcU)
p,q   = TrialFunction(spcP), TestFunction(spcP)
b         = lambda u1,u2,v: dot( dot(u1, nabla_grad(u2)), v)
epsilon   = lambda w: sym(nabla_grad(w))
a         = lambda u,v: inner(epsilon(u), epsilon(v))
mainModel = ( nu*dot(u-u_h_n,v) + b(u_h_n,u_h_n,v) + 2*mu*a(u,v) ) * dx
gradModel = -p*div(v) * dx
divModel  = -div(u)*q * dx

# use a Uzawa type algorithm to solve the problem
uzawa = Uzawa(gridView, spcU, spcP,
              [dbc_u_in,dbc_u_out,dbc_u_noflow],
              mainModel, gradModel, divModel, mu, nu,
              tolerance=1e-6,
              verbose=False,
              precondDbnd=[2.2 - x[0] <= 1e-8],
              lagrange=not useVem)

# time loop
if useVem:
    path="cylinderVem_p" + str(order)
    p = Path(path)
    p.mkdir(exist_ok=True)
else:
    path="cylinderTH_p" + str(order)
    p = Path(path)
    p.mkdir(exist_ok=True)

if coarse:
    gridView.writeVTK("cylinder", pointdata=[p_h], pointvector=[u_h])
    vtk = gridView.sequencedVTK(path+f"/2DCylinder",
                  subsampling=order, pointdata=[p_h], pointvector=[u_h])
else:
    gridView.writeVTK("cylinderFine", pointdata=[p_h],pointvector=[u_h])
    vtk = gridView.sequencedVTK(path+f"/2DCylinderFine",
                  subsampling=order, pointdata=[p_h], pointvector=[u_h])
vtk()
time, saveStep = 0, 0.05
nextSave = saveStep
while time < T:
    u_h_n.assign(u_h)
    info = uzawa.solve([u_h,p_h])
    time += tau
    if time > nextSave:
        print(time, info, flush=True)
        vtk()
        nextSave += saveStep
