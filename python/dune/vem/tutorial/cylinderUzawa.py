import pygmsh
import dune.fem, dune.vem
from uzawa import Uzawa # saddle point solver
from dune.ufl import Constant, DirichletBC
from ufl import SpatialCoordinate, TrialFunction, TestFunction,\
                inner, dot, div, nabla_grad, dx, as_vector, sym

dune.fem.threading.use = 8
coarse = True
order  = 4
T      = 2.5
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
print(gridView.size(0),flush=True)

# set up spaces for velocity and pressure
spcU = dune.vem.divFreeSpace( gridView, order=order)
spcP = dune.fem.space.finiteVolume( gridView )
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
dbc_p_out    = DirichletBC(spcP, 0, 2.2 - x[0] <= 1e-8)

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
              tolerance=1e-6, precondition=True, verbose=False)

# time loop
if coarse:
    gridView.writeVTK("cylinder_grid", pointdata=[p_h], pointvector=[u_h])
    vtk = gridView.sequencedVTK("cylinder",
                  subsampling=order, pointdata=[p_h], pointvector=[u_h])
else:
    gridView.writeVTK("cylinderFine_grid", pointdata=[p_h],pointvector=[u_h])
    vtk = gridView.sequencedVTK("cylinderFine",
                  subsampling=order, pointdata=[p_h], pointvector=[u_h])
vtk()
time, saveStep = 0, 0.05
nextSave = saveStep
while time < T:
    u_h_n.assign(u_h)
    info = uzawa.solve([u_h,p_h])
    time += tau
    print(time, tau, info, flush=True)
    if time > nextSave:
        print("  *********** saving ********  ")
        vtk()
        nextSave += saveStep
