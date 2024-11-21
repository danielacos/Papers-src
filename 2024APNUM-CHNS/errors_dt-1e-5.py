# FEniCSx version: 0.6

from mpi4py import MPI
import dolfinx
from ufl import FiniteElement, VectorElement, inner, dx, grad, sym
from dolfinx.fem import FunctionSpace, Function, assemble_scalar, form
import dolfinx.plot as plot
import adios4dolfinx as adx
import matplotlib as mpl
import pyvista
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
dx_ML = dx(scheme="vertex", metadata={"degree":1, "representation":"quadrature"}) # mass lumped terms

pyvista.OFF_SCREEN = False
do_plot = False

def printMPI(string, end='\n'):
    if rank == 0:
        print(string, end=end)

printMPI("\nCoupled errors:")

mesh_ex = adx.read_mesh(comm, "coupled/coupled_order_nx-200_dt-1e-5_delta-1e-6_p-1e-10/NSCH_DG-UPW_coupled_order_mesh", engine='BP4', ghost_mode=dolfinx.mesh.GhostMode.none)

topology, cell_types, geometry = plot.create_vtk_mesh(mesh_ex, mesh_ex.topology.dim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True, color="white")
plotter.view_xy()
if pyvista.OFF_SCREEN:
    plotter.screenshot(f"./mesh.png", transparent_background=True)
elif do_plot:
    plotter.show()
plotter.close()

P1c_ex = FiniteElement("Lagrange", mesh_ex.ufl_cell(), 1)
P0d_ex = FiniteElement("DG", mesh_ex.ufl_cell(), 0)
P1d_ex = FiniteElement("DG", mesh_ex.ufl_cell(), 1)
P2b_ex = VectorElement(FiniteElement("Lagrange", mesh_ex.ufl_cell(), 2) + FiniteElement("Bubble", mesh_ex.ufl_cell(), 3))

P1cs_ex = FunctionSpace(mesh_ex, P1c_ex)
P0ds_ex = FunctionSpace(mesh_ex, P0d_ex)
P1ds_ex = FunctionSpace(mesh_ex, P1d_ex)
P2bs_ex = FunctionSpace(mesh_ex, P2b_ex)

p1c_phi_ex = Function(P1cs_ex)
phi_ex = Function(P0ds_ex)
u_ex = Function(P2bs_ex)
p_ex = Function(P1ds_ex)

adx.read_function(p1c_phi_ex, "coupled/coupled_order_nx-200_dt-1e-5_delta-1e-6_p-1e-10/NSCH_DG-UPW_coupled_order_p1c_phi_i-50")
adx.read_function(phi_ex, "coupled/coupled_order_nx-200_dt-1e-5_delta-1e-6_p-1e-10/NSCH_DG-UPW_coupled_order_phi_i-50")
adx.read_function(u_ex, "coupled/coupled_order_nx-200_dt-1e-5_delta-1e-6_p-1e-10/NSCH_DG-UPW_coupled_order_u_i-50")
adx.read_function(p_ex, "coupled/coupled_order_nx-200_dt-1e-5_delta-1e-6_p-1e-10/NSCH_DG-UPW_coupled_order_p_i-50")

# Properties of the scalar bar
sargs = dict(height=0.6, vertical=True, position_x=0.8, position_y=0.2, title='', label_font_size=24, shadow=True,n_labels=5, fmt="%.2f", font_family="arial")

# Create a grid to attach the DoF values
topology, cell_types, geometry = plot.create_vtk_mesh(P1cs_ex)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
aux = p1c_phi_ex.x.array
aux[np.abs(aux + 1.0) < 1e-16] = -1.0
aux[np.abs(aux - 1.0) < 1e-16] = 1.0
grid.point_data["Pi1_phi"] = aux

grid.set_active_scalars("Pi1_phi")

plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=False, show_scalar_bar=True, cmap=mpl.colormaps["coolwarm"], scalar_bar_args=sargs)
plotter.view_xy()

if pyvista.OFF_SCREEN:
    plotter.screenshot(f"./Pi1_phi_i-50.png", transparent_background=True)
elif do_plot:
    plotter.show()
plotter.close()

for file in ["coupled/coupled_order_nx-40_dt-1e-5_delta-1e-6_p-1e-10", "coupled/coupled_order_nx-60_dt-1e-5_delta-1e-6_p-1e-10", "coupled/coupled_order_nx-80_dt-1e-5_delta-1e-6_p-1e-10", "coupled/coupled_order_nx-105_dt-1e-5_delta-1e-6_p-1e-10", "coupled/coupled_order_nx-120_dt-1e-5_delta-1e-6_p-1e-10"]:

    mesh_approx = adx.read_mesh(comm, file+"/NSCH_DG-UPW_coupled_order_mesh", engine='BP4', ghost_mode=dolfinx.mesh.GhostMode.none)

    topology, cell_types, geometry = plot.create_vtk_mesh(mesh_approx, mesh_approx.topology.dim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True, color="white")
    plotter.view_xy()
    if pyvista.OFF_SCREEN:
        plotter.screenshot(f"./mesh.png", transparent_background=True)
    elif do_plot:
        plotter.show()
    plotter.close()

    P1c_approx = FiniteElement("Lagrange", mesh_approx.ufl_cell(), 1)
    P0d_approx = FiniteElement("DG", mesh_approx.ufl_cell(), 0)
    P1d_approx = FiniteElement("DG", mesh_approx.ufl_cell(), 1)
    P2b_approx = VectorElement(FiniteElement("Lagrange", mesh_approx.ufl_cell(), 2) + FiniteElement("Bubble", mesh_approx.ufl_cell(), 3))

    P1cs_approx = FunctionSpace(mesh_approx, P1c_approx)
    P0ds_approx = FunctionSpace(mesh_approx, P0d_approx)
    P1ds_approx = FunctionSpace(mesh_approx, P1d_approx)
    P2bs_approx = FunctionSpace(mesh_approx, P2b_approx)

    p1c_phi_approx = Function(P1cs_approx)
    phi_approx = Function(P0ds_approx)
    u_approx = Function(P2bs_approx)
    p_approx = Function(P1ds_approx)

    adx.read_function(p1c_phi_approx, file+"/NSCH_DG-UPW_coupled_order_p1c_phi_i-50")
    adx.read_function(phi_approx, file+"/NSCH_DG-UPW_coupled_order_phi_i-50")
    adx.read_function(u_approx, file+"/NSCH_DG-UPW_coupled_order_u_i-50")
    adx.read_function(p_approx, file+"/NSCH_DG-UPW_coupled_order_p_i-50")

    # Properties of the scalar bar
    sargs = dict(height=0.6, vertical=True, position_x=0.8, position_y=0.2, title='', label_font_size=24, shadow=True,n_labels=5, fmt="%.2f", font_family="arial")

    # Create a grid to attach the DoF values
    topology, cell_types, geometry = plot.create_vtk_mesh(P1cs_approx)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    aux = p1c_phi_approx.x.array
    aux[np.abs(aux + 1.0) < 1e-16] = -1.0
    aux[np.abs(aux - 1.0) < 1e-16] = 1.0
    grid.point_data["Pi1_phi"] = aux

    grid.set_active_scalars("Pi1_phi")

    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=False, show_scalar_bar=True, cmap=mpl.colormaps["coolwarm"], scalar_bar_args=sargs)
    plotter.view_xy()

    if pyvista.OFF_SCREEN:
        plotter.screenshot(f"./Pi1_phi_i-50.png", transparent_background=True)
    elif do_plot:
        plotter.show()
    plotter.close()

    p1c_phi_ex_interp = Function(P1cs_approx)
    p1c_phi_ex_interp.interpolate(p1c_phi_ex)

    phi_ex_interp = Function(P0ds_approx)
    phi_ex_interp.interpolate(phi_ex)

    u_ex_interp = Function(P2bs_approx)
    u_ex_interp.interpolate(u_ex)

    p_ex_interp = Function(P1ds_approx)
    p_ex_interp.interpolate(p_ex)

    tdim = mesh_approx.topology.dim
    h = dolfinx.cpp.mesh.h(mesh_approx, tdim, range(mesh_approx.topology.index_map(tdim).size_local))
    printMPI(f"\n  h = {comm.allreduce(max(h)):.2e}:")

    printMPI(f"     Error L2:")
    printMPI(f"         phi: {np.sqrt(assemble_scalar(form((phi_approx - phi_ex_interp)**2 * dx))):.2e}")
    printMPI(f"         p1c_phi: {np.sqrt(assemble_scalar(form((p1c_phi_approx - p1c_phi_ex_interp)**2 * dx))):.2e}")
    printMPI(f"         u: {np.sqrt(assemble_scalar(form(inner(u_approx - u_ex_interp, u_approx - u_ex_interp) * dx))):.2e}")
    printMPI(f"         p: {np.sqrt(assemble_scalar(form((p_approx - p_ex_interp)**2 * dx))):.2e}")

    printMPI(f"     Error H1:")
    printMPI(f"         p1c_phi: {np.sqrt(assemble_scalar(form((p1c_phi_approx - p1c_phi_ex_interp)**2 * dx + inner(grad(p1c_phi_approx - p1c_phi_ex_interp), grad(p1c_phi_approx - p1c_phi_ex_interp)) * dx))):.2e}")
    printMPI(f"         u: {np.sqrt(assemble_scalar(form(inner(u_approx - u_ex_interp, u_approx - u_ex_interp) * dx + inner(grad(u_approx - u_ex_interp), grad(u_approx - u_ex_interp)) * dx))):.2e}")
    printMPI(f"         p: {np.sqrt(assemble_scalar(form((p_approx - p_ex_interp)**2 * dx + inner(grad(p_approx - p_ex_interp), grad(p_approx - p_ex_interp)) * dx))):.2e}")

    printMPI(f"     Error Linf:")
    printMPI(f"         phi: {comm.allreduce(max(np.fabs(phi_approx.x.array - phi_ex_interp.x.array))): .2e}")