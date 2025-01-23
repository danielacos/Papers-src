# FEniCSx version: 0.6

from mpi4py import MPI
import dolfinx
from dolfinx import mesh
from dolfinx import plot
from dolfinx.io import gmshio, XDMFFile
import gmsh
import pyvista

nx = 100
mesh_index = 4
plot_mesh = 1

mesh_list = ["square", "cube", "circle", "sphere", "rectangle"]
mesh_type = mesh_list[mesh_index]

if mesh_type == "square":
    domain = mesh.create_rectangle(MPI.COMM_WORLD, points=((-0.5, -0.5), (0.5, 0.5)), n=(nx, nx), cell_type=mesh.CellType.triangle, diagonal=dolfinx.cpp.mesh.DiagonalType.right_left)
elif mesh_type=="cube":
    gmsh.initialize()

    # Create a cube in 3D
    cube = gmsh.model.occ.addBox(-0.5, -0.5, -0.5, 0.5, 0.5, 0.5)
    gmsh.model.occ.synchronize()

    # Creates the physical model grouping all the elements
    gdim = 3
    gmsh.model.addPhysicalGroup(gdim, [cube])

    # Create the mesh using gmsh
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin",1/nx)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax",1/nx)
    gmsh.model.mesh.generate(gdim)

    # Convert the gmsh model into a FEniCSx mesh
    gmsh_model_rank = 0
    mesh_comm = MPI.COMM_WORLD
    domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim)
elif mesh_type=="circle":
    gmsh.initialize()

    # Create a circle or ellipse (if rx>ry) in 3D
    circle = gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
    gmsh.model.occ.synchronize()

    # Creates the physical model grouping all the elements
    gdim = 2
    gmsh.model.addPhysicalGroup(gdim, [circle])

    # Create the mesh using gmsh
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin",1/nx)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax",1/nx)
    gmsh.model.mesh.generate(gdim)

    # Convert the gmsh model into a FEniCSx mesh
    gmsh_model_rank = 0
    mesh_comm = MPI.COMM_WORLD
    domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim)
elif mesh_type == "sphere":
    gmsh.initialize()

    # Create a sphere in 3D
    sphere = gmsh.model.occ.addSphere(0, 0, 0, 1.0)
    gmsh.model.occ.synchronize()

    # Creates the physical model grouping all the elements
    gdim = 3
    gmsh.model.addPhysicalGroup(gdim, [sphere])

    # Create the mesh using gmsh
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin",1/nx)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax",1/nx)
    gmsh.model.mesh.generate(gdim)

    # Convert the gmsh model into a FEniCSx mesh
    gmsh_model_rank = 0
    mesh_comm = MPI.COMM_WORLD
    domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim)
elif mesh_type == "rectangle":
    domain = mesh.create_rectangle(MPI.COMM_WORLD, points=((0, 0), (1, 4)), n=(nx, 4*nx), cell_type=mesh.CellType.triangle, diagonal=dolfinx.cpp.mesh.DiagonalType.right_left)

tdim = domain.topology.dim
num_cells = domain.topology.index_map(tdim).size_local
h = dolfinx.cpp.mesh.h(domain, tdim, range(num_cells))
print(f"h = {str(max(h))}")

# Save mesh
with XDMFFile(MPI.COMM_WORLD, f"./mesh_{mesh_type}_nx-{nx:d}.xdmf", "w") as outfile:
        outfile.write_mesh(domain)

# Plot mesh
if plot_mesh == 1:
    topology, cell_types, geometry = plot.create_vtk_mesh(domain, domain.topology.dim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True, color="white")
    plotter.view_xy()
    plotter.show()
