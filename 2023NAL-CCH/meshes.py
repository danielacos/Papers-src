# FEniCS version: 2019.1.0

from dolfin import *
import matplotlib.pyplot as plt
from mshr import *

adv = 1 # Indicates if there is advection or not

if adv:
    nx = 10 # Triangles
    # print("Triangles = %d" % (nx))
    # meshdegree = 1
    # dim = 2
    # mesh = UnitDiscMesh.create(comm, nx, meshdegree, dim)
    domain = Circle(Point(0,0),1)
    mesh = generate_mesh(domain,nx)
else:
    nx = ny = 200
    mesh = RectangleMesh(Point(0,1), Point(1, 0), nx, ny, "right/left")

print("h = %f" % (mesh.hmax()))

if adv:
    with XDMFFile("./mesh_circle_nx-%d.xdmf" %(nx)) as outfile:
        outfile.write(mesh)
else:
    with XDMFFile("./mesh_square_nx-%d.xdmf" %(nx)) as outfile:
        outfile.write(mesh)
