from dolfin import *
import matplotlib.pyplot as plt

circle = 0

if circle:
    nx = 10 # Triangles
    # print("Triangles = %d" % (nx))
    # meshdegree = 1
    # dim = 2
    # mesh = UnitDiscMesh.create(comm, nx, meshdegree, dim)
    domain = Circle(Point(0,0),1)
    mesh = generate_mesh(domain,nx)
else:
    nx = ny = 220
    mesh = RectangleMesh(Point(-1/2,1/2), Point(1/2, -1/2), nx, ny, "right/left")

print("h = %f" % (mesh.hmax()))

if circle:
    with XDMFFile("./mesh_circle_nx-%d.xdmf" %(nx)) as outfile:
        outfile.write(mesh)
else:
    with XDMFFile("./mesh_square_nx-%d.xdmf" %(nx)) as outfile:
        outfile.write(mesh)
