# FEniCSx version: 0.6

from dolfin import *

circle = 0
big_square = 1

if circle:
    nx = 10 # Triangles
    # print("Triangles = %d" % (nx))
    # meshdegree = 1
    # dim = 2
    # mesh = UnitDiscMesh.create(comm, nx, meshdegree, dim)
    domain = Circle(Point(0,0),1)
    mesh = generate_mesh(domain,nx)
elif big_square:
    nx = ny = 200
    mesh = RectangleMesh(Point(-10,10), Point(10, -10), nx, ny, "right/left")
else:
    nx = ny = 50
    mesh = RectangleMesh(Point(-1,1), Point(1, -1), nx, ny, "right/left")

print("h = %f" % (mesh.hmax()))

if circle:
    with XDMFFile("./mesh_circle_nx-%d.xdmf" %(nx)) as outfile:
        outfile.write(mesh)
elif big_square:
    with XDMFFile("./mesh_big_square_nx-%d.xdmf" %(nx)) as outfile:
        outfile.write(mesh)
else:
    with XDMFFile("./mesh_square_nx-%d.xdmf" %(nx)) as outfile:
        outfile.write(mesh)
