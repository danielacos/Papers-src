# FEniCS version: 2019.1.0

from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

comm = MPI.comm_world
parameters['allow_extrapolation'] = True # I allow some points to appear outside the domain

test = "blow-up"
# test = "three-balls_14"
eps = "1e-14"
variable = "v" # u or v

save_vtk = 0 # Save the approximation in pvd file

func_1 = "test-9_nx-1000_T-0,0001_eps-0/"
if eps == "1e-10":
    func_2 = "test-9_nx-1000_T-0,0001/"
else:
    func_2 = "test-9_nx-1000_T-0,0001_eps-" + eps + "/"
nx = 1000
counter = 5
# elif test=="three-balls_14":
#     func_1 = "non_truncated_u/three-balls/test-14-Saito-good-tau-0/"
#     func_2 = "truncated_u/three-balls/test-14-Saito-good-tau-0/"
#     nx = 50
#     if variable == "v": counter = 14
#     else:  counter = 15

print("TEST: " + test)

mesh = Mesh()
with XDMFFile(comm,"../../meshes/mesh_square_nx-%d.xdmf" %(nx)) as infile:
    infile.read(mesh)

# print("u: h = %f" % (mesh1.hmax()))
print("h = %f" % (mesh.hmax()))

# plot(mesh)
# plt.show()

if variable == "u":
    P = FiniteElement("DG", mesh.ufl_cell(), 0)  # Space of polynomials
else:
    P = FiniteElement("Lagrange", mesh.ufl_cell(), 1)  # Space of polynomials
V = FunctionSpace(mesh, P)

u1 = Function(V)
u2 = Function(V)

with XDMFFile(func_1 + variable + "_ks_DG-UPW.xdmf") as infile:
    infile.read_checkpoint(u1, variable, counter)
with XDMFFile(func_2 + variable + "_ks_DG-UPW.xdmf") as infile:
    infile.read_checkpoint(u2, variable, counter)

# if variable == "w"):
    # u_exact = project(u1,V2)
    # u_aprox = u2
# else:
#     u_exact = project(u1,V3)
#     u_aprox = project(u2,V3) # We project the DG solution on a P1 continuous space to compute H^1 norm


print(f"Error L^2 ||{variable}_1-{variable}_2|| = {sqrt(assemble(pow(u1-u2,2)*dx)):.2e}")
# print("Error H^1 ||u-u_h|| = %.10f" % (sqrt(assemble(pow(u_exact-u_aprox,2)*dx)) + sqrt(assemble(dot(grad(u_exact-u2), grad(u_exact-u2))*dx))))
print(f"Error Linf ||{variable}_1-{variable}_2|| = {np.abs(u1.vector().get_local()-u2.vector().get_local()).max():.2e}")
print(f"Min {variable}_1-{variable}_2 = {np.abs(u1.vector().get_local()-u2.vector().get_local()).min():.2e}")

# plt.set_cmap('coolwarm')
# pic = plot(u1)
# plt.title("$u_1$")
# plt.colorbar(pic)
# plt.show()
# plt.close()

# plt.set_cmap('coolwarm')
# pic = plot(u2)
# plt.title("$u_2$" %(variable))
# plt.colorbar(pic)
# plt.show()
# plt.close()

# if save_vtk:
#     if scheme=="DG-UPW":
#         vtkfile = File("%s_sol_%s_nt-%d_nx-%d_T-%.3f_P0_adv-%.1f_nx-%d.pvd" %(variable, scheme, nt, nx, T, adv, nx))
#     else:
#         vtkfile = File("u_sol_%s_nt-%d_nx-%d_T-%.3f_P0_adv-%.1f_nx-%d.pvd" %(scheme, nt, nx, T, adv, nx))
#     vtkfile << (u2, T)
