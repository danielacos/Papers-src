from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from numpy import log
import numpy as np

comm = MPI.comm_world
# parameters['allow_extrapolation'] = True # I allow some points to appear outside the domain

variable = "u" # u
counter = [1, 2, 5]
nx = [100, 140, 180, 220]
# h = [2, 3/2, 4/3]

save_vtk = 0 # Save the approximation in pvd file

mesh1 = Mesh()
mesh2 = Mesh()

with XDMFFile(comm,"../../meshes/mesh_square_nx-2000.xdmf") as infile:
    infile.read(mesh1)
if variable == "u":
    P1 = FiniteElement("DG", mesh1.ufl_cell(), 0)  # Space of polynomials
else:
    P1 = FiniteElement("Lagrange", mesh1.ufl_cell(), 1)  # Space of polynomials
V1 = FunctionSpace(mesh1, P1)
u1 = Function(V1)

l2_global = []
alt_global = []
for i in counter:
    l2 = []
    alt = []
    h = []
    print("=========================================")
    for n in nx:
        with XDMFFile(comm,"../../meshes/mesh_square_nx-%d.xdmf" %(n)) as infile:
            infile.read(mesh2)
        h.append(mesh2.hmax())

        print("Counter: "  + str(i))
        print(variable + ": h = %f" % (mesh1.hmax()))
        print(variable + "_h: h = %f" % (h[-1]))

        # plot(mesh2)
        # plt.show()

        if variable == "u":
            P2 = FiniteElement("DG", mesh2.ufl_cell(), 0)  # Space of polynomials
        else:
            P2 = FiniteElement("Lagrange", mesh2.ufl_cell(), 1)  # Space of polynomials
        V2 = FunctionSpace(mesh2, P2)

        u2 = Function(V2)

        with XDMFFile("test-9_nx-2000_T-0,0001/" + variable + "_ks_DG-UPW.xdmf") as infile:
            infile.read_checkpoint(u1, variable, i)
        with XDMFFile(f"test-9_nx-{n}_T-0,0001/" + variable + "_ks_DG-UPW.xdmf") as infile:
            infile.read_checkpoint(u2, variable, i)

        u_exact = project(u1,V2)
        u_aprox = u2

        l2.append(sqrt(assemble(pow(u_exact-u_aprox,2)*dx)))

        print(f"Error L^2 ||{variable}-{variable}_h|| = {l2[-1]:.3e}")
        if variable == "u":
            alt.append(np.abs(u_exact.vector().get_local()-u_aprox.vector().get_local()).max())
            print("Error Linf ||u-u_h|| = %.3e" %(alt[-1]))
        else:
            alt.append(sqrt(assemble(pow(u_exact-u_aprox,2)*dx) + assemble(dot(grad(u_exact-u_aprox), grad(u_exact-u_aprox))*dx)))
            print("Error H^1 ||v-v_h|| = %.3e" % (alt[-1]))

    print("Order L2: ")
    for k in range(len(h)-1):
        print(f"{log(l2[k]/l2[k+1])/log(h[k]/h[k+1]):.3e}")

    print("Order Linf/H^1: ")
    for k in range(len(h)-1):
        print(f"{log(alt[k]/alt[k+1])/log(h[k]/h[k+1]):.3e}")

    l2_global.append(l2)
    alt_global.append(alt)

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(h, 100*np.array(h), color='blue', lw=2)
# if variable == "u":
    # ax.plot(h, 100 * np.array(h), color='blue', lw=2)
    # ax.plot(h, 20 * np.array(h)**(1/2), color='blue', lw=2)
# else:
    # ax.plot(h, 5 * np.array(h), color='blue', lw=2)
    # ax.plot(h, 20 * np.array(h)**(1/2), color='blue', lw=2)
# col = ['red', 'black', 'green']

# for i in range(len(l2_global)):
    # ax.plot(h, l2_global[i], color = col[i], ls='-.')
    # ax.plot(h, alt_global[i], color = col[i], ls=':')

# ax.set_yscale('log')
# ax.set_xscale('log')

# plt.show()

# plt.set_cmap('coolwarm')
# pic = plot(u1)
# plt.title("$u$")
# plt.colorbar(pic)
# plt.show()
# plt.close()

# plt.set_cmap('coolwarm')
# pic = plot(u2)
# plt.title("$%s_h$" %(variable))
# plt.colorbar(pic)
# plt.show()
# plt.close()

# if save_vtk:
#     if scheme=="DG-UPW":
#         vtkfile = File("%s_sol_%s_nt-%d_nx-%d_T-%.3f_P0_adv-%.1f_nx-%d.pvd" %(variable, scheme, nt, nx, T, adv, nx))
#     else:
#         vtkfile = File("u_sol_%s_nt-%d_nx-%d_T-%.3f_P0_adv-%.1f_nx-%d.pvd" %(scheme, nt, nx, T, adv, nx))
#     vtkfile << (u2, T)
