from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

comm = MPI.comm_world
parameters['allow_extrapolation'] = True # I allow some points to appear outside the domain

scheme = "DG-UPW" # Only FE, DG-SIP-Sigmoidal, DG-UPW
variable = "w" # u, only w if scheme is DG-UPW

save_vtk = 0 # Save the approximation in pvd file

adv = 1.0
T = 0.001
nt = 1000

nx = 200 # Only 50, 100, 150 or 200
mesh1 = Mesh()
mesh2 = Mesh()
if adv>DOLFIN_EPS:
    with XDMFFile(comm,"meshes/mesh_circle_nx-500.xdmf") as infile:
        infile.read(mesh1)
    with XDMFFile(comm,"meshes/mesh_circle_nx-%d.xdmf" %(nx)) as infile:
        infile.read(mesh2)
else:
    with XDMFFile(comm,"meshes/mesh_square_nx-1000.xdmf") as infile:
        infile.read(mesh1)
    with XDMFFile(comm,"meshes/mesh_square_nx-%d.xdmf" %(nx)) as infile:
        infile.read(mesh2)

print("u: h = %f" % (mesh1.hmax()))
print("u_h: h = %f" % (mesh2.hmax()))

plot(mesh2)
plt.show()

P1 = FiniteElement("Lagrange", mesh1.ufl_cell(), 1)  # Space of polynomials
V1 = FunctionSpace(mesh1, P1)

if scheme == "DG-UPW" and variable == "u":
    deg = 0  # Degree of polynomials in discrete space
else:
    deg = 1
if scheme == "FE" or (scheme == "DG-UPW" and variable == "w"):
    P2 = FiniteElement("Lagrange", mesh2.ufl_cell(), deg)  # Space of polynomials
else:
    P2 = FiniteElement("DG", mesh2.ufl_cell(), deg)  # Space of polynomials
    P3 = FiniteElement("Lagrange",mesh2.ufl_cell(),1) # We will project the DG solution on a P1 continuous to compute H^1 norm
    V3 = FunctionSpace(mesh2, P3)
V2 = FunctionSpace(mesh2, P2)

u1 = Function(V1)
u2 = Function(V2)

if adv>0.0:
    if T==0.05:
        with XDMFFile("cahn_hilliard_advection_FE+Eyre/fig/T-0,05/sol_exact_2D_adv-1_P1_T-0,05_nt-1000_nx-500_eps-0,01/u_sol_FE_nt-1000_T-0.05_P1_adv-1.0_nx-500_counters-6.xdmf") as infile:
            infile.read_checkpoint(u1, "u", 5)
    else:
        with XDMFFile("cahn_hilliard_advection_FE+Eyre/fig/T-0,001/sol_exact_2D_adv-1_P1_T-0,001_nt-1000_nx-500_eps-0,01/u_sol_FE_nt-1000_T-0.001_P1_adv-1.0_nx-500_counters-6.xdmf") as infile:
            infile.read_checkpoint(u1, "u", 5)
else:
    if T==0.05:
        with XDMFFile("cahn_hilliard_advection_FE+Eyre/fig/T-0,05/sol_exact_2D_adv-0_P1_T-0,05_nt-1000_nx-1000_eps-0,01/u_sol_FE_nt-1000_T-0.05_P1_adv-0.0_nx-1000_counters-6.xdmf") as infile:
            infile.read_checkpoint(u1, "u", 5)
    else:
        with XDMFFile("cahn_hilliard_advection_FE+Eyre/fig/T-0,001/sol_exact_2D_adv-0_P1_T-0,001_nt-1000_nx-1000_eps-0,01/u_sol_FE_nt-1000_T-0.001_P1_adv-0.0_nx-1000_counters-6.xdmf") as infile:
            infile.read_checkpoint(u1, "u", 5)
if T==0.05:
    if scheme=="DG-UPW":
        with XDMFFile("cahn_hilliard_advection_%s+Eyre/fig/coupled/T-0,05/sol_2D_adv-%d_P0_T-0,05_nt-%d_nx-%d_eps-0,01/%s_sol_%s_nt-%d_T-0.05_P0_adv-%.1f_nx-%d_counters-5.xdmf" %(scheme, adv, nt, nx, variable, scheme, nt, adv, nx)) as infile:
            infile.read_checkpoint(u2, variable, 5)
    else:
        with XDMFFile("cahn_hilliard_advection_%s+Eyre/fig/T-0,05/sol_2D_adv-%d_P1_T-0,05_nt-%d_nx-%d_eps-0,01/u_sol_%s_nt-%d_T-0.05_P1_adv-%.1f_nx-%d_counters-6.xdmf" %(scheme, adv, nt, nx, scheme, nt, adv, nx)) as infile:
            infile.read_checkpoint(u2, "u", 5)
else:
    if scheme=="DG-UPW":
        with XDMFFile("cahn_hilliard_advection_%s+Eyre/fig/coupled/T-0,001/sol_2D_adv-%d_P0_T-0,001_nt-%d_nx-%d_eps-0,01/%s_sol_%s_nt-%d_T-0.001_P0_adv-%.1f_nx-%d_counters-5.xdmf" %(scheme, adv, nt, nx, variable, scheme, nt, adv, nx)) as infile:
            infile.read_checkpoint(u2, variable, 5)
    else:
        with XDMFFile("cahn_hilliard_advection_%s+Eyre/fig/T-0,001/sol_2D_adv-%d_P1_T-0,001_nt-%d_nx-%d_eps-0,01/u_sol_%s_nt-%d_T-0.001_P1_adv-%.1f_nx-%d_counters-6.xdmf" %(scheme, adv, nt, nx, scheme, nt, adv, nx)) as infile:
            infile.read_checkpoint(u2, "u", 5)

if scheme == "FE" or (scheme == "DG-UPW" and variable == "w"):
    u_exact = project(u1,V2)
    u_aprox = u2
else:
    u_exact = project(u1,V3)
    u_aprox = project(u2,V3) # We project the DG solution on a P1 continuous space to compute H^1 norm


print("Error L^2 ||u-u_h|| = %.10f" % (sqrt(assemble(pow(u_exact-u_aprox,2)*dx))))
print("Error H^1 ||u-u_h|| = %.10f" % (sqrt(assemble(pow(u_exact-u_aprox,2)*dx)) + sqrt(assemble(dot(grad(u_exact-u2), grad(u_exact-u2))*dx))))
print("Error Linf ||u-u_h|| = %.10f" % (np.abs(u_exact.vector().get_local()-u_aprox.vector().get_local()).max()))

plt.set_cmap('coolwarm')
pic = plot(u1)
plt.title("$u$")
plt.colorbar(pic)
plt.show()
plt.close()

plt.set_cmap('coolwarm')
pic = plot(u2)
plt.title("$%s_h$" %(variable))
plt.colorbar(pic)
plt.show()
plt.close()

if save_vtk:
    if scheme=="DG-UPW":
        vtkfile = File("%s_sol_%s_nt-%d_nx-%d_T-%.3f_P0_adv-%.1f_nx-%d.pvd" %(variable, scheme, nt, nx, T, adv, nx))
    else:
        vtkfile = File("u_sol_%s_nt-%d_nx-%d_T-%.3f_P0_adv-%.1f_nx-%d.pvd" %(scheme, nt, nx, T, adv, nx))
    vtkfile << (u2, T)
