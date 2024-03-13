"""
Cahn-Hilliard equation with Neumann homogeneous conditions.

  u' = nabla(M_u * nabla(mu_u)) - div(u * v)    in the unit square
  mu_u = - epsilon^2 * Laplace(u) + F'(u)               in the unit square
  grad(u) * n = grad(mu_u) * n = 0                    on the boundary

where v is a vector valued function v(x,y)=adv*(y, -x), F(u) = Gamma * u^2 * (u-1)^2 if u\in[0,1], F(u) = Gamma * u^2 if u<0 and F(u) = Gamma * (1-u)^2 if u>1.

The mobility functions are M_u(u) = u(1-u).

We will comupute the energy functional

E = epsilon^2/2 * \int_\Omega |\nabla \phi|^2 + \int_\Omega F(u)

in each time step.

DG-SIP semidiscrete space scheme and Eyre semidiscrete time scheme
"""

from dolfin import *
import numpy as np
import random
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mshr import *

comm = MPI.comm_world
rank = comm.Get_rank()
parameters["ghost_mode"] = "shared_vertex" # Share the mesh between processors

T = 0.1           # final time
num_steps = 100     # number of time steps
dt = T / num_steps  # time step size

sigma = 4.0
kappa = 100.0

eps = 0.001
Gamma = 0.25
adv = 100.0
coef_Mov = 1.0

spinoidal_decomposition = 0 # Indicates if the initial condition is random
stokes = 0 # Indicates if we are using the Stokes velocity

showpic = 0 # Indicates if pictures are showed or not
savepic = 1  # Indicates if pictures are saved or not
savemax = 1 # Indicates if the maximum and minimum are saved or not
savesol = 1 # Indicates if solution is saved or not

if(rank==0):
    print("dt = %f" % (dt))

# Create mesh and define function space

# if adv==DOLFIN_EPS:
#     nx = ny = 20
#     mesh = RectangleMesh(Point(0,1), Point(1, 0), nx, ny, "right/left")
# else:
#     nx = 20 # Triangles
#     # print("Triangles = %d" % (nx))
#     # meshdegree = 1
#     # dim = 2
#     # mesh = UnitDiscMesh.create(comm, nx, meshdegree, dim)
#     domain = Circle(Point(0,0),1)
#     mesh = generate_mesh(domain,nx)

nx = 50
mesh = Mesh()
if stokes:
    with XDMFFile(comm,"../../stokes-cavity-test/mesh-stokes-cavity-test.xdmf") as infile:
        infile.read(mesh)
else:
    if adv>0.0:
        with XDMFFile(comm,"../meshes/mesh_circle_nx-%d.xdmf" %(nx)) as infile:
            infile.read(mesh)
    else:
        with XDMFFile(comm,"../meshes/mesh_square_nx-%d.xdmf" %(nx)) as infile:
            infile.read(mesh)

if showpic:
    plot(mesh)
    plt.show()

if(rank==0):
    print("h = %f" % (mesh.hmax()))

deg = 1  # Degree of polynomials in discrete space
P1 = FiniteElement("DG", mesh.ufl_cell(), deg)  # Space of polynomials
P2 = FiniteElement("DG", mesh.ufl_cell(), 2)  # Space of polynomials
W = FunctionSpace(mesh, MixedElement([P1, P1]))  # Space of functions
V1 = FunctionSpace(mesh, P1)
V2 = FunctionSpace(mesh, P2)

normal = FacetNormal(mesh)
h = CellDiameter(mesh)

# Definition of the convergence function
def ConvergenceFunction(iteration, v1, v0, abs_tol, rel_tol, convergence):
    absolute = sqrt(assemble(pow(v1, 2) * dx))
    if (iteration == 0):
        absolute0 = absolute
    else:
        absolute0 = v0
    relative = absolute / absolute0
    if absolute < abs_tol or relative < rel_tol:
        convergence = True
    return convergence, absolute, relative


# Random initial data
if spinoidal_decomposition:
    random.seed(1)
    class Init_u(UserExpression):
        def eval(self, values, x):
            values[0] = random.uniform(0.49, 0.51)

    init_u = Init_u(degree=2)  # Random data between 0.49 and 0.51
else:
    if adv>0.0:
        init_u = Expression("1.0/2.0*(tanh((0.2-sqrt(pow(x[0]+0.2,2)+pow(x[1],2)))/(sqrt(2.0)*eps))+1.0) + 1.0/2.0*(tanh((0.2-sqrt(pow(x[0]-0.2,2)+pow(x[1],2)))/(sqrt(2.0)*eps))+1.0)",eps=eps,degree=2)
    else:
        init_u = Expression("1.0/2.0*(tanh((0.2-sqrt(pow(x[0]-0.3,2)+pow(x[1]-0.5,2)))/(sqrt(2.0)*eps))+1.0) + 1.0/2.0*(tanh((0.2-sqrt(pow(x[0]-0.7,2)+pow(x[1]-0.5,2)))/(sqrt(2.0)*eps))+1.0)",eps=eps,degree=2)


u_0 = interpolate(init_u, V1)

plt.set_cmap('coolwarm')
c = plot(u_0)
plt.title("$u$ at $t = %.4f$" % (0.0))
plt.colorbar(c)
if savepic:
    if stokes:
        plt.savefig("fig/u_DG-SIP-Sigmoidal+Eyre_stokes_nt-%d_t-%.5f_P%d.png" %
                    (num_steps, 0.0, deg))
    else:
        plt.savefig("fig/u_DG-SIP-Sigmoidal+Eyre_nt-%d_t-%.5f_P%d_adv-%.1f_nx-%d.png" %
                    (num_steps, 0.0, deg, adv, nx))
if showpic:
    plt.show()
plt.close()

if(savesol):
    # Write `u` to a file:
    # fFile = HDF5File(MPI.comm_world,"fig/u_sol_FE_nt-%d_t-%.5f_P%d_adv-%.1f_nx-%d.h5" %(num_steps, 0.0, deg, adv, nx),"w")
    # fFile.write(u_0,"u_sol_FE")
    # fFile.close()
    if stokes:
        with XDMFFile("fig/u_sol_DG-SIP-Sigmoidal_stokes_nt-%d_T-%.3f_P%d_counters-5.xdmf" %(num_steps, T, deg)) as outfile:
            # outfile.write(mesh)
            counter  = 0
            outfile.write_checkpoint(u_0, "u", counter, append=False)
    else:
        with XDMFFile("fig/u_sol_DG-SIP-Sigmoidal_nt-%d_T-%.3f_P%d_adv-%.1f_nx-%d_counters-5.xdmf" %(num_steps, T, deg, adv, nx)) as outfile:
            # outfile.write(mesh)
            counter  = 0
            outfile.write_checkpoint(u_0, "u", counter, append=False)

# max_u_local = u_0.vector().get_local().max()
# if(rank==0):
#     print("Local: %f" %(max_u_local))

max_u_vector = []
min_u_vector = []
max_u = u_0.vector().max()
min_u = u_0.vector().min()
max_u_vector.append(max_u)
min_u_vector.append(min_u)

mass_u = assemble(u_0 * dx)

if(rank==0):
    print('max(u) = %f' % (max_u))
    print('min(u) = %f' % (min_u))
    print('mass_u = %f' % (mass_u))

# Define advection vector v
if stokes:
    V2_vector = VectorFunctionSpace(mesh, "P", 2)
    v = Function(V2_vector)
    with XDMFFile("../../stokes-cavity-test/velocity-stokes-cavity-test.xdmf") as infile:
        infile.read_checkpoint(v, "v", 0)
else:
    v = Expression(("adv * x[1]", "-adv * x[0]"), degree=1, adv=adv)

# Define the double well potential
F_u = Expression("u0<DOLFIN_EPS ? Gamma * (pow(u0,2)) : (u0>1.0-DOLFIN_EPS ? Gamma * (pow(u0-1,2)): Gamma * (pow(u0,2)) * (pow(u0-1,2)))",
                 u0=u_0, Gamma=Gamma, degree=deg)
dF_u = Expression("u0<DOLFIN_EPS ? 2 * Gamma * u0 : (u0>1.0-DOLFIN_EPS ? 2 * Gamma * (u0-1): 2 * Gamma * (u0 * pow(u0-1,2) + pow(u0,2) * (u0-1)))",
                  u0=u_0, Gamma=Gamma, degree=deg)

mu_u_0 = dF_u - pow(eps, 2) * div(grad(interpolate(init_u,V2)))
mu_u_0 = project(mu_u_0, V1)

# Define the dynamics vector
dynam = []

# Define the energy vector
E = []
energy = assemble(
    0.5*pow(eps,2)*dot(grad(u_0),grad(u_0))*dx \
    + interpolate(F_u, V1) * dx \
)
E.append(energy)
if(rank==0):
    print('E =', energy)

# Define variational functions
# Meaningless function used to define the variational formulation
solvector = Function(W)
# Meaningless function used to define the variational formulation
testvector = TestFunction(W)

u, mu_u = split(solvector)
barmu_u, baru = split(testvector)

# Define the derivative of the double well potential
f_im = 3 * Gamma * u

f_ex = Expression("u0<DOLFIN_EPS ? - Gamma * u0 : (u0>1.0-DOLFIN_EPS ? - Gamma * (u0+2.0): Gamma * (4.0*(pow(u0,3)) - 6.0*(pow(u0,2)) - u0))",
                  u0=u_0, Gamma=Gamma, degree=deg)


# Define variational problem

# Implicit
sigmoid_u = 1.0/(1.0+exp(-kappa * dot(avg(v),normal('+')))) # Sigmoidal approximation
# M_u = (abs(u*(1-u)) + u*(1-u))/2.0
def M(u):                   # Mobility function
    return(coef_Mov * (abs(u * (1.0 - u)) + u * (1.0 - u)) / 2.0)

# reg = pow(h,4)              # Regularization parameter

a1 = u * barmu_u * dx \
    + dt * ( # SIP bilinear form
        M(u) * dot(grad(mu_u),grad(barmu_u)) * dx \
        - dot(avg(M(u)*grad(mu_u)),normal('+')) * jump(barmu_u) * dS \
        - dot(avg(M(u)*grad(barmu_u)),normal('+')) * jump(mu_u) * dS \
        + sigma/avg(h) * jump(mu_u) * jump(barmu_u) * dS
        # Sigmoid bilinear form
        - u * dot(v,grad(barmu_u)) * dx \
        + (sigmoid_u * u('+') + (1-sigmoid_u) * u('-')) * dot(v,normal('+')) * jump(barmu_u) * dS
    )
L1 = u_0 * barmu_u * dx


a2 = mu_u * baru * dx \
    - pow(eps, 2) *(
        dot(grad(u),grad(baru)) * dx \
        - dot(avg(grad(u)),normal('+')) * jump(baru) * dS \
        - dot(avg(grad(baru)),normal('+')) * jump(u) * dS \
        + sigma/avg(h) * jump(u) * jump(baru) * dS
        ) \
    - f_im * baru * dx
L2 = f_ex * baru * dx

a = a1 + a2
L = L1 + L2
F = a - L

# Time-stepping
assign(solvector, [u_0, mu_u_0])
t = 0

auxtrial = TrialFunction(W)

dw = Function(W)
solvector_ = Function(W)
u_aux, mu_aux = split(solvector_)

right_vector = Function(W)
solvector_u = Function(W)
assign(solvector_u, [u_0, interpolate(Constant(0.0),V1)])

if(rank==0):
    print("Iteraciones:")

for i in range(num_steps):

    if(rank==0):
        print("\nIteración %d:" % (i + 1))

    assign(right_vector,[u_0,interpolate(Constant(0.0),V1)])

    # Update current time
    t += dt

    # Compute solution
    relaxation = 1.0
    iteration = 0
    iteration_max = 50
    absolute = 1.0
    absolute_tol = 1.0E-10
    relative_tol = 1.0E-9
    convergence = False
    while iteration < iteration_max and convergence != True:
        F_matrix = assemble(F)
        J = derivative(F, solvector, auxtrial)
        J_matrix = assemble(J)
        # print(f"cond(J) = {np.linalg.cond(J_matrix.array())}")
        solve(J_matrix, dw.vector(), -F_matrix, 'gmres')
        solvector_.vector()[:] = solvector.vector() + relaxation * dw.vector()
        convergence, absolute, relative = ConvergenceFunction(
            iteration, dw, absolute, absolute_tol, relative_tol, convergence)
        solvector.assign(solvector_)
        u_aux, mu_aux = solvector_.split()
        assign(solvector_u,[u_aux,interpolate(Constant(0.0),V1)])

        if(rank==0):
            print("My Newton iteration %d: r (abs) = %.3e (tol = %.3e) r (rel) = %.3e (tol = %.3e)" % (
                iteration, absolute, absolute_tol, relative, relative_tol))
        iteration += 1

    u, mu_u = solvector.split(True)

    # Plot solution
    plt.set_cmap('coolwarm')
    pic = plot(u)
    plt.title("$u$ at $t = %.4f$" % (t))
    plt.colorbar(pic)
    if(i == 0 or i == (num_steps / 4 - 1) or i == (num_steps / 2 - 1) or i == (3 * num_steps / 4 - 1) or i == (num_steps - 1)):
        if savepic:
            if stokes:
                plt.savefig("fig/u_DG-SIP-Sigmoidal+Eyre_stokes_nt-%d_t-%.5f_P%d.png" %
                            (num_steps, t, deg))
            else:
                plt.savefig("fig/u_DG-SIP-Sigmoidal+Eyre_nt-%d_t-%.5f_P%d_adv-%.1f_nx-%d.png" %
                            (num_steps, t, deg, adv, nx))
        if savesol:
            # Write `u` to a file:
            # fFile = HDF5File(MPI.comm_world,"fig/u_sol_FE_nt-%d_t-%.5f_P%d_adv-%.1f_nx-%d.h5" %(num_steps, t, deg, adv, nx),"w")
            # fFile.write(u,"u_sol_FE")
            # fFile.close()

            if stokes:
                with XDMFFile("fig/u_sol_DG-SIP-Sigmoidal_stokes_nt-%d_T-%.3f_P%d_counters-5.xdmf" %(num_steps, T, deg)) as outfile:
                    counter += 1
                    outfile.write_checkpoint(u, "u", counter, append=True)
            else:
                with XDMFFile("fig/u_sol_DG-SIP-Sigmoidal_nt-%d_T-%.3f_P%d_adv-%.1f_nx-%d_counters-5.xdmf" %(num_steps, T, deg, adv, nx)) as outfile:
                    counter += 1
                    outfile.write_checkpoint(u, "u", counter, append=True)
    plt.close()
    # plt.show()

    # Compute the mass
    mass_u = assemble(u * dx)
    if(rank==0):
        print('mass_u = %f' % (mass_u))

    # Compute dynamics
    dynamics = np.abs(u.vector().get_local()-u_0.vector().get_local()).max()/np.abs(u_0.vector().get_local()).max()
    dynam.append(dynamics)

    # Update previous solution

    u_0.assign(u)
    mu_u_0.assign(mu_u)

    F_u.u0 = u_0

    f_ex.u0 = u_0

    # We look at the signs
    max_u = u.vector().max()
    min_u = u.vector().min()
    max_u_vector.append(max_u)
    min_u_vector.append(min_u)

    if(rank==0):
        print('max(u) = %f' % (max_u))
        print('min(u) = %f' % (min_u))

    # Compute the energy
    energy = assemble(
        0.5*pow(eps,2)*dot(grad(u),grad(u))*dx \
        + interpolate(F_u, V1) * dx \
    )
    E.append(energy)
    if(rank==0):
        print('E =', energy)

if(savemax and rank==0):
    # open a binary file in write mode
    if stokes:
        file = open("fig/max_u_DG-SIP-Sigmoidal_stokes_nt-%d_T-%.3f_P%d" %(num_steps, T, deg), "wb")
    else:
        file = open("fig/max_u_DG-SIP-Sigmoidal_nt-%d_T-%.3f_P%d_adv-%.1f_nx-%d" %(num_steps, T, deg, adv, nx), "wb")
    # save array to the file
    np.save(file, max_u_vector)
    # close the file
    file.close

    # open a binary file in write mode
    if stokes:
        file = open("fig/min_u_DG-SIP-Sigmoidal_stokes_nt-%d_T-%.3f_P%d" %(num_steps, T, deg), "wb")
    else:
        file = open("fig/min_u_DG-SIP-Sigmoidal_nt-%d_T-%.3f_P%d_adv-%.1f_nx-%d" %(num_steps, T, deg, adv, nx), "wb")
    # save array to the file
    np.save(file, min_u_vector)
    # close the file
    file.close

plt.set_cmap('coolwarm')
pic = plot(u)
plt.title("$u$ in t = %.5f" % (t))
plt.colorbar(pic)
if showpic:
    plt.show()
plt.close()

plt.plot(np.linspace(0, T, num_steps + 1), E, color='red')
plt.title("Discrete energy")
plt.xlabel("Time")
plt.ylabel("Energy")
if savepic:
    if stokes:
        plt.savefig("fig/DG-SIP-Sigmoidal+Eyre_stokes_nt-%d_t-%.5f_P%d_energia.png" %
                    (num_steps, t, deg))
    else:
        plt.savefig("fig/DG-SIP-Sigmoidal+Eyre_nt-%d_t-%.5f_P%d_adv-%.1f_nx-%d_energia.png" %
                    (num_steps, t, deg, adv, nx))
if showpic:
    plt.show()
plt.close()

if(savesol):
    # Write `u` to a file:
    # fFile = HDF5File(MPI.comm_world,"fig/u_sol_FE_nt-%d_t-%.5f_P%d_adv-%.1f_nx-%d.h5" %(num_steps, t, deg, adv, nx),"w")
    # fFile.write(u,"u_sol_FE")
    # fFile.close()

    # if stokes:
    #     with XDMFFile("fig/u_sol_DG-SIP-Sigmoidal_stokes_nt-%d_T-%.3f_P%d_counters-5.xdmf" %(num_steps, T, deg)) as outfile:
    #         counter += 1
    #         outfile.write_checkpoint(u, "u", counter, append=True)
    # else:
    #     with XDMFFile("fig/u_sol_DG-SIP-Sigmoidal_nt-%d_T-%.3f_P%d_adv-%.1f_nx-%d_counters-5.xdmf" %(num_steps, T, deg, adv, nx)) as outfile:
    #         counter += 1
    #         outfile.write_checkpoint(u, "u", counter, append=True)

    # Write dynamics to a fFile
    if(rank==0):
        if stokes:
            file = open("fig/dynamics_DG-SIP-Sigmoidal_stokes_nt-%d_T-%.3f_P%d" %(num_steps, T, deg), "wb")
        else:
            file = open("fig/dynamics_DG-SIP-Sigmoidal_nt-%d_T-%.3f_P%d_adv-%.1f_nx-%d" %(num_steps, T, deg, adv, nx), "wb")
        np.save(file, dynam)
        file.close

    # Write energy to a file:
    if(rank==0):
        if stokes:
            file = open("fig/energy_DG-SIP-Sigmoidal_stokes_nt-%d_T-%.3f_P%d" %(num_steps, T, deg), "wb")
        else:
            file = open("fig/energy_DG-SIP-Sigmoidal_nt-%d_T-%.3f_P%d_adv-%.1f_nx-%d" %(num_steps, T, deg, adv, nx), "wb")
        np.save(file, E)
        file.close
