#
# Keller-Segel Test
# ===================
#

import dolfin as fe
from dolfin import (
    Mesh, Expression, XDMFFile, HDF5File, FacetNormal, FacetArea, UserExpression, MeshFunction,
    Function, FunctionSpace, TestFunction, TrialFunction, FiniteElement, MixedElement,
    dx, dS, dot, grad, div, avg, jump, sqrt, ln, cells,
    interpolate, assemble, solve, derivative, split, action, assign, project,
    Constant, plot
)
from mpi4py import MPI
import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys, os

fe.set_log_level(30)  # Only warnings (default: 20, information of general interest)

comm = fe.MPI.comm_world
rank = comm.Get_rank()
fe.parameters["ghost_mode"] = "shared_vertex" # Share the mesh between processors


def printMPI(string, end='\n'):
    if rank == 0:
        print(string, end=end)

#
# Mass laumping function
#
def assemble_mass_lumping(mass_bilinear_form, mass_lumped_variables):
    """Compute mass lumping for a mass bilinear form. This mass form is
    an expression like
        u*ub*dx + w*wb*dx,
    for a given function u and test function ub.
    The method expect the variables to be mass lumped given as an expression:
        assign(mass_lumped_variables, interpolate(Constant(1.0), V))
        or
        assign(mass_lumped_variables,
            [interpolate(Constant(0.0), V1), interpolate(Constant(1.0), V2)]).
    """
    # Mass lumping matrix
    mass_action_form = action(mass_bilinear_form, mass_lumped_variables)
    ML = assemble(mass_bilinear_form)
    ML.zero()
    ML.set_diagonal(assemble(mass_action_form))

    return ML

#
#   Conergence Function
#
def ConvergenceFunction(iteration, v1, v0, abs_tol, rel_tol, convergence):
    """Returns the absolute and relative error of an iterative method
    and a parameter that tells you if the iterative method has converged
    or not according to the tolerances.
    """
    absolute = sqrt(assemble(pow(v1, 2) * dx))
    # u, mu, wu, pi0mu = v1.split(True)
    # print(f'ABSOLUTE: {absolute}')
    # other_absolute = sqrt(assemble(pow(u,2) * dx) + assemble(pow(mu,2) * dx) + assemble(pow(wu,2) * dx) + assemble(pow(pi0mu,2) * dx))
    # print(f'OTHER: {other_absolute}') # Same value than absolute
    if (iteration == 0):
        absolute0 = absolute
    else:
        absolute0 = v0
    relative = absolute / absolute0
    if absolute < abs_tol or relative < rel_tol:
        convergence = True
    return convergence, absolute, relative

#
# Problem class
#
class KellerSegel_DG_UPW(object):
    r"""
    DG numerical solution of Keller-Segel equation
    with Neumann homogeneous conditions:

    u_t = k0* Laplace(u) - k1 * \nabla *(u grad(v))     in the unit square
    v_t = k2 * Laplace(v) - k3 * v + k4 * u             in the unit square
    grad(u) * n = grad(v) * n = 0                    on the boundary

    """

    def __init__(self, ks_parameters):
        #
        # Load PDE and discretization parameters
        #
        ks = self
        p = ks.parameters = ks_parameters
        ks.eps = p["eps"]
        ks.k0 = p["k0"]
        ks.k1 = p["k1"]
        ks.k2 = p["k2"]
        ks.k3 = p["k3"]
        ks.k4 = p["k4"]
        ks.tau = p["tau"]

        #
        # Load refinement parameters
        #
        ks.nx = p["nx"]

        ks.h_tol = p["h_tol"]
        ks.REF_LEVEL_MAX = p["REF_LEVEL_MAX"]
        ks.max_value_ref = p["max_value_ref"]
        ks.mesh_refine = p["mesh_refine"]

        ks.plot = p["plot"]
        ks.vtk = p["vtk"]
        ks.savefunc = p["savefunc"]

        ks.u_init = Expression(p["u0"], degree=2)
        ks.v_init = Expression(p["v0"], degree=2)

        ks.file_path = os.path.dirname(os.path.abspath(__file__))
        printMPI(f"file_path = {ks.file_path}")
        ks.mesh_file = f"{ks.file_path}/meshes/" + p["mesh"]
        printMPI(f"mesh_file = {ks.mesh_file}")

        ks.T = p["T"]
        ks.dt_min = p["dt_min"]
        ks.dt_max = p["dt_max"]
        ks.alpha = p["alpha"]
        ks.dt = Constant(ks.dt_min)
        ks.t = 0.

    def read_mesh(self):
        ks = self
        #
        # Read mesh
        #
        ks.mesh = Mesh()
        with XDMFFile(comm, ks.mesh_file) as infile:
            infile.read(ks.mesh)
        mesh = ks.mesh

        ks.h_min = ks.mesh.hmin()

        fig = plot(mesh)
        plt.savefig(f"mesh_level-0_hmin-{ks.h_min:.5f}.png")
        plt.close()

    def define_spaces(self):
        ks = self
        mesh = ks.mesh
        #
        # Build DG, FE spaces and functions
        #
        ks.P0d = FiniteElement("DG", mesh.ufl_cell(), 0)
        ks.P1c = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        ks.P2c = FiniteElement("Lagrange", mesh.ufl_cell(), 2)
        ks.Uh = FunctionSpace(ks.mesh, ks.P0d)
        ks.Vh1 = FunctionSpace(ks.mesh, ks.P1c)
        ks.Vh2 = FunctionSpace(ks.mesh, ks.P2c)
        ks.Wh = FunctionSpace(ks.mesh, MixedElement([ks.P0d, ks.P0d]))

    def define_variables(self):
        ks = self
        #
        # Variables to store solution at two time steps
        #
        ks.solvector, ks.trialvector, ks.testvector = Function(ks.Wh), TrialFunction(ks.Wh), TestFunction(ks.Wh)
        u, mu = ks.u, ks.mu = split(ks.solvector)
        v = ks.v = Function(ks.Vh1)
        u0, mu0 = ks.u0, ks.mu0 = Function(ks.Uh), Function(ks.Uh)
        v0 = ks.v0 = Function(ks.Vh1)

        #
        # Test variables
        #
        ub, mub = ks.ub, ks.mub = split(ks.testvector)
        vb = ks.vb = TestFunction(ks.Vh1)

        #
        # Trial variables
        #
        u_trial, mu_trial = ks.u_trial, ks.mu_trial = split(ks.trialvector)
        v_trial = ks.v_trial = TrialFunction(ks.Vh1)

        # Create a Newton step variable
        ks.Nwstep = Function(ks.Wh)

        # Create a auxiliar variable
        ks.solvector_ = Function(ks.Wh)

        #
        # Mass lumping matrix of v
        #
        mass_lumped_variables = Function(ks.Vh1)
        assign(mass_lumped_variables, interpolate(Constant(1.0), ks.Vh1))
        ks.ML = assemble_mass_lumping(v_trial*vb*dx, mass_lumped_variables)

    def load_initial_values(self):
        """Initialize variables"""
        ks = self

        ks.u0.assign(interpolate(ks.u_init, ks.Uh))
        ks.v0.assign(interpolate(ks.v_init, ks.Vh1))
        ks.mu0.assign(project(ks.k0 * ln(ks.u0 + ks.eps) - ks.k1 * ks.v0, ks.Uh))

    def variational_problem_v(self):
        """Build variational problem"""
        #
        # Load variables from Keller-Segel problem
        #
        ks = self
        dt = ks.dt
        v_trial, vb = ks.v_trial, ks.vb
        u0, v0 = ks.u0, ks.v0
        k2, k3, k4 = ks.k2, ks.k3, ks.k4
        tau = ks.tau

        #
        # Variational problem
        #
        if ks.tau:
            av = dt/tau * (  # FE approximation
                    k2 * dot(grad(v_trial),grad(vb)) * dx \
                    )
            Lv = dt * k4 * u0 * vb * dx
        else:
            av = k2 * dot(grad(v_trial),grad(vb)) * dx # FE approximation
            Lv = k4 * u0 * vb * dx

        ks.av = av
        ks.Lv = Lv

    def variational_problem_u(self):
        """Build variational problem"""
        #
        # Load variables from Keller-Segel problem
        #
        ks = self
        nx = ks.nx
        dt = ks.dt
        u, ub = ks.u, ks.ub
        v = ks.v
        mu, mub = ks.mu, ks.mub
        u0, mu0 = ks.u0, ks.mu0
        eps = ks.eps
        k0, k1 = ks.k0, ks.k1

        def pos(u):
            return ((abs(u) + u) / 2.0)
        def neg(u):
            return ((abs(u) - u) / 2.0)

        #
        # Variational problem
        #
        # normal = FacetNormal(ks.mesh)
        e_len = FacetArea(ks.mesh)
        l = 1/nx

        a1 = u * ub * dx \
            + dt * (  # UPW bilinear form
                + pos(jump(mu)/((2*(l**2))/(3*avg(e_len)))) * pos(u('+')) * jump(ub) * dS \
                - neg(jump(mu)/((2*(l**2))/(3*avg(e_len)))) * pos(u('-')) * jump(ub) * dS
            )
        L1 = u0 * ub * dx

        a2 = mu * mub * dx \
            - k0 * ln(u + eps) * mub * dx \
            + k1 * v * mub * dx

        a = a1 + a2
        L = L1
        ks.Fu = a - L

    def solve_v(self):
        #
        # Load v from Keller-Segel problem
        #
        ks = self
        v, v0 = ks.v, ks.v0
        ML = ks.ML
        k3 = ks.k3
        tau = ks.tau
        dt = ks.dt
        av_matrix = assemble(ks.av)
        Lv_vector = assemble(ks.Lv)
        if tau:
            av_matrix = av_matrix + (1 + k3 * dt / tau) * ML
            Lv_vector = Lv_vector + ML * v0.vector()
        else:
            av_matrix = av_matrix + (tau + k3) * ML
            Lv_vector = Lv_vector

        solve(av_matrix, v.vector(), Lv_vector, 'gmres')

    def solve_u(self, verbosity=0):
        """Computes Newton method"""
        #
        # Load u, mu from Keller-Segel problem
        #
        ks = self
        solvector = ks.solvector
        trialvector = ks.trialvector
        Fu = ks.Fu

        Nwstep = ks.Nwstep
        solvector_ = ks.solvector_

        #
        # Compute solution
        #
        relaxation = 1.0
        iteration = 0
        iteration_max = 10
        absolute = 1.0
        absolute_tol = 1.0E-10
        relative_tol = 1.0E-9
        convergence = False
        while iteration < iteration_max and convergence != True:
            Fu_matrix = assemble(Fu)
            Ju = derivative(Fu, solvector, trialvector)
            Ju_matrix = assemble(Ju)

            try:
                solve(Ju_matrix, Nwstep.vector(), -Fu_matrix, 'gmres')
            except:
                comm.Barrier();
                if verbosity: printMPI("Linear solver did not converge")
                convergence = False
                iteration = iteration_max + 1
                comm.Barrier();
            else:
                solvector_.vector()[:] = solvector.vector() + relaxation * Nwstep.vector()
                convergence, absolute, relative = ConvergenceFunction(
                    iteration, Nwstep, absolute, absolute_tol, relative_tol, convergence)
                solvector.assign(solvector_)
            if verbosity:
                printMPI(f"My Newton iteration {iteration}: r (abs) = {absolute:.3e}",
                        f"(tol = {absolute_tol:.3e}) r (rel) = {relative:.3e} (tol = {relative_tol:.3e})\n")
            iteration += 1

        if not convergence:
            comm.Barrier()            # wait for every MPI process
            raise(Exception("Newton method did not converge"))
            comm.Barrier()            # rank>0 deadlocks here because rank=0 exited early

    def refine_cell(self, cell, u, v, max_value=500):
        p = cell.midpoint()
        if (u(p) > max_value) or (v(p) > max_value):
            do_refine = True
        else:
            do_refine = False
        return do_refine

    def refine_mesh(self, level, sol, sol0, max_value=500, verbosity=0):
        (u, mu, v) = sol
        (u0, mu0, v0) = sol0

        ks = self
        if level < ks.REF_LEVEL_MAX:
            if rank == 0:
                cell_markers = MeshFunction("bool", ks.mesh, ks.mesh.topology().dim())
                cell_markers.set_all(False)
                for cell in cells(ks.mesh):
                    cell_markers[cell] = ks.refine_cell(cell, u, v, max_value)

            if any(cell_markers):
                u0_copy, mu0_copy = Function(ks.Uh), Function(ks.Uh)
                v0_copy = Function(ks.Vh1)
                u0_copy.assign(u0)
                mu0_copy.assign(mu0)
                v0_copy.assign(v0)

                if rank == 0:
                    ks.mesh = fe.refine(ks.mesh, cell_markers)
                    ks.h_min = ks.mesh.hmin()

                ks.define_spaces()
                ks.define_variables()

                ks.u0.assign(project(u0_copy, ks.Uh))
                ks.mu0.assign(project(mu0_copy, ks.Uh))
                ks.v0.assign(project(v0_copy, ks.Vh1))

                assign(ks.solvector, [ks.u0, ks.mu0])

                ks.variational_problem_v()
                ks.variational_problem_u()

                level += 1
                if verbosity:
                    printMPI(f"h_min = {ks.h_min:f} (h_tol = {ks.h_tol:f})")

                fig = plot(ks.mesh)
                plt.savefig(f"mesh_ref_level-{level:d}_hmin-{ks.h_min:.5f}.png")
                plt.close()
            else:
                level = ks.REF_LEVEL_MAX

        return level

    def project_u_P1c(self, u):
        ks = self
        uP1cb = TestFunction(ks.Vh1)
        auP1c_matrix = ks.ML
        LuP1c_vector = assemble(u*uP1cb*dx)
        uP1c = Function(ks.Vh1)
        solve(auP1c_matrix, uP1c.vector(), LuP1c_vector, 'gmres')
        return uP1c

    def time_iterator(self, verbosity=0):
        """Time iterator"""
        #
        # Load variables from Keller-Segel problem
        #
        ks = self

        #
        # Build variational problem
        #
        ks.variational_problem_v()
        ks.variational_problem_u()

        #
        # Run time iterations
        #
        plot_figs = ks.plot
        vtk = ks.vtk
        savefunc = ks.savefunc

        assign(ks.solvector, [ks.u0, ks.mu0])
        step = 0
        level = 0
        while ks.t < ks.T - fe.DOLFIN_EPS:
            max_level_reached = False
            h_tol_reached = False
            max_reached = True
            convergence = True
            while not(max_level_reached) and not(h_tol_reached) and max_reached:
                # --- Build variational problem
                # ks.variational_problem_v()
                # ks.variational_problem_u()

                # --- Solution of v
                ks.solve_v()
                v = ks.v

                # --- Newton method to solve u
                try:
                    ks.solve_u(verbosity)
                except:
                    comm.Barrier()            # wait for every MPI process
                    if rank==0:
                        raise(Exception("Newton method did not converge."))

                else:
                    # --- Save solution (to be used in next iteration)
                    u, mu = ks.solvector.split(True)

                    # -- Refine mesh
                    if ks.mesh_refine:
                        if ks.h_min <= ks.h_tol:
                            h_tol_reached = True
                        elif level >= ks.REF_LEVEL_MAX:
                            max_level_reached = True
                        elif (max(u.vector()) > ks.max_value_ref) or (max(v.vector()) > ks.max_value_ref):
                            level = ks.refine_mesh(level, (u, mu, v), (ks.u0, ks.mu0, ks.v0), max_value=ks.max_value_ref, verbosity=verbosity)
                            # ks.max_value_ref += 2/3 * ks.max_value_ref
                            ks.max_value_ref *= 2
                        else:
                            max_reached = False
                    else:
                        max_reached = False

            ks.t += float(ks.dt)
            step += 1
            if verbosity > 0:
                printMPI(f"Time iteration {step}, t={ks.t:.6e}")

            # --- Yield data computed in current iteration
            yield ({'step': step, 't': ks.t, "plot": plot_figs,
                'vtk': vtk, 'savefunc': savefunc, 'u': u, 'v': v, 'mu': mu,
                'u0': ks.u0, 'v0': ks.v0, 'mu0': ks.mu0})

            # --- Update solution
            max_u_old = max(ks.u0.vector())

            ks.v0.assign(v)
            ks.u0.assign(u)
            ks.mu0.assign(mu)

            max_u0 = max(ks.u0.vector())

            # --- Update time step size
            dt_max_scaled = ks.dt_max/np.sqrt(1 + ks.alpha * (np.abs(max_u0-max_u_old)/float(ks.dt))**2)
            ks.dt.assign(max(ks.dt_min, dt_max_scaled))
            if verbosity: print(f"Current dt = {ks.dt}")

# ---------------------------

def print_info(i, t, u_data, v_data, energy, dynamics = 0):
    u_max, u_min, u_mass = u_data
    if v_data:
        v_max, v_min, v_mass = v_data
        s = f"{i:3} {t:.6e} {u_max:.4e} {v_max:.4e}"
        s += f" {u_min:.4e} {v_min:.4e}"
        s += f" {u_mass:.4e} {v_mass:.4e}"
        s += f" {energy[0]:.4e} {energy[1]:.4e}"
        if dynamics:
            dynamics_u, dynamics_v = dynamics
            if dynamics_v < 0:
                s += f" {dynamics_u:.4e} nan"
            else:
                s += f" {dynamics_u:.4e} {dynamics_v:.4e}"
    else:
        s = f"{i:3} {t:.6e} {u_max:.4e} nan"
        s += f" {u_min:.4e} nan"
        s += f" {u_mass:.4e} nan"
        s += f" nan nan"

    printMPI(s)

def define_parameters_test_2d(param):
    # param.add("u0", "Expression_string")
    param.add("mesh", "mesh_square_nx-100.xdmf")
    return param


# def define_parameters_test_3d(param):
#     param.add("u0", "Cu0*exp(-Cu1*(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]))")
#     param.add("mesh", "sphere_0dot1.xml.gz")
#     return param


def define_parameters():
    param = fe.Parameters("ks_parameters")

    # Select 2d or 3d parameter set
    param.add("test", "2d")  # Default test: 2d
    param.parse(sys.argv)
    if param["test"] == "2d":
        param = define_parameters_test_2d(param)
    # else:
    #     param = define_parameters_test_3d(param)

    # Define remaining parameters
    param.add("eps", float(0.00001))
    param.add("k0", float(1.0))
    param.add("k1", float(1.0))
    param.add("k2", float(1.0))
    param.add("k3", float(1.0))
    param.add("k4", float(1.0))
    param.add("tau", float(1.0))

    param.add("u0", "1000 * exp(-100 * (pow(x[0],2) + pow(x[1],2)))")
    param.add("v0", "500 * exp(-50 * (pow(x[0],2) + pow(x[1],2)))")

    # Params for the discrete scheme
    param.add("T", float(1.e-4))
    param.add("dt_min", float(1.e-10))
    param.add("dt_max", float(1.e-7))
    param.add("alpha", float(1.e-19))

    # Refine parameters
    param.add("mesh_refine", 0)
    param.add("REF_LEVEL_MAX", 2)
    param.add("h_tol", 6.e-3)
    param.add("max_value_ref", 100)
    param.add("nx", 100)

    # Other parameters
    param.add("verbosity", 0) # No extra information shown
    param.add("plot", 0)  # No plot shown
    param.add("vtk",  0)  # No vtk photogram saved to disk
    param.add("vtkfile", "ks_DG-UPW")  # Name of vtk file
    param.add("save", 0) # No figures and output saved
    param.add("savefile", "ks_DG-UPW") # Name of output file
    param.add("savefunc", 0) # No functions saved in HDF5 format

    param.parse(sys.argv)

    # # Post-process parameters
    # if param["vtk"] != 0:
    #     N = param["tsteps"]
    #     n = min(param["vtk"], N)
    #     param.add("vtk_steps", [N/n * i for i in range(n)])
    #     # param.add("vtk_saving", int(param["tsteps"])//100)

    return param

#
# Main program
#
if(__name__ == "__main__"):
    #
    # Define parameters
    #
    p = parameters = define_parameters()
    printMPI("Parameters:")
    for k, value in parameters.items():
        printMPI(f"  {k} = {value.value()}")

    #
    # Init problem
    #
    problem_ks = KellerSegel_DG_UPW(parameters)
    problem_ks.read_mesh()
    problem_ks.define_spaces()
    problem_ks.define_variables()
    problem_ks.load_initial_values()

    printMPI("More info:")
    printMPI("  Mesh h    = " + str(problem_ks.mesh.hmax()))
    printMPI("  Initial dt   = " + str(problem_ks.dt_min))

    #
    # Time iterations
    #
    ks_iterations = problem_ks.time_iterator(verbosity=p["verbosity"])
    printMPI("Time steps:\n  i t u_max v_max u_min v_min u_mass v_mass energy energy_eps dynam_u dynam_v")

    time_vector = [problem_ks.t]
    for t_step in ks_iterations:

        i, t = t_step['step'], t_step['t']
        u, v = t_step['u'], t_step['v']
        u0, v0 = t_step['u0'], t_step['v0']
        plot_figs, vtk, savefunc = t_step['plot'], t_step['vtk'], t_step['savefunc']
        # printMPI(f"Time iteration {i}, t={t:.2e}")

        time_vector.append(t)

        if i == 1:
            u0_max, u0_min = max(u0.vector()), min(u0.vector())
            u0_mass = assemble(u0*dx)

            comm.barrier()
            u0_max = problem_ks.mesh.mpi_comm().allreduce(u0_max, op=MPI.MAX)
            u0_min = problem_ks.mesh.mpi_comm().allreduce(u0_min, op=MPI.MIN)

            max_u_list = [u0_max]
            min_u_list = [u0_min]
            if problem_ks.tau:
                energy0 = assemble(
                    problem_ks.k0 * u0 * ln(u0) * dx \
                    - problem_ks.k1 * u0 * v0 * dx \
                    + 0.5 * (problem_ks.k1 * problem_ks.k2 / problem_ks.k4) * dot(grad(v0), grad(v0)) * dx \
                    + 0.5 * (problem_ks.k1 * problem_ks.k3 / problem_ks.k4) * pow(v0, 2) * dx
                )
                energy0_modified = assemble(
                    problem_ks.k0 * (u0 + problem_ks.eps) * ln(u0 + problem_ks.eps) * dx \
                    - problem_ks.k1 * u0 * v0 * dx \
                    + 0.5 * (problem_ks.k1 * problem_ks.k2 / problem_ks.k4) * dot(grad(v0), grad(v0)) * dx \
                    + 0.5 * (problem_ks.k1 * problem_ks.k3 / problem_ks.k4) * pow(v0, 2) * dx
                )

                v0_max, v0_min = max(v0.vector()), min(v0.vector())
                v0_mass = assemble(v0*dx)

                comm.barrier()
                v0_max = problem_ks.mesh.mpi_comm().allreduce(v0_max, op=MPI.MAX)
                v0_min = problem_ks.mesh.mpi_comm().allreduce(v0_min, op=MPI.MIN)

                max_v_list = [v0_max]
                min_v_list = [v0_min]
                E = [energy0]
                E_eps = [energy0_modified]

                print_info(0, 0,
                           (u0_max, u0_min, u0_mass),
                           (v0_max, v0_min, v0_mass),
                           (energy0, energy0_modified))
            else:
                max_v_list = []
                min_v_list = []
                E = []
                E_eps = []

                print_info(0, 0,
                           (u0_max, u0_min, u0_mass),
                           0, 0)
            dynam_u_list = []
            dynam_v_list = []

            #
            # Save output
            #
            do_save = p["save"]
            base_name_save = p["savefile"]
            if do_save:
                 mpl.use('Agg')

            #
            # Plot
            #
            u0P1c =  problem_ks.project_u_P1c(u0)

            do_plot = (p["plot"] > 0)
            if do_plot:
                plt.set_cmap('RdYlGn')
                pic = plot(u0, title=f"u, t={0:.6e}")
                plt.colorbar(pic)
                if do_save: plt.savefig(f"{base_name_save}_u0.png")
                else: plt.show()
                plt.close()

                # Project u0  to P1c using mass-lumping
                plt.set_cmap('RdYlGn')
                pic = plot(u0P1c, title=f"u0_P1c, t={0:.6e}")
                plt.colorbar(pic)
                if do_save: plt.savefig(f"{base_name_save}_u0_P1c.png")
                else: plt.show()
                plt.close()

                if problem_ks.tau:
                    plt.set_cmap('plasma')
                    pic = plot(v0, title=f"v, t={0:.6e}")
                    plt.colorbar(pic)
                    if do_save: plt.savefig(f"{base_name_save}_v0.png")
                    else: plt.show()
                    plt.close()

            #
            # Save w to HDF5
            #
            do_save_func = p["savefunc"]
            if do_save_func:
                counter = 0
                with XDMFFile(comm,f'u_{base_name_save}.xdmf') as outfile:
                    outfile.write_checkpoint(u0, "u", counter, append=False)
                with XDMFFile(comm,f'uP1c_{base_name_save}.xdmf') as outfile:
                    outfile.write_checkpoint(u0P1c, "u", counter, append=False)
                if problem_ks.tau:
                    with XDMFFile(comm,f'v_{base_name_save}.xdmf') as outfile:
                        outfile.write_checkpoint(v0, "v", counter, append=False)

            #
            # Save to VTK
            #
            base_name_vtk = p["vtkfile"]
            vtk_u_file = fe.File(f"{base_name_vtk}_u.pvd")
            vtk_uP1c_file = fe.File(f"{base_name_vtk}_uP1c.pvd")
            vtk_v_file = fe.File(f"{base_name_vtk}_v.pvd")
            do_vtk = (p["vtk"] > 0)
            if (do_vtk):  # Only save some fotograms
                # printMPI("Saving vtk to " + p["vtkfile"])
                u0.rename("u", "Cell density")
                u0P1c.rename("u", "Cell density")
                v0.rename("v", "Chemical attractant concentration")
                vtk_u_file << (u0, 0)
                vtk_uP1c_file << (u0P1c, 0)

                if problem_ks.tau:
                    vtk_v_file << (v0, 0)

        #
        # Print info
        #
        u_max, u_min = max(u.vector()), min(u.vector())
        v_max, v_min = max(v.vector()), min(v.vector())

        comm.barrier()
        u_max = problem_ks.mesh.mpi_comm().allreduce(u_max, op=MPI.MAX)
        u_min = problem_ks.mesh.mpi_comm().allreduce(u_min, op=MPI.MIN)
        v_max = problem_ks.mesh.mpi_comm().allreduce(v_max, op=MPI.MAX)
        v_min = problem_ks.mesh.mpi_comm().allreduce(v_min, op=MPI.MIN)

        u_mass, v_mass = assemble(u*dx), assemble(v*dx)
        energy = assemble(
            problem_ks.k0 * u * ln(u) * dx \
            - problem_ks.k1 * u * v * dx \
            + 0.5 * (problem_ks.k1 * problem_ks.k2 / problem_ks.k4) * dot(grad(v), grad(v)) * dx \
            + 0.5 * (problem_ks.k1 * problem_ks.k3 / problem_ks.k4) * pow(v, 2) * dx
        )
        energy_eps = assemble(
            problem_ks.k0 * (u + problem_ks.eps) * ln(u + problem_ks.eps) * dx \
            - problem_ks.k1 * u * v * dx \
            + 0.5 * (problem_ks.k1 * problem_ks.k2 / problem_ks.k4) * dot(grad(v), grad(v)) * dx \
            + 0.5 * (problem_ks.k1 * problem_ks.k3 / problem_ks.k4) * pow(v, 2) * dx
        )

        max_u_list.append(u_max)
        min_u_list.append(u_min)
        max_v_list.append(v_max)
        min_v_list.append(v_min)
        E.append(energy)
        E_eps.append(energy_eps)

        u0_refined = project(u0,problem_ks.Uh)
        dynamics_u = np.abs(u.vector().get_local()-u0_refined.vector().get_local()).max()/np.abs(u0_refined.vector().get_local()).max()
        dynam_u_list.append(dynamics_u)

        if problem_ks.tau or (i > 1):
            v0_refined = project(v0,problem_ks.Vh1)
            dynamics_v = np.abs(v.vector().get_local()-v0_refined.vector().get_local()).max()/np.abs(v0_refined.vector().get_local()).max()
            dynam_v_list.append(dynamics_v)
        else:
            dynamics_v = -1

        print_info(i, t,
                   (u_max, u_min, u_mass),
                   (v_max, v_min, v_mass),
                   (energy, energy_eps),
                   (dynamics_u, dynamics_v))

        #
        # Plot
        #
        uP1c = problem_ks.project_u_P1c(u)
        if (do_plot and i % plot_figs == 0):  # Plot some steps
            plt.set_cmap('RdYlGn')
            pic = plot(u, title=f"u, t={t:.6e}")
            plt.colorbar(pic)
            if do_save: plt.savefig(f"{base_name_save}_u_i-{i}.png")
            else: plt.show()
            plt.close()

            # Project u0  to P1c using mass-lumping
            plt.set_cmap('RdYlGn')
            pic = plot(uP1c, title=f"uP1c, t={t:.6e}")
            plt.colorbar(pic)
            if do_save: plt.savefig(f"{base_name_save}_uP1c_i-{i}.png")
            else: plt.show()
            plt.close()

            plt.set_cmap('plasma')
            pic = plot(v, title=f"v, t={t:.6e}")
            plt.colorbar(pic)
            if do_save: plt.savefig(f"{base_name_save}_v_i-{i}.png")
            else: plt.show()
            plt.close()

        #
        # Save w to HDF5
        #
        if (do_save_func and i % savefunc == 0):
            counter += 1
            with XDMFFile(comm,f'u_{base_name_save}.xdmf') as outfile:
                outfile.write_checkpoint(u, "u", counter, append=True)
            with XDMFFile(comm,f'uP1c_{base_name_save}.xdmf') as outfile:
                outfile.write_checkpoint(uP1c, "u", counter, append=True)
            with XDMFFile(comm,f'v_{base_name_save}.xdmf') as outfile:
                outfile.write_checkpoint(v, "v", counter, append=True)

        #
        # Save to VTK
        #
        if (do_vtk and i % vtk == 0):
            u.rename("u", "Cell density")
            uP1c.rename("u", "Cell density")
            v.rename("v", "Chemical attractant concentration")
            vtk_u_file << (u, t)
            vtk_uP1c_file << (uP1c, t)
            vtk_v_file << (v, t)

    #
    # Plot
    #
    if do_plot:

        fig, axs = plt.subplots(2)
        axs[0].plot(time_vector,max_u_list,'--',c='orange')
        axs[1].plot(time_vector,np.zeros(i + 1),'-',c='lightgray',linewidth=2,label='_nolegend_')
        axs[1].plot(time_vector,min_u_list,'--',c='orange')
        plt.subplots_adjust(hspace=0.5, bottom=0.16)
        if do_save: plt.savefig(f"{base_name_save}_min-max_u.png")
        else: plt.show()
        plt.close()

        plt.plot(time_vector[1:], dynam_u_list, color='darkorange')
        plt.title("Dynamics u")
        plt.xlabel("Time")
        plt.ylabel("Dynamics")
        if do_save: plt.savefig(f"{base_name_save}_dynamics_u.png")
        else: plt.show()
        plt.close()

        if not problem_ks.tau:
            time_vector =  time_vector[1:]
            i -= 1

        fig, axs = plt.subplots(2)
        axs[0].plot(time_vector,max_v_list,'--',c='purple')
        axs[1].plot(time_vector,np.zeros(i + 1),'-',c='lightgray',linewidth=2,label='_nolegend_')
        axs[1].plot(time_vector,min_v_list,'--',c='purple')
        plt.subplots_adjust(hspace=0.5, bottom=0.16)
        if do_save: plt.savefig(f"{base_name_save}_min-max_v.png")
        else: plt.show()
        plt.close()

        plt.plot(time_vector, E, color='red')
        plt.title("Discrete energy")
        plt.xlabel("Time")
        plt.ylabel("Energy")
        if do_save: plt.savefig(f"{base_name_save}_energy.png")
        else: plt.show()
        plt.close()

        plt.plot(time_vector, E_eps, color='darkgreen')
        plt.title("Discrete modified energy")
        plt.xlabel("Time")
        plt.ylabel("Modified energy")
        if do_save: plt.savefig(f"{base_name_save}_energy_eps.png")
        else: plt.show()
        plt.close()


        plt.plot(time_vector[1:], dynam_v_list, color='darkblue')
        plt.title("Dynamics v")
        plt.xlabel("Time")
        plt.ylabel("Dynamics")
        if do_save: plt.savefig(f"{base_name_save}_dynamics_v.png")
        else: plt.show()
        plt.close()

    #
    # Save output
    #
    # if do_save and rank == 0:
    #     with open(f"{base_name_save}_max_u", 'wb') as file:
    #         np.save(file, max_u_list)
    #     with open(f"{base_name_save}_min_u", 'wb') as file:
    #         np.save(file, min_u_list)
    #     with open(f"{base_name_save}_max_w", 'wb') as file:
    #         np.save(file, max_w_list)
    #     with open(f"{base_name_save}_min_w", 'wb') as file:
    #         np.save(file, min_w_list)
    #     with open(f"{base_name_save}_energy", 'wb') as file:
    #         np.save(file, E)
    #     with open(f"{base_name_save}_dynamics_u", 'wb') as file:
    #         np.save(file, dynam_u_list)
    #     with open(f"{base_name_save}_dynamics_w", 'wb') as file:
    #         np.save(file, dynam_w_list)
