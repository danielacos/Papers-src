#
# chemo Test
# ===================
# FEniCSx 0.8

import dolfinx
from dolfinx import fem
from dolfinx.fem import (
    Expression, Function, functionspace, Constant,
    assemble_scalar, form, petsc
)
from dolfinx.fem.petsc import LinearProblem
from dolfinx import log
from basix.ufl import element, mixed_element, enriched_element
from ufl import(
     TestFunction, TrialFunction,
     SpatialCoordinate,
     dx, dS, ds, inner, grad, div, avg, jump,
     exp, ln,
     split,
     FacetArea, FacetNormal 
)
from dolfinx.io import (
    XDMFFile
)
import dolfinx.plot as plot
from mpi4py import MPI
from petsc4py import PETSc
import pyvista
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from PIL import Image

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def printMPI(string, end='\n'):
    if rank == 0:
        print(string, end=end)

DOLFIN_EPS = 1e-16
dx_ML = dx(scheme="vertex", metadata={"degree":1, "representation":"quadrature"}) # mass lumped terms

#
# Problem class
#
class chemo_DG_UPW(object):

    def __init__(self, chemo_parameters):
        #
        # Load PDE and discretization parameters
        #
        chemo = self
        p = chemo.parameters = chemo_parameters
        chemo.eps = float(p.eps)
        chemo.chi_coef = float(p.chi_coef)
        chemo.xi_coef = float(p.xi_coef)
        chemo.lambda_coef = float(p.lambda_coef)
        chemo.mu_coef = float(p.mu_coef)
        chemo.c_coef = float(p.c_coef)
        chemo.a_coef = float(p.a_coef)
        chemo.b_coef = float(p.b_coef)
        chemo.e_coef = float(p.e_coef)
        chemo.d_coef = float(p.d_coef)
        chemo.rho_exp = float(p.rho_exp)
        chemo.k_exp = float(p.k_exp)
        chemo.gamma_exp = float(p.gamma_exp)
        chemo.alpha_exp = float(p.alpha_exp)
        chemo.beta_exp = float(p.beta_exp)
        chemo.m1_exp = float(p.m1_exp)
        chemo.m2_exp = float(p.m2_exp)
        chemo.m3_exp = float(p.m3_exp)
        chemo.tau = int(p.tau)

        file_path = os.path.dirname(os.path.abspath(__file__))
        printMPI(f"file_path = {file_path}")
        mesh_file = f"{file_path}/meshes/" + f"mesh_{p.mesh}_nx-{p.nx}.xdmf"
        printMPI(f"mesh_file = {mesh_file}")

        #
        # Read mesh
        #
        with XDMFFile(comm, mesh_file, 'r') as infile:
            mesh = chemo.mesh = infile.read_mesh()

        cell_dim = chemo.mesh.topology.dim
        def mark_all(x):
            return np.ones(x.shape[1])
        chemo.num_edges = dolfinx.mesh.locate_entities(mesh, cell_dim-1, mark_all).size
        
        chemo.nx = int(p.nx)
        chemo.dt = float(p.dt)
        chemo.t = 0.

        #
        # Build DG, FE spaces and functions
        #
        chemo.normal_gradient = int(p.normal_grad)
        chemo.P0d = element("DG", mesh.topology.cell_name(), degree=0)
        chemo.P1c = element("Lagrange", mesh.topology.cell_name(), degree=1)
        chemo.Uh = functionspace(mesh, chemo.P0d)
        chemo.Vh1 = functionspace(mesh, chemo.P1c)
        chemo.v, chemo.v_trial, chemo.vb = Function(chemo.Vh1), TrialFunction(chemo.Vh1), TestFunction(chemo.Vh1)
        chemo.w, chemo.w_trial, chemo.wb = Function(chemo.Vh1), TrialFunction(chemo.Vh1), TestFunction(chemo.Vh1)
        chemo.mu, chemo.mu_trial, chemo.mub =  Function(chemo.Vh1), TrialFunction(chemo.Vh1), TestFunction(chemo.Vh1)
        chemo.u, chemo.u_trial, chemo.ub =  Function(chemo.Uh), TrialFunction(chemo.Uh), TestFunction(chemo.Uh)
        chemo.v0, chemo.w0, chemo.mu0, chemo.u0 = Function(chemo.Vh1), Function(chemo.Vh1), Function(chemo.Vh1), Function(chemo.Uh)
        chemo.p1c_u = Function(chemo.Vh1)

        # Domain size
        aux = Function(chemo.Vh1)
        aux.x.array[:] = 1.0
        chemo.domain_size = assemble_scalar(form(aux * dx))

    def project(self, u, space, mass_lumping=False):
        chemo = self

        Piu_trial = TrialFunction(space)
        Piub = TestFunction(space)

        if mass_lumping:
            a = inner(Piu_trial, Piub) * dx_ML
        else:
            a = inner(Piu_trial, Piub) * dx

        L = inner(u, Piub) * dx

        problem = petsc.LinearProblem(a, L)
        return problem.solve()

    def load_initial_values(self):
        """Initialize variables"""
        chemo = self
        eps = chemo.eps
        chi_coef, xi_coef = chemo.chi_coef, chemo.xi_coef
        max_u0 = float(chemo.parameters.max_u0)
        exp_u0 = float(chemo.parameters.exp_u0)

        #
        # Initial condition
        #
        x = SpatialCoordinate(chemo.mesh)
    
        u_init = max_u0 * exp(-exp_u0 * np.dot(x,x))
        v_init = 10 * exp(-35 * np.dot(x,x))
        w_init = 10 * exp(-35 * np.dot(x,x))

        printMPI("u0 =", str(u_init))
        if chemo.tau:
            printMPI("v0 =", str(v_init))
            if xi_coef>DOLFIN_EPS:
                printMPI("w0 =", str(w_init))

        u_init_expr = Expression(u_init, chemo.Uh.element.interpolation_points())
        v_init_expr = Expression(v_init, chemo.Vh1.element.interpolation_points())
        w_init_expr = Expression(w_init, chemo.Vh1.element.interpolation_points())

        # Initial values
        chemo.u0.interpolate(u_init_expr)
        chemo.u0.x.scatter_forward()
        u0 = chemo.u0

        chemo.v0.interpolate(v_init_expr)
        chemo.v0.x.scatter_forward()
        if int(chemo.parameters.nolocal):
            chemo.v0.x.array[:] = chemo.v0.x.array[:] - 1.0/chemo.domain_size * assemble_scalar(form(chemo.v0 * dx))
        v0 = chemo.v0

        if xi_coef>DOLFIN_EPS:
            chemo.w0.interpolate(w_init_expr)
            chemo.w0.x.scatter_forward()
            if int(chemo.parameters.nolocal):
                chemo.w0.x.array[:] = chemo.w0.x.array[:] - 1.0/chemo.domain_size * assemble_scalar(form(chemo.w0 * dx))
        else:
            chemo.w0.x.array[:] = 0.0
            chemo.w0.x.scatter_forward()
        w0 = chemo.w0

        chemo.mu0.x.array[:] = chemo.project(ln(u0+eps), chemo.Vh1).x.array
        chemo.mu0.x.scatter_forward()
        mu0 = chemo.mu0

        p1c_u0 = chemo.p1c_u0 = chemo.project(u0, chemo.Vh1, mass_lumping=True)

    def variational_problem_v(self):
        #
        # Load variables from chemo problem
        #
        chemo = self
        dt = chemo.dt
        a_coef, b_coef = chemo.a_coef, chemo.b_coef
        alpha_exp = chemo.alpha_exp
        v_trial, vb = chemo.v_trial, chemo.vb
        u0, v0 = chemo.u0, chemo.v0
        tau = chemo.tau

        domain_size = chemo.domain_size

        if int(chemo.parameters.nolocal):
            if tau:
                a_v = v_trial * vb * dx \
                    + dt/tau * inner(grad(v_trial), grad(vb)) * dx
                L_v = v0 * vb * dx \
                    - dt/tau * 1.0/domain_size * assemble_scalar(form(u0**alpha_exp * dx)) * vb * dx \
                    + dt/tau * u0**alpha_exp * vb * dx            
            else:
                a_v = inner(grad(v_trial), grad(vb)) * dx
                L_v = - 1.0/domain_size * assemble_scalar(form(u0**alpha_exp * dx)) * vb * dx \
                    + u0**alpha_exp * vb * dx
        else:
            if tau:
                a_v = v_trial * vb * dx_ML \
                    + dt/tau * inner(grad(v_trial), grad(vb)) * dx \
                    + dt/tau * a_coef * v_trial * vb * dx_ML
                L_v = v0 * vb * dx_ML \
                    + dt/tau * b_coef * u0**alpha_exp * vb * dx
            
            else:
                a_v = inner(grad(v_trial), grad(vb)) * dx \
                    + a_coef * v_trial * vb * dx_ML
                L_v = b_coef * u0**alpha_exp * vb * dx
        
        chemo.a_v = a_v
        chemo.L_v = L_v

    def variational_problem_w(self):
        #
        # Load variables from chemo problem
        #
        chemo = self
        dt = chemo.dt
        e_coef, d_coef = chemo.e_coef, chemo.d_coef
        beta_exp = chemo.beta_exp
        w_trial, wb = chemo.w_trial, chemo.wb
        u0, w0 = chemo.u0, chemo.w0
        tau = chemo.tau

        domain_size = chemo.domain_size

        if int(chemo.parameters.nolocal):
            if tau:
                a_w = w_trial * wb * dx \
                    + dt/tau * inner(grad(w_trial), grad(wb)) * dx
                L_w = w0 * wb * dx \
                    - dt/tau * 1.0/domain_size * assemble_scalar(form(u0**beta_exp * dx)) * wb * dx \
                    + dt/tau * u0**beta_exp * wb * dx            
            else:
                a_w = inner(grad(w_trial), grad(wb)) * dx
                L_w = - 1.0/domain_size * assemble_scalar(form(u0**beta_exp * dx)) * wb * dx \
                    + u0**beta_exp * wb * dx
        else:
            if tau:
                a_w = w_trial * wb * dx_ML \
                    + dt/tau * inner(grad(w_trial), grad(wb)) * dx \
                    + dt/tau * d_coef * w_trial * wb * dx_ML
                L_w = w0 * wb * dx_ML \
                    + dt/tau * e_coef * u0**beta_exp * wb * dx
            
            else:
                a_w = inner(grad(w_trial), grad(wb)) * dx \
                    + d_coef * w_trial * wb * dx_ML
                L_w = e_coef * u0**beta_exp * wb * dx
        
        chemo.a_w = a_w
        chemo.L_w = L_w

    def variational_problem_mu(self):
        """Build variational problem"""
        #
        # Load variables from chemo problem
        #
        chemo = self
        mu_trial, mub = chemo.mu_trial, chemo.mub
        v, w = chemo.v, chemo.w
        u0 = chemo.u0
        eps, chi_coef, xi_coef = chemo.eps, chemo.chi_coef, chemo.xi_coef

        a_mu =  mu_trial * mub * dx
        L_mu = ln(u0 + eps) * mub * dx
        
        chemo.a_mu = a_mu
        chemo.L_mu = L_mu

    def variational_problem_u(self):
        """Build variational problem"""
        #
        # Load variables from chemo problem
        #
        chemo = self
        dt = chemo.dt
        nx = chemo.nx
        u_trial, ub = chemo.u_trial, chemo.ub
        mu, v, w = chemo.mu, chemo.v, chemo.w
        chi_coef, xi_coef = chemo.chi_coef, chemo.xi_coef
        p1c_u0 = chemo.p1c_u0
        u0 = chemo.u0
        eps, chi_coef, xi_coef, lambda_coef, mu_coef, c_coef = chemo.eps, chemo.chi_coef, chemo.xi_coef, chemo.lambda_coef, chemo.mu_coef, chemo.c_coef
        m1_exp, m2_exp, m3_exp = chemo.m1_exp, chemo.m2_exp, chemo.m3_exp
        rho_exp, k_exp, gamma_exp = chemo.rho_exp, chemo.k_exp, chemo.gamma_exp

        def pos(u):
            return ((abs(u) + u) / 2.0)
        chemo.pos = pos
        def neg(u):
            return ((abs(u) - u) / 2.0)

        #
        # Variational problem
        #
        e_len = FacetArea(chemo.mesh)
        n_e = chemo.n_e = FacetNormal(chemo.mesh)
        l = 1.0/nx

        if chemo.normal_gradient:
            def aupw(mu, u, f_u0, ub):
                # UPW bilinear form
                return (
                    pos(inner(avg(f_u0 * grad(mu)), n_e('+'))) * u('+') * jump(ub) * dS \
                    - neg(inner(avg(f_u0 * grad(mu)), n_e('+'))) * u('-') * jump(ub) * dS
                )

        a_u = inner(u_trial, ub) * dx \
            + dt * aupw(-mu, u_trial, (u0+1.0)**(m1_exp-1.0), ub) \
            + dt * chi_coef * aupw(v, u_trial, (u0+1.0)**(m2_exp-1.0), ub) \
            + dt * mu_coef * u_trial * u0**(k_exp-1) * ub * dx

        if xi_coef>DOLFIN_EPS:
            a_u = a_u + dt * xi_coef * aupw(-w, u_trial, (u0+1.0)**(m3_exp-1.0), ub)
        if c_coef >= 1e-16:
            a_u = a_u + dt * c_coef * u_trial * u0**(gamma_exp-1) * inner(grad(mu), grad(mu))**(gamma_exp/2) * ub * dx 

        L_u = inner(u0, ub) * dx \
            + dt * lambda_coef**rho_exp * u0 * ub * dx

        chemo.a_u = a_u
        chemo.L_u = L_u

    def build_systems(self, verbosity=0):
        #
        # Load variables from chemo problem
        #
        chemo = self

        #
        # PETSc options
        #
        petsc_options = {"ksp_type": "gmres", "pc_type": "ilu"}
        if p.test == '2d':
            petsc_options_2 = {"ksp_type": "gmres", "pc_type": "lu"}
        else:
            petsc_options_2 = {"ksp_type": "gmres", "pc_type": "ilu"}
        if verbosity:
            petsc_options["ksp_view"] = None
            petsc_options["ksp_monitor"] = None

            petsc_options_2["ksp_view"] = None
            petsc_options_2["ksp_monitor"] = None

        #
        # Nullspace for nonlocal case
        #
        if int(chemo.parameters.nolocal) and (chemo.tau==0):
            nsp = PETSc.NullSpace().create(constant=True)

        #
        # Define problems
        #
        chemo.problem_v = LinearProblem(chemo.a_v, chemo.L_v, petsc_options=petsc_options)
        if int(chemo.parameters.nolocal) and (chemo.tau==0):
            chemo.problem_v.A.setNullSpace(nsp)

        if chemo.xi_coef > DOLFIN_EPS:
            chemo.problem_w = LinearProblem(chemo.a_w, chemo.L_w, petsc_options=petsc_options)
            if int(chemo.parameters.nolocal) and (chemo.tau==0):
                chemo.problem_w.A.setNullSpace(nsp)

        chemo.problem_mu = LinearProblem(chemo.a_mu, chemo.L_mu, petsc_options=petsc_options)

        chemo.problem_u = LinearProblem(chemo.a_u, chemo.L_u, petsc_options=petsc_options_2)
    
    def time_iterator(self, tsteps=1, first_step=1, verbosity=0):
        """Time iterator"""

        #
        # Run time iterations
        #
        step = first_step - 1

        #
        # Initialization
        #
        last_step = first_step + tsteps

        while step < last_step:
            if step == first_step - 1:
                v0 = chemo.v0
                w0 = chemo.w0
                mu0 = chemo.mu0
                u0 = chemo.u0
                p1c_u0 = chemo.p1c_u0
                # --- Yield initial data
                yield {'step': step, 't': chemo.t, 'v': v0, 'w': w0, 'u': u0, 'mu': mu0, 'p1c_u': p1c_u0}

            else:
                chemo.t += chemo.dt

                # Solve v
                chemo.v.x.array[:] = chemo.problem_v.solve().x.array
                chemo.v.x.scatter_forward()
                if int(chemo.parameters.nolocal) and (chemo.tau==0):
                    chemo.v.x.array[:] = chemo.v.x.array[:] - 1.0/chemo.domain_size * assemble_scalar(form(chemo.v * dx))
                    chemo.v.x.scatter_forward()

                # Solve w
                if chemo.xi_coef > DOLFIN_EPS:
                    chemo.w.x.array[:] = chemo.problem_w.solve().x.array
                    chemo.w.x.scatter_forward()
                    if int(chemo.parameters.nolocal) and (chemo.tau==0):
                        chemo.w.x.array[:] = chemo.w.x.array[:] - 1.0/chemo.domain_size * assemble_scalar(form(chemo.w * dx))
                        chemo.w.x.scatter_forward()
                else:
                    chemo.w.x.array[:] = 0.0

                # Solve mu
                chemo.mu.x.array[:] = chemo.problem_mu.solve().x.array
                chemo.mu.x.scatter_forward()

                # Solve u
                chemo.u.x.array[:] = chemo.problem_u.solve().x.array
                chemo.u.x.scatter_forward()

                # --- Save solution (to be used in next iteration)
                v, v0 = chemo.v, chemo.v0
                w, w0 = chemo.w, chemo.w0
                mu, mu0 = chemo.mu, chemo.mu0
                u, u0 = chemo.u, chemo.u0
                p1c_u, p1c_u0 = chemo.project(chemo.u, chemo.Vh1, mass_lumping=True), chemo.p1c_u0

                # --- Yield data computed in current iteration
                yield {'step': step, 't': chemo.t, 'v': v, 'w': w, 'u': u, 'mu': mu, 'p1c_u': p1c_u, 'v0': v0, 'w0':w0, 'u0': u0, 'mu0': mu0, 'p1c_u0': p1c_u0}

                # --- Update solution
                chemo.v0.x.array[:] = chemo.v.x.array
                chemo.v0.x.scatter_forward()
                chemo.w0.x.array[:] = chemo.w.x.array
                chemo.w0.x.scatter_forward()
                chemo.mu0.x.array[:] = chemo.mu.x.array
                chemo.mu0.x.scatter_forward()
                chemo.u0.x.array[:] = chemo.u.x.array
                chemo.u0.x.scatter_forward()
                chemo.p1c_u0.x.array[:] = chemo.p1c_u.x.array
                chemo.p1c_u0.x.scatter_forward()

            step = step + 1


# ---------------------------

def print_info(i, t, u_data, p1c_u_data, v_data, w_data, jumps_gr_data):
# energy, dynamics = 0):
    u_max, u_min, u_mass = u_data
    p1c_u_max, p1c_u_min, p1c_u_mass = p1c_u_data
    v_max, v_min, v_mass = v_data
    w_max, w_min, w_mass = w_data
    jump_grmu, jump_grv, jump_grw = jumps_gr_data
    s = f"{i:3} {t:.6e} {u_max:.4e} {p1c_u_max:.4e} {v_max:.4e} {w_max:.4e}"
    s += f" {u_min:.4e} {p1c_u_min:.4e} {v_min:.4e} {w_min:.4e}"
    s += f" {u_mass:.4e} {p1c_u_mass:.4e} {v_mass:.4e} {w_mass:.4e}"
    s += f" {jump_grmu:.4e} {jump_grv:.4e} {jump_grw:.4e}"
    printMPI(s)


def define_parameters():

    parser = argparse.ArgumentParser()

    # Define remaining parameters
    parser.add_argument('--eps', default=1e-5)

    parser.add_argument('--chi_coef', default=1.0)
    parser.add_argument('--xi_coef', default=1.0)
    parser.add_argument('--lambda_coef', default=1.0)
    parser.add_argument('--mu_coef', default=1.0)
    parser.add_argument('--c_coef', default=0.0)

    parser.add_argument('--a_coef', default=1.0)
    parser.add_argument('--b_coef', default=1.0)
    parser.add_argument('--e_coef', default=1.0)
    parser.add_argument('--d_coef', default=1.0)

    parser.add_argument('--rho_exp', default=1.0)
    parser.add_argument('--k_exp', default=1.1)
    parser.add_argument('--gamma_exp', default=1.75)
    parser.add_argument('--alpha_exp', default=1.0)
    parser.add_argument('--beta_exp', default=1.0)

    parser.add_argument('--m1_exp', default=1.0)
    parser.add_argument('--m2_exp', default=1.0)
    parser.add_argument('--m3_exp', default=1.0)

    parser.add_argument('--tau', default=0.0)

    parser.add_argument('--test', choices=['2d', '3d'], default='2d')
    parser.add_argument('--nolocal', default=1, help="Use nonlocal model")
    parser.add_argument('--mesh', default="circle")
    parser.add_argument('--max_u0', default=100)
    parser.add_argument('--exp_u0', default=35)

    # Params for the discrete scheme
    parser.add_argument('--nx', default=50)
    parser.add_argument('--dt', default=1e-4)
    parser.add_argument('--tsteps', default=200)
    
    parser.add_argument('--normal_grad', default=1, help="Use normal gradient instead of its approximation by jump")

    # Other parameters
    parser.add_argument('--verbosity', default=0, help="No extra information shown")
    parser.add_argument('--plot', default=10, help="Plot shown every number of time steps")
    parser.add_argument('--plot_mesh', default=0, help="Plot mesh")
    parser.add_argument('--vtk', default=0, help="No vtk photogram saved to disk")
    parser.add_argument('--vtkfile', default="chemo_DG-UPW", help="Name of vtk file")
    parser.add_argument('--save', default=1, help="No figures and output saved")
    parser.add_argument('--savefile', default="chemo_DG-UPW", help="Name of output file")
    parser.add_argument('--savefunc', default=0, help="No functions saved in HDF5 format")
    parser.add_argument('--server', default=0, help="Set to 1 if the code is set to run on a server")

    param = parser.parse_args()

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
    for k, v in vars(parameters).items():
        printMPI(f"  {k} = {v}")
    
    if int(p.verbosity):
        log.set_log_level(log.LogLevel.INFO)
        opts = PETSc.Options()
        opts["ksp_monitor"] = True
    else:
        log.set_log_level(log.LogLevel.ERROR)

    #
    # Init problem
    #
    chemo = chemo_DG_UPW(parameters)
    chemo.load_initial_values()
    chemo.variational_problem_v()
    if chemo.xi_coef > DOLFIN_EPS:
        chemo.variational_problem_w()
    chemo.variational_problem_mu()
    chemo.variational_problem_u()
    chemo.build_systems(verbosity=int(p.verbosity))

    #
    # Save output
    #
    do_save = bool(p.save)
    server = int(p.server)
    base_name_save = p.savefile
    savefunc = int(p.savefunc)

    #
    # Save mesh to XDMF
    #
    if savefunc:
         with XDMFFile(comm, f"{base_name_save}.xdmf", "w") as xdmf:
            xdmf.write_mesh(chemo.mesh)

    #
    # Plot
    #
    if server:
        pyvista.start_xvfb()
    if do_save:
        pyvista.OFF_SCREEN = True

    do_plot = (int(p.plot) > 0)
    plot_mesh = (int(p.plot_mesh) > 0)
    pyvista.set_plot_theme("document")

    if plot_mesh: # Plot mesh
        topology, cell_types, geometry = plot.vtk_mesh(chemo.mesh, chemo.mesh.topology.dim)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        plotter = pyvista.Plotter()
        plotter.add_mesh(grid, show_edges=True, color="white")
        plotter.view_xy()
        if pyvista.OFF_SCREEN:
            plotter.screenshot("mesh.png", transparent_background=True)

            comm.Barrier()
            if rank == 0:
                img = Image.open(f"mesh.png")
                width, height = img.size
                # Setting the points for cropped image
                left = width/6
                top = 0.08 * height
                right = 5 * width/6
                bottom = 0.92 * height
                im_cropped = img.crop((left, top, right, bottom)) # default window size is 1024x768
                im_cropped.save(f"mesh.png")
                im_cropped.close()
                img.close()
            comm.Barrier()
        else:
            plotter.show()
        plotter.close()

    #
    # More info
    #  
    printMPI("\nMore info:")
    tdim = chemo.mesh.topology.dim
    num_cells = chemo.mesh.topology.index_map(tdim).size_local
    list_h = dolfinx.cpp.mesh.h(
            chemo.mesh._cpp_object, 2, np.arange(chemo.mesh.topology.index_map(2).size_local,
                                        dtype=np.int32))
    h_max = comm.allreduce(max(list_h),op=MPI.MAX)
    print("h = ", h_max)
    printMPI("  Max mass values:")
    printMPI(f"\t\tint u0 = {assemble_scalar(form(chemo.u0*dx))}")
    printMPI(f"\t\tcoef = {(chemo.lambda_coef/(chemo.mu_coef * chemo.domain_size**(chemo.rho_exp-chemo.k_exp)))**(1/(chemo.k_exp - chemo.rho_exp))}")

    #
    # Save max, min and energy
    #
    max_u_list = []
    min_u_list = []
    max_p1c_u_list = []
    min_p1c_u_list = []
    max_v_list = []
    min_v_list = []
    max_w_list = []
    min_w_list = []

    #
    # Print info
    #
    printMPI("Time steps:\n  i t u_max p1c_u_max v_max w_max u_min p1c_u_min v_min w_min u_mass p1c_u_mass v_mass w_mass jump_grmu jump_grv jump_grw")

    #
    # Time iterations
    #
    chemo_iterations = chemo.time_iterator(tsteps=int(p.tsteps), verbosity=int(p.verbosity))
    
    for t_step in chemo_iterations:

        i, t = t_step['step'], t_step['t']
        u, p1c_u, mu, v, w = t_step['u'], t_step['p1c_u'], t_step['mu'], t_step['v'], t_step['w']

        #
        # Print info
        #
        u_max, u_min = comm.allreduce(max(u.x.array), MPI.MAX), comm.allreduce(min(u.x.array), MPI.MIN)
        p1c_u_max, p1c_u_min = comm.allreduce(max(p1c_u.x.array), MPI.MAX), comm.allreduce(min(p1c_u.x.array), MPI.MIN)
        v_max, v_min = comm.allreduce(max(v.x.array), MPI.MAX), comm.allreduce(min(v.x.array), MPI.MIN)
        w_max, w_min = comm.allreduce(max(w.x.array), MPI.MAX), comm.allreduce(min(w.x.array), MPI.MIN)
        u_mass, p1c_u_mass = assemble_scalar(form(u*dx)), assemble_scalar(form(p1c_u *dx))
        v_mass, w_mass = assemble_scalar(form(v*dx)), assemble_scalar(form(w*dx))
        jump_grmu = assemble_scalar(form(abs(inner(jump(grad(mu)),chemo.n_e('+')))*dS))/chemo.num_edges
        jump_grv = assemble_scalar(form(abs(inner(jump(grad(v)),chemo.n_e('+')))*dS))/chemo.num_edges
        jump_grw = assemble_scalar(form(abs(inner(jump(grad(w)),chemo.n_e('+')))*dS))/chemo.num_edges

        if rank == 0:
            max_u_list.append(u_max)
            min_u_list.append(u_min)
            max_p1c_u_list.append(p1c_u_max)
            min_p1c_u_list.append(p1c_u_min)
            max_v_list.append(v_max)
            min_v_list.append(v_min)
            max_w_list.append(w_max)
            min_w_list.append(w_min)

        if rank == 0:
            print_info(i, t,
                    (u_max, u_min, u_mass),
                    (p1c_u_max, p1c_u_min, p1c_u_mass),
                    (v_max, v_min, v_mass),
                    (w_max, w_min, w_mass),
                    (jump_grmu, jump_grv, jump_grw))
            
        #
        # Plot
        #
        if (do_plot and i % int(p.tsteps) % int(p.plot) == 0):

            # Properties of the scalar bar
            if chemo.parameters.test == '2d':
                sargs = dict(height=0.6, vertical=True, position_x=0.75, position_y=0.2, title='', label_font_size=24, shadow=True,n_labels=5, fmt="%.2g", font_family="arial")
            else:
                sargs = dict(height=0.6, vertical=True, position_x=0.85, position_y=0.26, title='', label_font_size=24, shadow=True,n_labels=5, fmt="%.2g", font_family="arial")

            # Create a grid to attach the DoF values
            cells, types, x = plot.vtk_mesh(chemo.Vh1)
            grid = pyvista.UnstructuredGrid(cells, types, x)
            grid.point_data["Pi1_u"] = p1c_u.x.array[:]

            grid.set_active_scalars("Pi1_u")

            plotter = pyvista.Plotter()
            if chemo.parameters.test == '2d':
                warped = grid.warp_by_scalar(factor=1.75*1/(np.max([max_p1c_u_list[-1], 1])))
                plotter.add_mesh(warped, show_edges=False, show_scalar_bar=True, cmap=mpl.colormaps["coolwarm"], scalar_bar_args=sargs)
                plotter.view_xz()
            else:
                clipped = grid.clip('y')
                plotter.add_mesh(clipped, show_edges=False, show_scalar_bar=True, cmap=mpl.colormaps["coolwarm"], scalar_bar_args=sargs)
                plotter.view_xy()
                plotter.camera.elevation = 45.0 # Angle of view
                plotter.camera.zoom(1.2)

            # If environment variable PYVISTA_OFF_SCREEN is set to true save a png
            # otherwise create interactive plot
            if pyvista.OFF_SCREEN:
                plotter.screenshot(f"./{base_name_save}_Pi1_u_i-{i}.png", transparent_background=True)
                
                comm.Barrier()
                if rank == 0:
                    img = Image.open(f"./{base_name_save}_Pi1_u_i-{i}.png")
                    width, height = img.size
                    # Setting the points for cropped image
                    if chemo.parameters.test == "2d":
                        left = width/4
                        top = 0.15 * height
                        right = 0.9 * width
                        bottom = 0.85 * height
                    else:
                        left = width/6
                        top = 0.08 * height
                        right = width
                        bottom = 0.8 * height
                    im_cropped = img.crop((left, top, right, bottom)) # default window size is 1024x768
                    im_cropped.save(f"./{base_name_save}_Pi1_u_i-{i}.png")
                    im_cropped.close()
                    img.close()
                comm.Barrier()
            else:
                plotter.show()
            plotter.close()

            grid.point_data["v"] = v.x.array[:]

            grid.set_active_scalars("v")

            plotter = pyvista.Plotter()
            sargs["color"] = "black"
            if chemo.parameters.test == '2d':
                warped = grid.warp_by_scalar(factor=1.75*1/(np.max([max_v_list[-1], 1])))
                plotter.add_mesh(warped, show_edges=False, show_scalar_bar=True, cmap=mpl.colormaps["Blues"], scalar_bar_args=sargs)
                plotter.view_xz()
            else:
                clipped = grid.clip('y')
                plotter.add_mesh(clipped, show_edges=False, show_scalar_bar=True, cmap=mpl.colormaps["Blues"], scalar_bar_args=sargs)
                plotter.view_xy()
                plotter.camera.elevation = 45.0 # Angle of view
                plotter.camera.zoom(1.2)

            # If environment variable PYVISTA_OFF_SCREEN is set to true save a png
            # otherwise create interactive plot
            if pyvista.OFF_SCREEN:
                plotter.screenshot(f"./{base_name_save}_v_i-{i}.png", transparent_background=True)

                comm.Barrier()
                if rank == 0:
                    img = Image.open(f"./{base_name_save}_v_i-{i}.png")
                    width, height = img.size
                    # Setting the points for cropped image
                    if chemo.parameters.test == "2d":
                        left = width/4
                        top = 0.15 * height
                        right = 0.9 * width
                        bottom = 0.85 * height
                    else:
                        left = width/6
                        top = 0.08 * height
                        right = width
                        bottom = 0.8 * height
                    im_cropped = img.crop((left, top, right, bottom)) # default window size is 1024x768
                    im_cropped.save(f"./{base_name_save}_v_i-{i}.png")
                    im_cropped.close()
                    img.close()
                comm.Barrier()
            else:
                plotter.show()
            plotter.close()

            if chemo.xi_coef > DOLFIN_EPS:
                grid.point_data["w"] = w.x.array[:]

                grid.set_active_scalars("w")

                plotter = pyvista.Plotter()
                if chemo.parameters.test == '2d':
                    warped = grid.warp_by_scalar(factor=1.75*1/(np.max([max_w_list[-1], 1])))
                    plotter.add_mesh(warped, show_edges=False, show_scalar_bar=True, cmap=mpl.colormaps["Reds"], scalar_bar_args=sargs)
                    plotter.view_xz()
                else:
                    clipped = grid.clip('y')
                    plotter.add_mesh(clipped, show_edges=False, show_scalar_bar=True, cmap=mpl.colormaps["Reds"], scalar_bar_args=sargs)
                    plotter.view_xy()
                    plotter.camera.elevation = 45.0 # Angle of view
                    plotter.camera.zoom(1.2)

                # If environment variable PYVISTA_OFF_SCREEN is set to true save a png
                # otherwise create interactive plot
                if pyvista.OFF_SCREEN:
                    plotter.screenshot(f"./{base_name_save}_w_i-{i}.png", transparent_background=True)

                    comm.Barrier()
                    if rank == 0:
                        img = Image.open(f"./{base_name_save}_w_i-{i}.png")
                        width, height = img.size
                        # Setting the points for cropped image
                        if chemo.parameters.test == "2d":
                            left = width/4
                            top = 0.15 * height
                            right = 0.9 * width
                            bottom = 0.85 * height
                        else:
                            left = width/6
                            top = 0.08 * height
                            right = width
                            bottom = 0.8 * height
                        im_cropped = img.crop((left, top, right, bottom)) # default window size is 1024x768
                        im_cropped.save(f"./{base_name_save}_w_i-{i}.png")
                        im_cropped.close()
                        img.close()
                    comm.Barrier()
                else:
                    plotter.show()
                plotter.close()

        #
        # Save functions to XDMF
        #
        if (savefunc and i % int(p.tsteps) % savefunc == 0):
            with dolfinx.io.XDMFFile(comm, f"{base_name_save}.xdmf", "a") as xdmf:
                u.name = "u"
                p1c_u.name = "p1c_u"
                v.name = "v"
                w.name = "w"
                xdmf.write_function(u, t=t)
                xdmf.write_function(p1c_u, t=t)
                xdmf.write_function(v, t=t)
                xdmf.write_function(w, t=t)

    #
    # Plot
    #
    if do_plot:
        time_steps = np.linspace(0, t, int(p.tsteps) + 1)
        fig, axs = plt.subplots(2)
        axs[0].plot(time_steps,np.full(int(p.tsteps) + 1, 1),'-',c='lightgray',linewidth=2,label='_nolegend_')
        axs[0].plot(time_steps,max_u_list,'--',c='orange')
        axs[1].plot(time_steps,np.zeros(int(p.tsteps) + 1),'-',c='lightgray',linewidth=2,label='_nolegend_')
        axs[1].plot(time_steps,min_u_list,'--',c='orange')
        plt.subplots_adjust(hspace=0.5, bottom=0.16)
        if do_save: plt.savefig(f"{base_name_save}_min-max_u.png")
        else: plt.show()
        plt.close()

        time_steps = np.linspace(0, t, int(p.tsteps) + 1)
        fig, axs = plt.subplots(2)
        axs[0].plot(time_steps,np.full(int(p.tsteps) + 1, 1),'-',c='lightgray',linewidth=2,label='_nolegend_')
        axs[0].plot(time_steps,max_p1c_u_list,'--',c='orange')
        axs[1].plot(time_steps,np.zeros(int(p.tsteps) + 1),'-',c='lightgray',linewidth=2,label='_nolegend_')
        axs[1].plot(time_steps,min_p1c_u_list,'--',c='orange')
        plt.subplots_adjust(hspace=0.5, bottom=0.16)
        if do_save: plt.savefig(f"{base_name_save}_min-max_p1c_u.png")
        else: plt.show()
        plt.close()

        fig, axs = plt.subplots(2)
        axs[0].plot(time_steps,np.full(int(p.tsteps) + 1, 1),'-',c='lightgray',linewidth=2,label='_nolegend_')
        axs[0].plot(time_steps,max_v_list,'--',c='orange')
        axs[1].plot(time_steps,np.zeros(int(p.tsteps) + 1),'-',c='lightgray',linewidth=2,label='_nolegend_')
        axs[1].plot(time_steps,min_v_list,'--',c='orange')
        plt.subplots_adjust(hspace=0.5, bottom=0.16)
        if do_save: plt.savefig(f"{base_name_save}_min-max_v.png")
        else: plt.show()
        plt.close()

        if chemo.xi_coef > DOLFIN_EPS:
            fig, axs = plt.subplots(2)
            axs[0].plot(time_steps,np.full(int(p.tsteps) + 1, 1),'-',c='lightgray',linewidth=2,label='_nolegend_')
            axs[0].plot(time_steps,max_w_list,'--',c='orange')
            axs[1].plot(time_steps,np.zeros(int(p.tsteps) + 1),'-',c='lightgray',linewidth=2,label='_nolegend_')
            axs[1].plot(time_steps,min_w_list,'--',c='orange')
            plt.subplots_adjust(hspace=0.5, bottom=0.16)
            if do_save: plt.savefig(f"{base_name_save}_min-max_w.png")
            else: plt.show()
            plt.close()