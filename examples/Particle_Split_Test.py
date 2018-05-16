# Numpy
import numpy
from numpy import ( ones_like, zeros, zeros_like, mgrid, pi,
                    arange, sqrt, concatenate, sin, cos, where )

# PySPH base
from pysph.base.utils import get_particle_array as gpa
from pysph.base.kernels import CubicSpline

# PySPH solver and integrator
from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.integrator import EulerIntegrator
from pysph.sph.integrator_step import IntegratorStep

# PySPH equations
from pysph.sph.equation import Group
from equation.SWE import *
from equation.SWE import get_particle_array_swe as gpa_swe

# PySPH Evaluator
from pysph.tools.sph_evaluator import SPHEvaluator

# PySPH Interpolator
from pysph.tools.interpolator import Interpolator


# Constants
rho_w = 10000.0
g = 9.81
hdx = 1.2
d = 1.0
dx = 50
dim = 2
h_max = 2 * hdx * dx
len_fluid_domain = 1400


class ParticleSplitTest(Application):
    def create_particles(self):
        # Fluid particles
        x, y = mgrid[0:len_fluid_domain+1e-4:dx, 0:len_fluid_domain+1e-4:dx]
        x = x.ravel()
        y = y.ravel()

        idx_inner_pa_to_split = []
        for idx, (x_i, y_i) in enumerate(zip(x, y)):
            if (6*dx<=x_i<=len_fluid_domain-6*dx and
                6*dx<=y_i<=len_fluid_domain-6*dx):
                idx_inner_pa_to_split.append(idx)
        idx_inner_pa_to_split = numpy.array(idx_inner_pa_to_split)

        m = ones_like(x) * dx * dx * rho_w * d
        h = ones_like(x) * hdx * dx
        h0 = ones_like(x) * hdx * dx

        rho = ones_like(x) * rho_w * d
        rho0 = ones_like(x) * rho_w * d

        A = m / rho
        A[idx_inner_pa_to_split] = 3000

        pa = gpa_swe(x=x, y=y, m=m, rho0=rho0, rho=rho, h=h, h0=h0, A=A,
                     name='fluid')

        # Boundary Particles
        x, y = mgrid[-2*dx:len_fluid_domain+2*dx+1e-4:dx,
                     -2*dx:len_fluid_domain+2*dx+1e-4:dx]
        x = x.ravel()
        y = y.ravel()
        boun_idx = np.where( (x < 0) | (y < 0) | (x > len_fluid_domain) | (y >
                             len_fluid_domain) )
        x = x[boun_idx]
        y = y[boun_idx]
        m = ones_like(x) * dx * dx * rho_w * d
        h = ones_like(x) * hdx * dx
        rho = ones_like(x) * rho_w * d

        boundary = gpa_swe(name='boundary', x=x, y=y, m=m, h=h, rho=rho)

        compute_initial_props([pa])
        return [pa, boundary]

    def create_solver(self):
        kernel = CubicSpline(dim=2)
        integrator = EulerIntegrator(fluid=EulerStep())
        dt = 1e-4; tf = 1e-4
        solver = Solver(
            kernel=kernel,
            dim=2,
            integrator=integrator,
            dt=dt,
            tf=tf
            )
        return solver

    def create_equations(self):
        equations = [
                Group(
                    equations=[
                    Group(
                        equations=[
                            InitialGuessDensityVacondio(dim, dest='fluid',
                                sources=['fluid',]),
                                ]
                            ),
                    Group(
                        equations=[
                            GatherDensityEvalNextIteration(dest='fluid',
                                                sources=['fluid', 'boundary']),
                            ]
                        ),
                    Group(
                        equations=[
                            NonDimensionalDensityResidual(dest='fluid')
                            ]
                        ),
                    Group(
                        equations=[
                            UpdateSmoothingLength(dim, dest='fluid')
                            ], update_nnps=True
                        ),
                    Group(
                        equations=[
                            CheckConvergenceDensityResidual(dest='fluid')
                            ],
                    )], iterate=True, max_iterations=100
            ),
            ]
        return equations

    def pre_step(self, solver):
        for pa in self.particles:
            ps = ParticleSplit(pa)
            ps.do_particle_split()
        self.nnps.update()

    def post_process(self):
        rho_exact = 1e4
        rho_num = self.particles[0].rho
        print('\nMax rho is %0.3f '%max(rho_num))
        l2_err_rho = sqrt(np.sum((rho_exact - rho_num)**2)
                          / len(rho_num))
        print('L2 error in density is %0.3f \n'%l2_err_rho)


def compute_initial_props(particles):
    one_time_equations = [
                Group(
                    equations=[
                        CheckForParticlesToSplit(dest='fluid', h_max=h_max,
                            A_max=2900, x_min=300, x_max=1100, y_min=300,
                            y_max=1100)
                            ], update_nnps=False
                    )
            ]
    kernel = CubicSpline(dim=2)
    sph_eval = SPHEvaluator(particles, one_time_equations, dim=2,
                            kernel=kernel)
    sph_eval.evaluate()


if __name__ == '__main__':
    app = ParticleSplitTest()
    app.run()
    app.post_process()
