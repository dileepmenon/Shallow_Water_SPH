# Numpy
from numpy import ( ones_like, zeros, zeros_like, mgrid, pi,
                    arange, sqrt, concatenate, sin, cos, where )

# PySPH base
from pysph.base.kernels import CubicSpline

# PySPH solver and integrator
from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.integrator_step import IntegratorStep

# PySPH equations
from pysph.sph.equation import Group
from equation.SWE import *
from equation.SWE import get_particle_array_swe as gpa_swe

# PySPH Evaluator
from pysph.tools.sph_evaluator import SPHEvaluator


# Constants
rho_w = 1000.0
g = 9.81
hdx = 1.3
d = 1.0
dx = 0.025
dim = 2


class RectangularDamBreak(Application):
    def create_particles(self):
        """Create the Rectangular patch of fluid."""
        x, y = mgrid[-2:2+1e-4:dx, -0.5:0.5+1e-4:dx]
        x = x.ravel()
        y = y.ravel()

        m = ones_like(x) * dx * dx * rho_w * d
        h = ones_like(x) * hdx * dx
        h0 = ones_like(x) * hdx * dx

        rho = ones_like(x) * rho_w * d
        rho0 = ones_like(x) * rho_w * d

        pa = gpa_swe(x=x, y=y, m=m, rho=rho, rho0=rho0, h=h, h0=h0,
                     name='fluid')

        compute_initial_props([pa])
        return [pa]

    def create_solver(self):
        kernel = CubicSpline(dim=2)
        integrator = SWEIntegrator(fluid=SWEStep())
        tf = 2
        solver = Solver(
            kernel=kernel,
            dim=2,
            integrator=integrator,
            cfl=0.3,
            adaptive_timestep=True,
            output_at_times=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            tf=tf
            )
        return solver

    def create_equations(self):
        equations = [
            Group(
                equations=[
                    InitialGuessDensity(dim, dest='fluid', sources=['fluid',],),
                    UpdateSmoothingLength(dim, dest='fluid')
                    ], update_nnps=True
                ),
            Group(
                equations=[
                    Group(
                        equations=[
                            CorrectionFactorVariableSmoothingLength(
                                dest='fluid', sources=['fluid',]
                                ),
                            SummationDensity(dest='fluid', sources=['fluid',]),
                            ]
                        ),
                    Group(
                        equations=[
                            DensityResidual('fluid'),
                            DensityNewtonRaphsonIteration(dim, dest='fluid')
                            ]
                        ),
                    Group(
                        equations=[
                            UpdateSmoothingLength(dim, dest='fluid')
                            ], update_nnps=True
                        ),
                    Group(
                        equations=[
                            SummationDensity(dest='fluid', sources=['fluid',]),
                            ],
                        ),
                    Group(
                        equations=[
                            DensityResidual(dest='fluid'),
                            CheckConvergence(dest='fluid')
                            ],
                    )], iterate=True, max_iterations=100
            ),
            Group(
                equations=[
                    CorrectionFactorVariableSmoothingLength(
                        dest='fluid', sources=['fluid',]),
                    ]
                ),
            Group(
                equations=[
                    SWEOS(dest='fluid'),
                    ParticleAccelerations(dim, dest='fluid', sources=['fluid',],
                                          u_only=True),
                    ],
                ),
            ]
        return equations


def compute_initial_props(particles):
    one_time_equations = [
        Group(
            equations=[
                CorrectionFactorVariableSmoothingLength(dest='fluid',
                                                        sources=['fluid',]),
                SWEOS(dest='fluid'),
                ], update_nnps=False
            )
    ]
    kernel = CubicSpline(dim=2)
    sph_eval = SPHEvaluator(particles, one_time_equations, dim=2,
                            kernel=kernel)
    sph_eval.evaluate()


if __name__ == '__main__':
    app = RectangularDamBreak()
    app.run()
