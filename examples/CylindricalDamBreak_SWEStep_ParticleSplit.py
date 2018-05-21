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
hdx = 2.0
d = 1.0
n = 50
dr = 0.5 / n
dx = dr
dim = 2
A_max = 6.0 * (1.56*dx**2)
h_max = 3.0 * hdx * dx


class CircularDamBreak(Application):
    def create_particles(self):
        """Create the circular patch of fluid."""
        x = zeros(0)
        y = zeros(0)
        m = zeros(0)

        # Create circular patch in a radial grid
        rad = 0.0
        for j in range(1, n+1):
                npnts = 4 * j
                dtheta = (2*pi) / npnts

                theta = arange(0, 2*pi-1e-10, dtheta)
                rad = rad + dr

                _x = rad * cos(theta)
                _y = rad * sin(theta)

                x = concatenate( (x, _x) )
                y = concatenate( (y, _y) )

        m = ones_like(x) * (1.56*dr*dr) * rho_w * d

        rho = ones_like(x) * rho_w * d
        rho0 = ones_like(x) * rho_w * d

        h = ones_like(x) * hdx * dx
        h0 = ones_like(x) * hdx * dx

        pa = gpa_swe(x=x, y=y, m=m, rho=rho, rho0=rho0, h=h, h0=h0,
                     name='fluid')

        compute_initial_props([pa])
        return [pa]


    def create_solver(self):
        kernel = CubicSpline(dim=2)
        integrator = SWEIntegrator(fluid=SWEStep())
        tf = 0.3
        solver = Solver(
            kernel=kernel,
            dim=2,
            integrator=integrator,
            cfl=0.3,
            adaptive_timestep=True,
            output_at_times=(0.1, 0.2, 0.3, 0.4, 0.5),
            tf=tf
            )
        return solver


    def create_equations(self):
        equations = [
            Group(
                equations=[
                    Group(
                        equations=[
                            GatherDensityEvalNextIteration(
                                dest='fluid', sources=['fluid',]
                                ),
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
                    )], iterate=True, max_iterations=10
            ),
            Group(
                equations=[
                    CorrectionFactorVariableSmoothingLength(
                        dest='fluid', sources=['fluid',]),
                    ]
                ),
            Group(
                equations=[
                    DaughterVelocityEval(rho_w, dest='fluid',
                                         sources=['fluid',]),
                    ]
                ),
            Group(
                equations=[
                    SWEOS(dest='fluid'),
                    ]
                ),
            Group(
                equations=[
                    ParticleAcceleration(dim, dest='fluid', sources=['fluid',],
                                        ),
                    ]
                ),
            Group(
                equations=[
                    CheckForParticlesToSplit(dest='fluid', h_max=h_max,
                                             A_max=A_max)
                    ],
                ),
            ]
        return equations

    def pre_step(self, solver):
        for pa in self.particles:
            ps = ParticleSplit(pa)
            ps.do_particle_split()
        self.nnps.update()


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
    app = CircularDamBreak()
    app.run()
