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
from pysph.sph.SWE.basic_equations import *

# PySPH Evaluator
from pysph.tools.sph_evaluator import SPHEvaluator


# Constants
rho_w = 10000.0
g = 9.81
hdx = 1.2
d = 1.0
dx = 50
h0 = hdx * dx
dim = 2


class SummationDensity(Equation):
    def initialize(self, d_idx, d_rho):
        d_rho[d_idx] = 0.0

    def loop(self, d_idx, d_rho, s_idx, s_m, WI):
        d_rho[d_idx] += s_m[s_idx] * WI



class ParticleSplitTest(Application):
    def create_particles(self):
        x, y = mgrid[0:1400+1e-4:dx, 0:1400+1e-4:dx]
        x = x.ravel()
        y = y.ravel()

        idx_inner_pa_to_split = []
        for idx, (x_i, y_i) in enumerate(zip(x, y)):
            if 300<=x_i<=1100 and 300<=y_i<=1100:
                idx_inner_pa_to_split.append(idx)
        idx_inner_pa_to_split = numpy.array(idx_inner_pa_to_split)

        m = ones_like(x) * dx * dx * rho_w
        h = ones_like(x) * h0

        rho = ones_like(x) * rho_w
        A = m / rho
        A[idx_inner_pa_to_split] = 3000
        rho0 = ones_like(x) * rho_w

        dw = ones_like(x) * d
        p = ones_like(x) * 0.5 * rho_w * g * d**2

        pa_to_split = zeros_like(x)
        alpha = ones_like(x)
        cs = ones_like(x)
        dt_cfl = ones_like(x)


        u = zeros_like(x)
        u_prev_iter = zeros_like(x)
        v = zeros_like(x)
        v_prev_iter = zeros_like(x)

        au = zeros_like(x)
        av = zeros_like(x)

        pa = gpa(x=x, y=y, m=m, A=A, u=u, v=v, au=au, av=av,
                pa_to_split=pa_to_split, rho=rho, rho0=rho0, h=h, p=p, dw=dw,
                u_prev_iter=u_prev_iter, v_prev_iter=v_prev_iter, cs=cs,
                dt_cfl=dt_cfl, alpha=alpha, name='fluid')

        props = ['m', 'h', 'rho', 'p', 'pa_to_split']
        pa.add_output_arrays(props)
        compute_initial_props([pa])
        return [pa]

    def create_solver(self):
        kernel = CubicSpline(dim=2)
        integrator = EulerIntegrator(fluid=EulerStep())
        dt = 1e-4; tf = 1e-4
        solver = Solver(
            kernel=kernel,
            dim=2,
            integrator=integrator,
            #output_at_times=[0, 1e-4, 2e-4],
            dt=dt,
            tf=tf
            )
        return solver

    def create_equations(self):
        equations = [
                    Group(
                        equations=[
                             SummationDensity(dest='fluid',
                                              sources=['fluid',]),
                             ]
                            ),
                    Group(
                        equations=[
                            UpdateSmoothingLength(h0, dim, dest='fluid'),
                            ], update_nnps=True
                        ),
                    Group(
                        equations=[
                            SWEOS(dest='fluid'),
                            ]
                        ),
                    Group(
                        equations=[
                            CheckForParticlesToSplit(dest='fluid', A_max=2900)
                            ],
                        ),
                    ]
        return equations

    def pre_step(self, solver):
        for pa in self.particles:
            ps = ParticleSplit(pa)
            ps.do_particle_split()
        self.particles[0].A = ones(len(self.particles[0].A)) * 2500
        self.nnps.update()




def compute_initial_props(particles):
    one_time_equations = [
                Group(
                    equations=[
                        SummationDensity(dest='fluid', sources=['fluid',]),
                            ]
                    ),
                Group(
                    equations=[
                        UpdateSmoothingLength(h0, dim, dest='fluid'),
                        ], update_nnps=True
                    ),
                Group(
                    equations=[
                        SWEOS(dest='fluid'),
                        ]
                    ),
            ]
    kernel = CubicSpline(dim=2)
    sph_eval = SPHEvaluator(particles, one_time_equations, dim=2,
                            kernel=kernel)
    sph_eval.evaluate()



if __name__ == '__main__':
    app = ParticleSplitTest()
    app.run()
