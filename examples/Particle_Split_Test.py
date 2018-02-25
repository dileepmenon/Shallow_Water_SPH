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
h0 = hdx * dx
dim = 2


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
        rho0 = ones_like(x) * rho_w
        rho_prev_iter = zeros_like(x)
        rho_residual= zeros_like(x)
        positive_rho_residual = zeros_like(x)
        summation_rho = zeros_like(x)

        A = m / rho
        A[idx_inner_pa_to_split] = 3000

        dw = ones_like(x) * d
        p = ones_like(x) * 0.5 * rho_w * g * d**2

        pa_to_split = zeros_like(x)
        alpha = zeros_like(x)
        psi = zeros_like(x)
        l2_err_rho_residual = zeros_like(x)
        cs = ones_like(x)
        dt_cfl = ones_like(x)

        tv = zeros_like(x)
        tu = zeros_like(x)

        u = zeros_like(x)
        u_prev_iter = zeros_like(x)
        v = zeros_like(x)
        v_prev_iter = zeros_like(x)

        au = zeros_like(x)
        av = zeros_like(x)

        consts = {'tmp_comp': [0.0, 0.0]}

        pa = gpa(x=x, y=y, m=m, rho0=rho0, rho=rho, alpha=alpha,
                 rho_prev_iter=rho_prev_iter, psi=psi,
                 h=h, u=u, v=v, u_prev_iter=u_prev_iter,
                 v_prev_iter=v_prev_iter, au=au, av=av, A=A, cs=cs,
                 dt_cfl=dt_cfl, tv=tv, tu=tu, p=p, dw=dw,
                 pa_to_split=pa_to_split, name='fluid', constants=consts)

        props = ['m', 'h', 'rho', 'p']
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
                            ScatterDensityEvalNextIteration(dest='fluid', sources=['fluid',]),
                            NonDimensionalDensityResidual(dest='fluid')
                            ]),
                    Group(
                        equations=[
                            UpdateSmoothingLength(h0, dim, dest='fluid')
                            ], update_nnps=True
                        ),
                    Group(
                        equations=[
                            CheckConvergenceDensityResidual(dest='fluid')
                            ],
                    )], iterate=True, max_iterations=100
            ),
            Group(
                equations=[
                    CorrectionFactorVariableSmoothingLength(dest='fluid',
                                                            sources=['fluid',]),
            ]),
            Group(
                equations=[
                    SWEOS(dest='fluid'),
                    ParticleAccelerations(dim, dest='fluid', sources=['fluid',],
                                         ),
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

    def post_process(self):
        rho_exact = 1e4
        rho_num = self.particles[0].rho
        # Filter values of rho, ignoring values of boundary particles
        rho_num = rho_num[rho_num >= 10000]
        print('\nMax rho is %0.3f '%max(rho_num))
        l2_err_rho = sqrt(np.sum((rho_exact - rho_num)**2)
                          / len(rho_num))
        print('L2 error in density is %0.3f \n'%l2_err_rho)


def compute_initial_props(particles):
    one_time_equations = [
                Group(
                    equations=[
                        InitialTimeSummationDensity(dest='fluid', sources=['fluid',]),
                        SWEOS(dest='fluid'),
                        CheckForParticlesToSplit(dest='fluid', A_max=2900)
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
