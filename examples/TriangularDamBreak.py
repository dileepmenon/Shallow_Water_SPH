# Numpy
from numpy import ( linspace, ones, ones_like, zeros, zeros_like, mgrid, pi,
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


# Constants
rho_w = 1000.0
g = 9.81
hdx = 2.3
d = 1.0
dx = 0.02
h0 = hdx * dx
dim = 2


class ParticleAccelerations(Equation):
    def __init__(self, dim, dest, sources, dhxi=0, dhyi=0, dhxxi=0,
                 dhxyi=0, dhyyi=0, u_only=False, v_only=False):
        super(ParticleAccelerations, self).__init__(dest, sources)
        self.g = 9.81
        self.rhow = 1000.0
        self.ct = self.g/(2*self.rhow)
        self.dim = dim
        self.dhxi = dhxi
        self.dhyi = dhyi
        self.dhxxi = dhxxi
        self.dhxyi = dhxyi
        self.dhyyi = dhyyi
        self.u_only = u_only
        self.v_only = v_only

    def initialize(self, d_idx, d_tu, d_tv):
        d_tu[d_idx] = 0.0
        d_tv[d_idx] = 0.0

    def loop(self, d_x, d_y, t, dt, d_rho , d_idx , s_m , s_idx , s_rho , d_m ,
             DWI ,DWJ , d_au, d_av , s_alpha , d_alpha , s_p , d_p , d_tu ,
             s_tu, d_tv , s_tv):
        tmp1 = (s_rho[s_idx]*self.dim) / s_alpha[s_idx]
        tmp2 = (d_rho[d_idx]*self.dim) / d_alpha[d_idx]
        d_tu[d_idx] += s_m[s_idx] * self.ct * (tmp1*DWJ[0] + tmp2*DWI[0])
        d_tv[d_idx] += s_m[s_idx] * self.ct * (tmp1*DWJ[1] + tmp2*DWI[1])

    def post_loop(self, t, dt, d_x ,d_y,d_idx ,d_u ,d_v ,d_tu ,d_tv ,d_au, d_av):
        dhxi =  self.dhxi
        if d_y[d_idx] > 0:
            k = 1
        elif d_y[d_idx] < 0:
            k = -1
        else:
            k = 0
        dhyi = self.dhxi * k
        dhxxi = self.dhxxi; dhxyi = self.dhxyi
        dhyyi = self.dhyyi;
        vikivi = d_u[d_idx]*d_u[d_idx]*dhxxi + 2*d_u[d_idx]*d_v[d_idx]*dhxyi + \
                 d_v[d_idx]*d_v[d_idx]*dhyyi
        tidotdhi = d_tu[d_idx]*dhxi + d_tv[d_idx]*dhyi
        dhidotdhi = dhxi**2 + dhyi**2
        temp3 = self.g + vikivi - tidotdhi
        temp4 = 1 +  dhidotdhi
        if not self.v_only:
            d_au[d_idx] = -(temp3/temp4)*dhxi - d_tu[d_idx]
        if not self.u_only:
            d_av[d_idx] = -(temp3/temp4)*dhyi - d_tv[d_idx]


class TriangularDamBreak(Application):
    def create_particles(self):
        """Create the triangular patch of fluid."""
        r1 = [2, 2]
        r2 = [2, -2]
        x = zeros(0)
        y = zeros(0)
        while r1[0] > 0:
            r = linspace(r1[-1], r2[-1], (r1[-1]-r2[-1])*25 + 3)
            x = concatenate((x, (ones(len(r))*r1[0])), axis=0)
            y = concatenate((y, r))
            r1[0] -= dx
            r2[0] -= dx
            r1[1] = r1[0]
            r2[1] = -r1[0]

        x = concatenate((zeros(1), x))
        y = concatenate((zeros(1), y))

        m = ones_like(x) * dx * dx * rho_w
        h = ones_like(x) * h0

        rho = ones_like(x) * rho_w
        rho0 = ones_like(x) * rho_w
        rho_prev_iter = ones_like(x)
        rho_residual= zeros_like(x)
        positive_rho_residual = zeros_like(x)
        summation_rho = zeros_like(x)

        dw = ones_like(x) * d
        cs = ones_like(x) * sqrt(9.8*d)
        p = ones_like(x) * 0.5 * rho_w * g * d**2
        A = m / rho
        alpha = ones_like(x)
        exp_lambda = zeros_like(x)

        tv = zeros_like(x)
        tu = zeros_like(x)

        u = zeros_like(x)
        u_prev_iter = zeros_like(x)
        v = zeros_like(x)
        v_prev_iter = zeros_like(x)

        dt_cfl = zeros_like(x)

        au = zeros_like(x)
        av = zeros_like(x)

        pa = gpa(x=x, y=y, m=m, rho0=rho0, rho=rho, exp_lambda=exp_lambda,
                rho_prev_iter=rho_prev_iter, rho_residual=rho_residual,
                positive_rho_residual=positive_rho_residual,
                summation_rho=summation_rho, alpha=alpha, h=h, u=u, v=v,
                u_prev_iter=u_prev_iter, v_prev_iter=v_prev_iter, au=au, av=av,
                A=A, cs=cs, dt_cfl=dt_cfl, tv=tv, tu=tu, p=p, dw=dw,
                name='fluid')

        props = ['u', 'v', 'm', 'tu', 'tv', 'dw', 'au', 'av', 'h', 'rho', 'p']
        pa.add_output_arrays(props)
        print ( "Triangular Dam break :: %d particles"
                %(pa.get_number_of_particles()) )

        return [pa]

    def create_equations(self):
        equations = [
            Group(
                equations=[
                    InitialGuessDensity(dim, dest='fluid', sources=['fluid',],),
                    UpdateSmoothingLength(h0, dim, dest='fluid')
                    ], update_nnps=True
                ),
            Group(
                equations=[
                    Group(
                        equations=[
                            CorrectionFactorVariableSmoothingLength(dest='fluid',
                                                                    sources=['fluid',]),
                            SummationDensity(dest='fluid', sources=['fluid',]),
                            DensityResidual('fluid'),
                            DensityNewtonRaphsonIteration(dim, dest='fluid')
                            ]),
                    Group(
                        equations=[
                            UpdateSmoothingLength(h0, dim, dest='fluid')
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
                    CorrectionFactorVariableSmoothingLength(dest='fluid',
                                                            sources=['fluid',]),
            ]),
            Group(
                equations=[
                    SWEOS(dest='fluid'),
                    ParticleAcceleration(dim, dhxi=0.4, dest='fluid',
                                         sources=['fluid',]),
                    ],
                ),
            ]
        return equations


if __name__ == '__main__':
    app = TriangularDamBreak()
    one_time_equations = [
        Group(
            equations=[
                CorrectionFactorVariableSmoothingLength(dest='fluid',
                                                        sources=['fluid',]),
                InitialTimeSummationDensity(dest='fluid', sources=['fluid',]),
                SWEOS(dest='fluid'),
                ], update_nnps=False
            )
    ]

    from pysph.sph.acceleration_eval import AccelerationEval
    from pysph.sph.sph_compiler import SPHCompiler
    kernel = CubicSpline(dim=2)
    integrator = EulerIntegrator(fluid=EulerStep())
    dt = 1e-4; tf = 2
    solver = Solver(
        kernel=kernel,
        dim=2,
        integrator=integrator,
        dt=dt,
        tf=tf
        )
    part_arr = app.create_particles
    eqns = app.create_equations()
    app.setup(solver=solver, equations=eqns,
          particle_factory=part_arr)
    a_eval = AccelerationEval(
        solver.particles, equations=one_time_equations, kernel=CubicSpline(dim=2))
    compiler = SPHCompiler(a_eval, None)
    compiler.compile()
    a_eval.set_nnps(app.nnps)
    a_eval.compute(0.0, dt)
    app.run()
