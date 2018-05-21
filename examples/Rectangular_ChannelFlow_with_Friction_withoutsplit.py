# Numpy
from numpy import ( ones_like, zeros, zeros_like, mgrid, pi,
                    arange, sqrt, concatenate, sin, cos, where )

# PySPH base
from pysph.base.kernels import CubicSpline
from pysph.base.nnps import DomainManager

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

# PySPH Inlet_Outlet
from pysph.sph.simple_inlet_outlet import SimpleInlet, SimpleOutlet
from pysph.sph.integrator_step import InletOutletStep


# Constants
rho_w = 1000.0
g = 9.81
hdx = 1.2
d = 5.0
dx = 10.
dim = 2
A_max = 0.9 * dx * dx
A_min = 0.4 * dx * dx
h_max = 2 * hdx * dx
y_min = 0.
y_max = 400.
n_inlet = 2
n_outlet = 2
x_max_inlet = 0.
x_min_inlet = -dx * n_inlet
l_tunnel = 800.
x_min_outlet = l_tunnel
x_max_outlet = l_tunnel + n_outlet*0.95*dx


class RectangularOpenChannelFlow(Application):
    def create_particles(self):
        # Inlet Properties
        y = np.arange(dx/2, y_max-(dx/4.), dx)
        x = np.zeros_like(y) - 0.5*dx

        q = 14.645 # Specific Discharge
        u_inlet = q / d

        m = ones_like(x) * dx * dx * rho_w * d
        h = ones_like(x) * hdx * dx
        h0 = ones_like(x) * hdx * dx

        rho = ones_like(x) * rho_w * d
        rho0 = ones_like(x) * rho_w * d
        alpha = dim * rho
        cs = sqrt(9.8 * rho/rho_w)

        A = m / rho

        u = ones_like(x) * u_inlet
        uh = ones_like(x) * u_inlet

        inlet = gpa_swe(x=x, y=y, m=m, rho0=rho0, rho=rho, h0=h0, h=h, u=u,
                        uh=uh, alpha=alpha, cs=cs, name='inlet')
        boundary_props = ['dw_inner_reimann', 'u_inner_reimann',
                          'v_inner_reimann', 'shep_corr']
        inlet.add_output_arrays(boundary_props)

        # Fluid Properties
        xf, yf = np.mgrid[0.5*dx:x_max_inlet+l_tunnel:dx, dx/2:y_max-(dx/4.):dx]
        xf, yf = (np.ravel(t) for t in (xf, yf))
        m = ones_like(xf) * dx * dx * rho_w * d
        h = ones_like(xf) * hdx * dx
        h0 = ones_like(xf) * hdx * dx
        rho = ones_like(xf) * rho_w * d
        rho0 = ones_like(xf) * rho_w * d
        uh = ones_like(xf) * u_inlet
        u = ones_like(xf) * u_inlet
        fluid = gpa_swe(name='fluid', x=xf, y=yf, m=m, rho0=rho0, rho=rho, h=h,
                        h0=h0, uh=uh, u=u)

        # Outlet Properties
        xo, yo = np.mgrid[dx/2.:n_outlet*dx:dx, dx/2:y_max-(dx/4.):dx]
        xo, yo = (np.ravel(t) for t in (xo, yo))
        xo += l_tunnel
        dw = ones_like(xo) * d
        m = ones_like(xo) * dx * dx * rho_w * d
        h = ones_like(xo) * hdx * dx
        h0 = ones_like(xo) * hdx * dx
        rho = ones_like(xo) * rho_w * d
        rho0 = ones_like(xo) * rho_w * d
        cs = sqrt(9.8 * rho/rho_w)
        alpha = dim * rho
        outlet = gpa_swe(name='outlet', x=xo, y=yo, dw=dw, m=m, rho0=rho0,
                         alpha=alpha, rho=rho, h=h, h0=h0, cs=cs)
        outlet.add_output_arrays(boundary_props)

        # Bed Properties
        xb, yb = np.mgrid[-5*dx:l_tunnel+5*dx:dx, 0:y_max+dx/2.:dx]
        xb = np.ravel(xb)
        yb = np.ravel(yb)

        Vb = ones_like(xb) * dx * dx
        nb = ones_like(xb) * 0.0316 # Manning Coefficient
        hb = ones_like(xb) * hdx * dx

        bed = gpa_swe(name='bed', x=xb, y=yb, V=Vb, n=nb, h=hb)

        # Closed Boundary
        xcb_top = np.arange(x_min_inlet, x_max_outlet+dx, dx)
        ycb_top = np.concatenate((ones_like(xcb_top)*(y_max+0.5*dx),
                                  ones_like(xcb_top)*(y_max+1.5*dx)), axis=0)
        xcb_top = np.tile(xcb_top, 2)

        xcb_bottom = np.arange(x_min_inlet, x_max_outlet+dx, dx)
        ycb_bottom = np.concatenate((zeros_like(xcb_bottom)-0.5*dx,
                                     zeros_like(xcb_bottom)-1.5*dx), axis=0)
        xcb_bottom = np.tile(xcb_bottom, 2)

        xcb_all = np.concatenate((xcb_top, xcb_bottom), axis=0)
        ycb_all = np.concatenate((ycb_top, ycb_bottom), axis=0)
        m_cb = ones_like(xcb_all) * dx * dx * rho_w * d
        h_cb = ones_like(xcb_all) * hdx * dx
        rho_cb = ones_like(xcb_all) * rho_w * d
        dw_cb = ones_like(xcb_all) * d
        cs_cb = sqrt(9.8 * dw_cb)
        alpha_cb = dim * rho_cb
        u_cb = ones_like(xcb_all) * u_inlet

        boundary = gpa_swe(name='boundary', x=xcb_all, y=ycb_all, m=m_cb,
                           h=h_cb, rho=rho_cb, dw=dw_cb, cs=cs_cb,
                           alpha=alpha_cb, u=u_cb)

        return [inlet, fluid, outlet, bed, boundary]

    def create_inlet_outlet(self, particle_arrays):
        f_pa = particle_arrays['fluid']
        i_pa = particle_arrays['inlet']
        o_pa = particle_arrays['outlet']
        b_pa = particle_arrays['bed']
        cb_pa = particle_arrays['boundary']

        inlet = SimpleInlet(
            i_pa, f_pa, spacing=dx, n=n_inlet, axis='x', xmin=x_min_inlet,
            xmax=x_max_inlet, ymin=y_min, ymax=y_max
        )
        outlet = SimpleOutlet(
            o_pa, f_pa, xmin=x_min_outlet, xmax=x_max_outlet, ymin=y_min,
            ymax=y_max
        )

        compute_initial_props([i_pa, f_pa, o_pa, b_pa, cb_pa])

        return [inlet, outlet]

    def create_solver(self):
        kernel = CubicSpline(dim=2)
        integrator = SWEIntegrator(inlet=InletOutletStep(), fluid=SWEStep(),
                                   outlet=InletOutletStep())
        tf = 100
        solver = Solver(
            kernel=kernel,
            dim=2,
            integrator=integrator,
            cfl=0.1,
            adaptive_timestep=True,
            tf=tf
            )
        return solver

    def create_equations(self):
        equations = [
            Group(
                equations=[
                    Group(
                        equations=[
                            GatherDensityEvalNextIteration(dest='fluid',
                                sources=['inlet','fluid','outlet', 'boundary']
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
                    )], iterate=True, max_iterations=100
            ),
            Group(
                equations=[
                    CorrectionFactorVariableSmoothingLength(
                        dest='fluid', sources=['fluid', 'inlet', 'outlet',
                            'boundary']),
                        ]
                ),
            Group(
                equations=[
                    SWEOS(dest='fluid'),
                    ]
                ),
            Group(
                equations=[
                    BoundaryInnerReimannStateEval(dest='inlet',
                                                  sources=['fluid']),
                    BoundaryInnerReimannStateEval(dest='outlet',
                                                  sources=['fluid'])
                    ]
                ),
            Group(
                equations=[
                    SubCriticalInFlow(dest='inlet'),
                    SubCriticalOutFlow(dest='outlet')
                    ]
                ),
            Group(
                equations=[
                    BedFrictionSourceEval(
                        dest='fluid', sources=['bed'])
                    ]
                ),
            Group(
                equations=[
                    ParticleAcceleration(dim, dest='fluid', sources=['fluid',
                                    'inlet', 'outlet', 'boundary'], bx=-0.001,
                                    ),
                    ]
                ),
            ]
        return equations

    def pre_step(self, solver):
        for pa in self.particles:
            if pa.name == 'fluid':
                ps = ParticleSplit(pa)
                ps.do_particle_split()
                self.nnps.update()

    def post_step(self, solver):
        for pa in self.particles:
            if pa.name == 'outlet':
                o_pa = pa
        arr_ones = ones_like(o_pa.rho)
        o_pa.alpha = arr_ones * dim * rho_w * d
        o_pa.dw = arr_ones * d
        o_pa.cs = sqrt(9.8 * o_pa.dw)

def compute_initial_props(particles):
    one_time_equations = [
                Group(
                    equations=[
                        SWEOS(dest='fluid'),
                        ]
                    ),
                Group(
                    equations=[
                        BoundaryInnerReimannStateEval(dest='inlet',
                                                      sources=['fluid']),
                        BoundaryInnerReimannStateEval(dest='outlet',
                                                      sources=['fluid'])
                        ]
                    ),
                Group(
                    equations=[
                        SubCriticalInFlow(dest='inlet'),
                        SubCriticalOutFlow(dest='outlet')
                        ]
                    ),
                Group(
                    equations=[
                        CorrectionFactorVariableSmoothingLength(
                            dest='fluid', sources=['fluid', 'inlet', 'outlet',
                                                   'boundary']),
                        ]
                    ),
            ]
    kernel = CubicSpline(dim=2)
    sph_eval = SPHEvaluator(particles, one_time_equations, dim=2,
                            kernel=kernel)
    sph_eval.evaluate()


if __name__ == "__main__":
    app = RectangularOpenChannelFlow()
    app.run()
