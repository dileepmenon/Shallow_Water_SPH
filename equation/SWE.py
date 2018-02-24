from pysph.sph.equation import Equation
from pysph.sph.integrator_step import IntegratorStep
from pysph.sph.integrator import Integrator
from pysph.base.utils import get_particle_array
from numpy import sqrt, cos, sin, ones, zeros, pi
import numpy as np


def get_particle_array_swe(constants=None, **props):
    """Return a particle array for the Shallow Water formulation.

    This sets the default properties to be::

        ['x', 'y', 'z', 'u', 'v', 'w', 'h', 'rho', 'm', 'p',
         'A', 'cs', 'rho0', 'rho_prev_iter', 'rho_residual',
         'positive_rho_residual', 'summation_rho', 'd', 'alpha',
         'exp_lambda', 'tv', 'tu', 'u_prev_iter', 'v_prev_iter',
         'uh', 'vh', 'dt_cfl', 'pa_to_split']

    Parameters
    ----------
    constants : dict
        Dictionary of constants

    Other Parameters
    ----------------
    props : dict
        Additional keywords passed are set as the property arrays.

    See Also
    --------
    get_particle_array

    """

    swe_props = ['A', 'cs', 'rho0', 'rho_prev_iter', 'rho_residual',
                 'positive_rho_residual', 'summation_rho', 'd', 'alpha',
                 'exp_lambda', 'tv', 'tu', 'u_prev_iter', 'v_prev_iter',
                 'uh', 'vh', 'dt_cfl', 'pa_to_split']

    pa = get_particle_array(
        constants=constants, additional_props=swe_props, **props
    )

    # default property arrays to save out.
    pa.set_output_arrays([
        'x', 'y', 'u', 'v', 'rho', 'm', 'h', 'A', 'cs',
        'd', 'p', 'au', 'av', 'pid', 'gid', 'tag'
    ])

    return pa


class CheckForParticlesToSplit(Equation):
    def __init__(self, dest, A_max=1e9):
        self.A_max = A_max
        super(CheckForParticlesToSplit, self).__init__(dest, None)

    def initialize(self, d_pa_to_split, d_idx):
        d_pa_to_split[d_idx] = 0.0

    def post_loop(self, d_idx, d_rho0, d_A, d_pa_to_split):
        if d_A[d_idx] >= self.A_max:
            d_pa_to_split[d_idx] = 1


class ParticleSplit(object):
    def __init__(self, pa_arr):
        self.pa_arr = pa_arr
        self.center_pa_mass_frac = 0.1787
        self.edge_pa_mass_frac = 0.1369
        self.pa_h_ratio = 0.9
        self.center_and_edge_pa_separation_frac = 0.4
        self.props = self.pa_arr.properties.keys() # Change
        self.idx_pa_to_split = self._get_idx_of_particles_to_split()
        self.num_edge_pa_after_single_split = 6

    def do_particle_split(self, solver=None):
        if not self.idx_pa_to_split.size:
            return
        else:
            # Parent particle properties
            h_parent = self.pa_arr.h[self.idx_pa_to_split]
            m_parent = self.pa_arr.m[self.idx_pa_to_split]
            x_parent = self.pa_arr.x[self.idx_pa_to_split]
            y_parent = self.pa_arr.y[self.idx_pa_to_split]
            rho_parent = self.pa_arr.rho[self.idx_pa_to_split]
            rho0_parent = self.pa_arr.rho0[self.idx_pa_to_split]

            # Edge daughter particle properties update
            n = self.num_edge_pa_after_single_split
            h_edge_pa  = self.pa_h_ratio * np.repeat(h_parent, n)
            m_edge_pa  = self.edge_pa_mass_frac * np.repeat(m_parent, n)
            edge_pa_pos = self._get_edge_pa_positions(h_parent)
            x_edge_pa  = edge_pa_pos[0] + np.repeat(x_parent, n)
            y_edge_pa  = edge_pa_pos[1] + np.repeat(y_parent, n)

            total_num_edge_pa = x_edge_pa.size
            #rho0_edge_pa = np.max(self.pa_arr.rho0) * ones(total_num_edge_pa)
            rho0_edge_pa = np.repeat(rho0_parent, n)
            rho_edge_pa = np.repeat(rho_parent, n)

            # Center daughter particle properties update
            for idx in self.idx_pa_to_split:
                self.pa_arr.m[idx] *= self.center_pa_mass_frac
                self.pa_arr.h[idx] *= self.pa_h_ratio

            self._add_edge_pa_prop(h_edge_pa, m_edge_pa, x_edge_pa, y_edge_pa,
                                   rho0_edge_pa, rho_edge_pa)

    def _get_idx_of_particles_to_split(self):
        idx_pa_to_split = []
        for idx, val in enumerate(self.pa_arr.pa_to_split):
            if val:
                idx_pa_to_split.append(idx)
        return np.array(idx_pa_to_split)

    def _get_edge_pa_positions(self, h_parent):
        num_of_pa_to_split = len(self.idx_pa_to_split)
        n = self.num_edge_pa_after_single_split
        x = zeros(n)
        y = zeros(n)
        r = self.center_and_edge_pa_separation_frac
        for i, theta in enumerate(range(0, 360, 60)):
            x[i] = r * cos((pi/180)*theta)
            y[i] = r * sin((pi/180)*theta)
        x = np.tile(x, num_of_pa_to_split) * np.repeat(h_parent, n)
        y = np.tile(y, num_of_pa_to_split) * np.repeat(h_parent, n)
        return x.copy(), y.copy()

    def _add_edge_pa_prop(self, h_edge_pa, m_edge_pa, x_edge_pa, y_edge_pa,
                          rho0_edge_pa, rho_edge_pa):
        add_prop = {}
        for prop in self.props:
            if prop == 'm':
                add_prop[prop] = m_edge_pa
            elif prop == 'h':
                add_prop[prop] = h_edge_pa
            elif prop == 'x':
                add_prop[prop] = x_edge_pa
            elif prop == 'y':
                add_prop[prop] = y_edge_pa
            elif prop == 'rho0':
                add_prop[prop] = rho0_edge_pa
            elif prop == 'rho':
                add_prop[prop] = rho_edge_pa
            else:
                pass
        self.pa_arr.add_particles(**add_prop)


class EulerStep(IntegratorStep):
    """Fast but inaccurate integrator. Use this for testing"""
    def initialize(self, d_u, d_v, d_u_prev_iter, d_v_prev_iter, d_idx):
        d_u_prev_iter[d_idx] = d_u[d_idx]
        d_v_prev_iter[d_idx] = d_v[d_idx]

    def stage1(self, d_idx, d_u, d_v, d_au, d_av, d_x, d_y, dt):
        d_u[d_idx] += dt * d_au[d_idx]
        d_v[d_idx] += dt * d_av[d_idx]
        d_x[d_idx] += dt * d_u[d_idx]
        d_y[d_idx] += dt * d_v[d_idx]


class SWEStep(IntegratorStep):
    """Predictor corrector Integrator for Shallow Water problems"""
    def initialize(self, d_u, d_v, d_u_prev_iter, d_v_prev_iter, d_idx):
        d_u_prev_iter[d_idx] = d_u[d_idx]
        d_v_prev_iter[d_idx] = d_v[d_idx]

    def stage1(self, d_uh, d_vh, d_idx, d_au, d_av, dt):
        d_uh[d_idx] +=  dt * d_au[d_idx]
        d_vh[d_idx] +=  dt * d_av[d_idx]

    def stage2(self, d_u, d_v, d_uh, d_vh, d_idx, d_au, d_av, d_x, d_y, dt):
        d_x[d_idx] += dt * d_uh[d_idx]
        d_y[d_idx] += dt * d_vh[d_idx]
        d_u[d_idx] = d_uh[d_idx] + dt/2.*d_au[d_idx]
        d_v[d_idx] = d_vh[d_idx] + dt/2.*d_av[d_idx]


class SWEIntegrator(Integrator):
    def one_timestep(self, t, dt):
        self.compute_accelerations()

        self.initialize()

        # Predict
        self.stage1()

        # Call any post-stage functions.
        self.do_post_stage(0.5*dt, 1)


        # Correct
        self.stage2()

        # Call any post-stage functions.
        self.do_post_stage(dt, 2)


class DensityEvalNextIteration(Equation):
    def initialize(self, d_rho, d_idx, d_rho_prev_iter):
        d_rho_prev_iter[d_idx] = d_rho[d_idx]
        d_rho[d_idx] = 0.0

    def loop(self, d_rho, d_idx, s_m, s_idx, WJ):
        d_rho[d_idx] += s_m[s_idx] * WJ


class NonDimensionalDensityResidual(Equation):
    def __init__(self, dest, sources=None):
        super(NonDimensionalDensityResidual, self).__init__(dest, sources)

    def post_loop(self, d_psi, d_rho, d_rho_prev_iter, d_idx):
        d_psi[d_idx] = abs(d_rho[d_idx]-d_rho_prev_iter[d_idx]) / d_rho[d_idx]


class CheckConvergenceDensityResidual(Equation):
    def __init__(self, dest, sources=None):
        super(CheckConvergenceDensityResidual, self).__init__(dest, sources)
        self.eqn_has_converged = 0

    def initialize(self):
        self.eqn_has_converged = 0

    def reduce(self, dst):
        dst.tmp_comp[0] = serial_reduce_array(dst.psi > 0.0, 'sum')
        dst.tmp_comp[1] = serial_reduce_array(dst.psi**2, 'sum')
        dst.tmp_comp.set_data(parallel_reduce_array(dst.tmp_comp, 'sum'))
        epsilon = sqrt(dst.tmp_comp[1] / dst.tmp_comp[0])
        print(epsilon)
        if epsilon <= 1e-3:
            print('Converged')
            self.eqn_has_converged = 1

    def converged(self):
        return self.eqn_has_converged


class InitialTimeSummationDensity(Equation):
    def initialize(self, d_rho0, d_idx):
        d_rho0[d_idx] = 0.0

    def loop(self, d_rho0, d_idx, s_m, s_idx, WI):
        d_rho0[d_idx] += s_m[s_idx] * WI

    def post_loop(self, d_rho0, d_rho, d_idx):
        d_rho[d_idx] = d_rho0[d_idx]


class CorrectionFactorVariableSmoothingLength(Equation):
    def initialize(self, d_idx, d_alpha):
         d_alpha[d_idx] = 0.0

    def loop(self, d_alpha, d_idx, DWI, XIJ, s_idx, s_m):
        d_alpha[d_idx] += -s_m[s_idx] * (DWI[0]*XIJ[0] + DWI[1]*XIJ[1])


class SummationDensity(Equation):
    def initialize(self, d_summation_rho, d_idx):
        d_summation_rho[d_idx] = 0.0

    def loop(self, d_summation_rho, d_idx, s_m, s_idx, WI):
        d_summation_rho[d_idx] += s_m[s_idx] * WI



class InitialGuessDensity(Equation):
    def __init__(self, dim, dest, sources):
        super(InitialGuessDensity, self).__init__(dest, sources)
        self.dim = dim

    def initialize(self, d_exp_lambda, d_idx):
        d_exp_lambda[d_idx] = 0.0

    def loop(self, d_exp_lambda, d_u_prev_iter, d_v_prev_iter, d_alpha, d_idx,
             s_m, s_u_prev_iter, s_v_prev_iter, s_idx, DWI, dt, t):
        a1 = (s_u_prev_iter[s_idx]-d_u_prev_iter[d_idx]) * DWI[0]
        a2 = (s_v_prev_iter[s_idx]-d_v_prev_iter[d_idx]) * DWI[1]
        const = (self.dim*dt) / d_alpha[d_idx]
        d_exp_lambda[d_idx] +=  const * (s_m[s_idx]*(a1+a2))

    def post_loop(self, d_rho, d_exp_lambda, d_idx):
        d_rho[d_idx] = d_rho[d_idx] * e**(d_exp_lambda[d_idx])


class UpdateSmoothingLength(Equation):
    def __init__(self, h0, dim, dest, sources=None):
        self.h0 = h0
        self.dim = dim
        super(UpdateSmoothingLength, self).__init__(dest, sources)

    def post_loop(self, d_h, d_rho0, d_rho, d_idx):
        d_h[d_idx] = self.h0 * (d_rho0[d_idx]/d_rho[d_idx])**(1./self.dim)


class DensityResidual(Equation):
    def __init__(self, dest, sources=None):
        super(DensityResidual, self).__init__(dest, sources)

    def post_loop(self, d_rho, d_idx, d_rho_residual, d_summation_rho, t):
        d_rho_residual[d_idx] = d_rho[d_idx] - d_summation_rho[d_idx]


class DensityNewtonRaphsonIteration(Equation):
    def __init__(self, dim, dest, sources=None):
        self.dim = dim
        super(DensityNewtonRaphsonIteration, self).__init__(dest, sources)

    def initialize(self, d_rho, d_rho_prev_iter, d_idx):
        d_rho_prev_iter[d_idx] = d_rho[d_idx]

    def post_loop(self, d_rho, d_idx, d_alpha, d_rho_residual):
        a1 = d_rho_residual[d_idx] * self.dim
        a2 = a1 + d_alpha[d_idx]
        const = 1 - (a1/a2)
        d_rho[d_idx] = d_rho[d_idx] * const


class CheckConvergence(Equation):
    def __init__(self, dest, sources=None):
        super(CheckConvergence, self).__init__(dest, sources)
        self.eqn_has_converged = 0

    def initialize(self):
        self.eqn_has_converged = 0

    def post_loop(self, d_positive_rho_residual, d_rho_residual,
                  d_rho_prev_iter,  d_idx, t):
        d_positive_rho_residual[d_idx] = abs(d_rho_residual[d_idx])

    def reduce(self, dst):
        max_epsilon = max(dst.positive_rho_residual / dst.rho_prev_iter)
        if max_epsilon <= 1e-15:
            self.eqn_has_converged = 1

    def converged(self):
        return self.eqn_has_converged


class SWEOS(Equation):
    def __init__(self, dest, sources=None, g=9.81, rhow=1000.0):
        self.rhow = rhow
        self.g = g
        self.fac = 0.5 * (g/rhow)
        super(SWEOS, self).__init__(dest, sources)

    def post_loop(self, d_rho, d_cs, d_u, d_v, d_idx, d_p, d_dw, d_dt_cfl,
                 d_m, d_A, d_alpha):
        d_p[d_idx] = self.fac * (d_rho[d_idx])**2
        d_cs[d_idx] = sqrt(self.g * d_rho[d_idx]/self.rhow)
        #d_A[d_idx] = d_m[d_idx] / d_rho[d_idx]
        d_dw[d_idx] = d_rho[d_idx] / self.rhow
        d_dt_cfl[d_idx] = d_cs[d_idx] + (d_u[d_idx]**2 + d_v[d_idx]**2)**0.5


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

    def loop(self, d_x, d_y, d_rho , d_idx , s_m , s_idx , s_rho , d_m ,
             DWI ,DWJ , d_au, d_av , s_alpha , d_alpha , s_p , d_p , d_tu ,
             s_tu, d_tv , s_tv):
        tmp1 = (s_rho[s_idx]*self.dim) / s_alpha[s_idx]
        tmp2 = (d_rho[d_idx]*self.dim) / d_alpha[d_idx]
        d_tu[d_idx] += s_m[s_idx] * self.ct * (tmp1*DWJ[0] + tmp2*DWI[0])
        d_tv[d_idx] += s_m[s_idx] * self.ct * (tmp1*DWJ[1] + tmp2*DWI[1])

    def post_loop(self, t, dt, d_x ,d_y,d_idx ,d_u ,d_v ,d_tu ,d_tv ,d_au, d_av):
        dhxi =  self.dhxi ; dhyi = self.dhyi
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
