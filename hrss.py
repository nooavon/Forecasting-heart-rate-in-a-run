"""State-space model: HR relaxes toward a power-driven target, with an optional random-walk offset (cardiac drift).

 state  x_t   = latent (denoised) heart rate
        L_t   = offset of the HR target (random walk, 'drift'); constant if drift is switched off
 x_{t+1} = (1-a) x_t + a (L_t + g u_{t-d}) + eta_t ,   L_{t+1} = L_t + zeta_t ,   y_t = x_t + eps_t
 drift: False = constant offset, True = random-walk offset with estimated noise, float = random walk with that fixed noise sd.
 a = 1/tau is the relaxation rate, g the steady-state gain in bpm per unit of input u (watts).
"""
import numpy as np
from statsmodels.tsa.statespace.mlemodel import MLEModel

class HRStateSpace(MLEModel):
    param_names = ["a", "g", "sd_eta", "sd_zeta", "sd_eps"]
    def __init__(self, y, u, drift=True, delay=0):
        y = np.asarray(y, float); u = np.asarray(u, float)
        super().__init__(y, k_states=2, k_posdef=2, initialization="approximate_diffuse")
        self.drift = drift; self.delay = delay
        us = np.r_[np.full(delay, u[0]), u][:len(u)]            # input delayed by `delay` seconds (causal)
        self._u = us
        self.ssm["design", 0, 0] = 1.0
        self.ssm["selection"] = np.eye(2)
        self.ssm["transition", 1, 1] = 1.0
        self.ssm["state_intercept"] = np.zeros((2, self.nobs))
        self.initialize_approximate_diffuse(1e4)
    @property
    def start_params(self):
        return np.array([1 / 60, 0.15, 0.3, 0.05 if self.drift is True else 1e-3, 0.5])
    def transform_params(self, p):
        a = 1 / (1 + np.exp(-p[0])); return np.array([a, p[1], np.exp(p[2]), np.exp(p[3]), np.exp(p[4])])
    def untransform_params(self, p):
        return np.array([np.log(p[0] / (1 - p[0])), p[1], np.log(p[2]), np.log(p[3]), np.log(p[4])])
    def update(self, params, **kw):
        params = super().update(params, **kw)
        a, g, se, sz, sv = params
        if self.drift is False: sz = 1e-6
        elif self.drift is not True: sz = float(self.drift)      # fixed drift noise level (bpm per sqrt(s))
        self.ssm["transition", 0, 0] = 1 - a; self.ssm["transition", 0, 1] = a
        self.ssm["state_intercept", 0, :] = a * g * self._u
        self.ssm["state_cov"] = np.diag([se ** 2, sz ** 2])
        self.ssm["obs_cov", 0, 0] = sv ** 2
