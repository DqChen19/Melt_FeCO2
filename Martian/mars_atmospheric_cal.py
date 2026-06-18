import numpy as np
from math import floor

MARS_RAD = 3.39E6   # [m]

_alt     = np.load('alt.npy')
_rho_a   = np.load('rho_a.npy')
_rho_co2 = np.load('rho_co2.npy')
_rho_o   = np.load('rho_o.npy')
_rho_o2  = np.load('rho_o2.npy')

def atmospheric_density_and_oxygen(altitude):
    current_alt = (altitude - MARS_RAD) / 1000  # m → km

    # 超出范围 clamp 到边界，不崩溃也不返回数组
    current_alt = float(np.clip(current_alt, _alt[0], _alt[-1] - 1e-9))

    idx = int(floor(current_alt))          # ← 修复：current_alt，不是 alt
    frac_low = 1 - (current_alt - _alt[idx]) / (_alt[idx+1] - _alt[idx])

    rho_a   = _rho_a[idx]   * frac_low + _rho_a[idx+1]   * (1 - frac_low)
    rho_co2 = _rho_co2[idx] * frac_low + _rho_co2[idx+1] * (1 - frac_low)
    rho_o   = _rho_o[idx]   * frac_low + _rho_o[idx+1]   * (1 - frac_low)
    rho_o2  = _rho_o2[idx]  * frac_low + _rho_o2[idx+1]  * (1 - frac_low)

    return rho_a, rho_co2, rho_o, rho_o2