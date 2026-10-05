"""
FORK(HONDA_ACCORD_9G_AU): brake law v2 for HONDA_ELESYS, the "measured brake law" (HondaElesysBrakeLawV2, default
OFF, CP_SP flag HondaFlagsSP.ELESYS_BRAKE_LAW_V2 = 32). With the flag clear nothing in this module runs and the car
brakes exactly as before.

TODAY'S LAW (carcontroller.py): brake = -net/2.6 (compute_gb_honda_elesys), minus an aero credit wb(v) =
interp(v, [0, 2.3, 35], [0.001, 0.002, 0.15]), times the learned scalar gain, as BRAKE_COMMAND counts. So it assumes
the car coasts at -2.6*wb (-0.10 m/s^2 at 10 m/s, -0.21 at 20) and that every count brakes 2.6/256 m/s^2 from the
first one. Measured, neither is true: the car coasts at -0.3..-0.5 m/s^2 above 5 m/s, and the brake has a dead zone
of roughly 20-40 counts before it delivers anything. 33% of openpilot's brake time was spent asking for decel that
coasting alone gives, at a median 6-9 counts - inside the dead zone, doing nothing but running the pump.

V2 (learnaudit A_synth section 3 B2; fit batch3/fit/PARAMS.json, reference implementation batch3/sim/laws.py):

  net    = accel + hill - creep(accel, v)                  the same net compute_gb_honda_elesys() makes
  zb(v)  = coast(v) - creep(v) - D_EXTRA                   brake-on point: below this coasting is not enough
  zp(v)  = coast(v) - creep(v) + DELTA(v)                  pedal-zero point (the gas law's window, elesys_gas.py)
  brake  = max(zb - net, 0) / 2.6                          into the UNCHANGED actuator_hysteresis and rate limit
  counts = h^-1(brake * 256 / k(v); c0(v), TOE)            no aero credit (coast replaces it), brake gain fixed 1.0

h is a soft hinge: effective brake = 0 below c0 - TOE, (cb - c0 + TOE)^2 / (4 TOE) in the toe, cb - c0 above it.
Its inverse jumps to c0 - TOE (8-32 counts) when the brake comes on - "the low end of the dead zone": c0 sits at or
below the low end of each band's bootstrap interval, because braking a little less at brake-on beats jumping.
Between zp and zb is the COAST BAND: no pedal and no brake.

Below V_LO, outside the PID state (stopping, starting), not longActive, or on a non-finite input, TODAY'S CODE PATH
RUNS, byte for byte: the soft stop, the standstill hold (~174-189 counts) and the creep table are not touched.
From V_LO to V_HI every quantity is a linear blend from today's equivalent (zb = -2.6 wb, zp = -1.95 wb, c0 = 0,
k = 1, TOE = 0, the gas law's offset) to v2's, so nothing steps at 4 m/s except the aero credit (< 1.5 counts).
The one thing the flag changes on those frames too: the scalar brake gain (dynamic_tuning.py) is held at 1.0 for the
whole drive and learns nothing (its stored value is kept). At a standstill it was already faded to 1.0.
It needs gas law v2 (brake_law_v2_enabled()); with HondaElesysGasLawV2 off it stays off for the drive.

TWO MEASURED CHANGES BEYOND THE B2 SKETCH (without them v2 is worse than today on pump starts, +24% in the sim):
  * DELTA. Any positive interceptor command lifts the engine computer's pedal (0x17C) from 0 to ~6.6 counts and
    ends coasting, adding 0.17-0.37 m/s^2 over coasting. A pedal-zero point at coast(v) leaves the PID cycling
    across the brake-on point; at coast + DELTA the pedal reaches 0 exactly where the car starts coasting.
  * G0, the pedal at net 0, re-measured on the current pedal calibration. 0x17C = -3.2 + 254.7 * cmd on routes up to
    000000da, = 6.6 + 245.1 * cmd from 000000dd (same commit 715ea5df6, so the pedal/interceptor side changed);
    ~0.036 below ELESYS_FF_G0. The window slope G0/|zp| is 0.20-0.56 per m/s^2, against 0.46-0.85 today.

OFFLINE ACCEPTANCE (steady braking, response = IMU force minus coast(v); fit routes <= 0000010f, held out 110-115):
held-out RMS 0.158 m/s^2 (today 0.284; criterion <= 0.17), bias -0.001; fit 0.171; 5-fold by route 0.172. Every
held-out band with 3+ routes within +-0.015, but 20-25 m/s is -0.076 on 9.6 s from 2 routes (route biases -0.03 and
-0.11; the fit set's per-route spread there is 0.08): that band FAILS the +-0.05 rule, and there is no held-out data
above 25 m/s. Firm braking at 20 m/s and above is extrapolated (steady data there: median 19 counts, p90 54).

CLOSED LOOP (batch3/sim: the real LongControl and controller helpers, a plant fitted separately; 87 engaged min on
7 routes, pump C1 on both): brake applications -33%, light ones (peak < 12 counts) 87 -> 0, moving brake time -37%,
braking at targets coasting reaches -57%, pump starts -7%, pump time -13%, tracking equal or better, stops unchanged
in paired comparison - under all 10 plant perturbations. THE COST is the onset: in the first second of an
application the car decelerates 0.18 m/s^2 less than asked (today 0.09) and the integrator over-corrects for ~2 s
(-0.08 at 5-10 m/s, -0.06 at 15-20 while braking; B4 wants +-0.05). Levers if the road shows it: a smaller TOE or a
negative D_EXTRA pre-fills earlier (each ~+10% applications at -0.05); the c0 table is not one (<= 0.015).

Known limits: coast(v) is D with ECON off and is used in every mode; the coast curve at 3-6 m/s varies by route
(the 4-6 m/s blend exists for that); the plant's absolute onset figures rest on brake dynamics the logs barely
identify. Route check: sunnypilot/tools/brake_route_check.py (B4 metrics). CAR doc section 7.10.

Runs inside CarController.update(), which must never raise (no 0x1FA -> BRAKE_ERROR ~1 s later): every entry point
answers None/0.0 on anything non-finite, and None sends the frame down today's path.
"""

import math

import numpy as np

from opendbc.car import structs
from opendbc.car.honda.values import HONDA_ELESYS
from opendbc.sunnypilot.car.honda.elesys_gas import elesys_ff_offset
from opendbc.sunnypilot.car.honda.values_ext import HondaFlagsSP

LongCtrlState = structs.CarControl.Actuators.LongControlState

# --- the fitted law (batch3/fit/PARAMS.json "brake_law_v2"; do not edit without refitting) ----------------------------
V_LO = 4.0        # m/s; at or below this (and outside PID) today's law runs unchanged
V_HI = 6.0        # m/s; V_LO..V_HI: linear blend from today's equivalents to v2

# coast(v): median grade-free accel with pedal 0 and brake 0, D, ECON off, fit routes only (2,094 s)
COAST_BP = [0.0, 0.5, 1.5, 2.5, 3.5, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]
COAST_V = [0.606, 0.606, 0.241, -0.096, -0.191, -0.309, -0.476, -0.509, -0.403, -0.41, -0.429, -0.482, -0.489]

# soft dead zone center c0 (BRAKE_COMMAND counts) and slope k (units of today's 2.6/256 m/s^2 per count) per band
C0_BP = [3.0, 7.5, 12.5, 17.5, 22.5, 27.5]
C0_V = [18., 22., 26., 26., 34., 42.]
K_BP = C0_BP
K_V = [1.079, 0.917, 0.991, 0.915, 0.874, 0.782]
TOE = 10.0        # counts, half-width of the soft toe; the brake-on jump is c0 - TOE
D_EXTRA = 0.0     # m/s^2 below coast before braking (actuator_hysteresis still adds its 0.02 * 2.6 = 0.052)

# DELTA(v): the step the smallest positive interceptor command adds over coasting; pedal-zero = coast + DELTA
DELTA_BP = [4.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0, 27.0, 29.0, 33.0]
DELTA_V = [0.253, 0.323, 0.37, 0.333, 0.28, 0.237, 0.217, 0.193, 0.2, 0.197, 0.193, 0.177, 0.173, 0.183]

# G0(v): interceptor command at net 0 (flat cruise) on the CURRENT pedal calibration (routes >= 000000dd)
G0_BP = [4.0, 6.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]
G0_V = [0.012, 0.035, 0.053, 0.077, 0.1, 0.12, 0.172, 0.19]

# --- today's law, as carcontroller.py has it (test_elesys_brake.py pins these to the real functions) ------------------
FULL_BRAKE_ACCEL = 2.6                       # m/s^2 at full brake: compute_gb_honda_elesys's divisor
BRAKE_COUNTS = 256                           # CarControllerParams.NIDEC_BRAKE_MAX
CREEP_BP = [0., 0.75, 1.75, 3.0, 5.0]        # compute_gb_honda_elesys's creep table
CREEP_V = [1.15, 0.8, 0.45, 0.3, 0.0]
WIND_BP = [0.0, 2.3, 35.0]                   # carcontroller.py's wind_brake (the aero credit; wind_scale is 1.0)
WIND_V = [0.001, 0.002, 0.15]
TODAY_ZB_PER_WB = 2.6                        # today's brake comes on at net = -2.6 * wb ...
TODAY_ZP_PER_WB = 1.95                       # ... and its pedal reaches 0 at net = -1.95 * wb (0.75 * 2.6)
ZP_MAX = -1e-3                               # the pedal-zero point is always below net 0


def wind_brake(v_ego: float) -> float:
  return float(np.interp(v_ego, WIND_BP, WIND_V))


def creep(accel: float, v_ego: float) -> float:
  """compute_gb_honda_elesys's creep term: the table, faded out by positive demand. 0 above 5 m/s."""
  c = float(np.interp(v_ego, CREEP_BP, CREEP_V))
  return c * float(np.clip(1. - max(float(accel), 0.) / 0.8, 0., 1.))


def coast_accel(v_ego: float) -> float:
  return float(np.interp(v_ego, COAST_BP, COAST_V))


def blend(v_ego: float) -> float:
  """0 at V_LO and below (today's law), 1 at V_HI and above (v2)."""
  return float(np.clip((v_ego - V_LO) / (V_HI - V_LO), 0., 1.))


def law_point(v_ego: float) -> tuple[float, float, float, float, float, float]:
  """(zb, c0, k, toe, off, zp) at this speed, blended toward today's equivalents below V_HI.
  zb: net accel at which the brake comes on; zp: net accel at which the pedal reaches 0; off: the pedal at net 0."""
  s = blend(v_ego)
  wb = wind_brake(v_ego)
  coast = coast_accel(v_ego)
  full_creep = creep(-1., v_ego)
  zb2 = coast - full_creep - D_EXTRA
  zp2 = coast - full_creep + float(np.interp(v_ego, DELTA_BP, DELTA_V))
  c02 = float(np.interp(v_ego, C0_BP, C0_V))
  k2 = float(np.interp(v_ego, K_BP, K_V))
  off2 = float(np.interp(v_ego, G0_BP, G0_V))
  off_t = elesys_ff_offset(v_ego, wb)

  def lerp(a, b):
    return a + s * (b - a)
  return (lerp(-TODAY_ZB_PER_WB * wb, zb2), lerp(0., c02), lerp(1., k2), lerp(0., TOE), lerp(off_t, off2),
          lerp(-TODAY_ZP_PER_WB * wb, min(zp2, ZP_MAX)))


def soft_hinge(cb: float, c0: float, toe: float) -> float:
  """Effective counts the brake delivers for a command of cb: 0 below c0 - toe, a quadratic toe, cb - c0 above."""
  if toe <= 0.:
    return max(cb - c0, 0.)
  if cb <= c0 - toe:
    return 0.
  if cb < c0 + toe:
    return (cb - c0 + toe) ** 2 / (4. * toe)
  return cb - c0


def soft_hinge_inverse(u: float, c0: float, toe: float) -> float:
  """The command (counts) that delivers u effective counts beyond the soft dead zone; 0 for u <= 0."""
  if u <= 0.:
    return 0.
  if toe > 0. and u < toe:
    return c0 - toe + 2. * math.sqrt(toe * u)
  return c0 + u


class BrakeLawV2Frame:
  """Brake law v2 for one 100 Hz control frame. Built by law_frame() only where v2 applies; its brake_lin replaces
  compute_gb's brake as actuator_hysteresis's input, brake_frac() replaces the count map, `window` goes to the gas law."""
  __slots__ = ("v_ego", "net", "zb", "c0", "k", "toe", "off", "zp", "brake_lin")

  def __init__(self, net: float, v_ego: float):
    self.v_ego = v_ego
    self.net = net
    self.zb, self.c0, self.k, self.toe, self.off, self.zp = law_point(v_ego)
    self.brake_lin = max(self.zb - net, 0.) / FULL_BRAKE_ACCEL

  def counts(self, brake_last: float) -> float:
    """BRAKE_COMMAND counts (float, before int/clip) for the rate-limited brake fraction."""
    if not math.isfinite(brake_last) or brake_last <= 0.:
      return 0.
    return soft_hinge_inverse(min(brake_last, 1.) * BRAKE_COUNTS / self.k, self.c0, self.toe)

  def brake_frac(self, brake_last) -> float:
    """counts / 256 in [0, 1], for carcontroller's `int(clip(apply_brake * gain * 256, 0, 255))`; exact, since the
    gain is 1.0 and 256 is a power of two. 0.0 for anything non-finite, as today's path falls back."""
    try:
      frac = self.counts(float(brake_last)) / BRAKE_COUNTS
    except Exception:
      return 0.
    return float(min(max(frac, 0.), 1.)) if math.isfinite(frac) else 0.

  @property
  def window(self) -> tuple[float, float, float]:
    """(net, off, zp) for elesys_gas.elesys_pedal_v2_window()."""
    return self.net, self.off, self.zp


def law_frame(accel, v_ego, long_active, long_control_state) -> BrakeLawV2Frame | None:
  """This frame's v2 law, or None where today's path runs: not longActive, not PID, v at or below V_LO, or a
  non-finite input. `accel` is carcontroller's adjust_accel (planner accel + the pitch feedforward). Never raises."""
  try:
    if not long_active or long_control_state != LongCtrlState.pid:
      return None
    v = float(v_ego)
    a = float(accel)
    if not (math.isfinite(v) and math.isfinite(a)) or v <= V_LO:
      return None
    frame = BrakeLawV2Frame(a - creep(a, v), v)
    if not all(math.isfinite(x) for x in (frame.brake_lin, frame.zb, frame.c0, frame.k, frame.toe, frame.off, frame.zp)):
      return None
    return frame
  except Exception:
    return None


def brake_law_v2_enabled(CP, CP_SP, elesys_gas) -> bool:
  """Fixed for the drive at CarController init: the flag (set only with openpilot longitudinal, never in stock ACC
  mode, by _initialize_honda), this car, and gas law v2 - the law's pedal window is built on gas law v2's slope and
  was fitted and simulated with it. With HondaElesysGasLawV2 off it stays off (carlog says so) and today's law runs."""
  try:
    if not int(CP_SP.flags) & HondaFlagsSP.ELESYS_BRAKE_LAW_V2.value:
      return False
    if CP.carFingerprint not in HONDA_ELESYS or not CP.openpilotLongitudinalControl or \
       int(CP_SP.flags) & HondaFlagsSP.ELESYS_STOCK_ACC.value:
      return False
    if elesys_gas is None or not getattr(elesys_gas, "v2", False):
      _warn("HondaElesysBrakeLawV2 is on but gas law v2 is off: brake law v2 needs it, today's brake law runs")
      return False
    return True
  except Exception:
    return False


def _warn(msg: str) -> None:
  try:
    from opendbc.car.carlog import carlog
    carlog.warning(msg)
  except Exception:
    pass
