"""
FORK(HONDA_ELESYS): the comma pedal's gas law on this car (2013-15 Accord AU, HONDA_ELESYS), and
the drive-mode plumbing around it.

GasInterceptorCarController (gas_interceptor.py) hands the ELESYS branch to ElesysGasLaw.update()
and keeps upstream's law for every other car. Everything that is this car's own lives here, so
the upstream file carries one import and one call.

Two laws, picked ONCE at CarController init by HondaElesysGasLawV2 (default on):

  v1  pedal = gm1(v) * (gas - brake + 0.75 * wb)          the shipped law, kept bit-identical
  v2  pedal = off(v) + gas * gm2(v) / MODE_K[slot]        net >= 0
            = off(v) * (1 - brake / (0.75 * wb))           net <  0, i.e. today's negative branch

where `gas = net/4.8` and `brake = -net/2.6` come from compute_gb_honda_elesys(), `wb` is the
aero term (wind_brake), `gm1` is elesys_gas_multiplier() and

  gm2(v) = 4.8 / k(v)                       k: the MEASURED pedal response, m/s^2 per unit pedal
  off(v) = min(G0(v), 0.75 * wb * gm1(v))   G0: the MEASURED flat-cruise pedal; the min keeps
                                            today's offset wherever it is the smaller of the two

Why (routes 99..103, 51 routes, engaged, D, steady pedal 1 s, pedal lagged 0.4 s, route
jackknife; re-derived independently by the skeptic review to within 0.01):

  speed (m/s)                3     6     10    15    20    25    30
  measured k                 8.7   6.8   4.7   3.39  2.89  2.03  1.98
  v1's k = 4.8 / gm1         5.65  4.00  3.10  2.46  1.75  1.75  1.75
  measured cruise pedal G0   .045  .080  .102  .110  .122  .150  .211
  v1's offset 0.75*wb*gm1    .003  .017  .043  .087  .169  .216  .263

v1 asked for 1.4-1.7x too much pedal per m/s^2 at 6-20 m/s. In demand episodes (aTarget >= 0.6
held 1.5 s) aEgo ran 1.34-1.42 against aTarget 0.95-1.15 for the first 2 s and openpilot's
integrator then fell to -0.1..-0.3: the car lunged, then sagged. v2 uses the measured slope.

What v2 deliberately does NOT change (skeptic review, which overrides the audit here):

  * BELOW ~16.9 m/s THE OFFSET IS v1's. The measured G0 is 2-6x higher there, but the pedal has
    to fall from the offset to 0 inside the window between the pedal-zero point (net = -1.95*wb)
    and 0, which is only 0.04-0.12 m/s^2 wide at 6-15 m/s. Raising the offset alone steepens
    that window 1.3-4.8x; with ~0.5 s of plant delay that is a likely surge or limit cycle in
    stop-and-go. So min(G0, v1) -- below ~16.9 m/s v1's offset is the smaller, the window is
    exactly v1's. Above it G0 is smaller, so the window there is GENTLER than v1's (0.76 against
    1.06 per m/s^2 at 20 m/s). Raise the low-speed G0 only together with moving the brake-on
    point to the measured coast deceleration, in the brake work.
  * THE NEGATIVE BRANCH IS v1's SHAPE: a straight line from the offset at net = 0 to zero at
    net = -1.95*wb, so the pedal-zero point and the brake-on point (net = -2.6*wb, carcontroller)
    do not move.
  * 0 AND 3 m/s ARE v1's VALUES (gm 0.55 and 0.85, i.e. k = 8.73 and 5.65), and the table is
    interpolated in the same quantity v1 interpolates (pedal per unit `gas`, 4.8/k), so the
    whole 0-3 m/s segment is v1's law up to the launch cap below. The measured 3 m/s k (8.7) is
    NOT used: it would cut the pedal for every demand, and small ones already under-deliver.
  * The k table only covers pedal up to ~0.25-0.28 (the data's p95). Above that the car
    under-delivers against the line (kickdown, +0.14..+0.35 m/s^2 residual), so "full pedal at
    20 m/s needs 2.55 m/s^2" is an extrapolation, not a measurement.

Expect it to feel SOFTER on take-off from a roll at 6-20 m/s: it removes the 1.3-1.5x onset
over-delivery. That is the measurement, not a regression; the PI adds what is really missing.

LAUNCH CAP (2026-10; route 115 t 511: target 1.6-2.0 m/s^2, aEgo 2.4-2.7, the PI integrator wound
down to -0.8). Below LAUNCH_CAP_V_END = 6 m/s v2 is also capped:

  pedal = min(v2, LAUNCH_CAP_P0 + net / K_launch(v))      P0 0.08; K_launch 13.0 / 12.0 / 6.8 at 0 / 3 / 6

At launch speeds this car answers the pedal like a hinge, not a line through zero: creep only
up to ~0.08-0.09 pedal, then ~13 m/s^2 per unit at 1.5-3 m/s, 10.7 at 3-4.5 and 8.1 at 4.5-6
(48 routes, 099..115, engaged, no pedals, pedal lagged 0.4 s). v1's 0.55-0.85 is a line through
zero, so it asks too little for small demands and too much for large ones; the two cross at
~1 m/s^2. The cap is the hinge's own line, so it only binds above that crossing (net > 0.8 at
any speed): stop-and-go demand gets exactly v2's pedal, a launch gets the pedal the car needs.
It never asks for more than v2 (less pedal from a stop, the safer direction); at net <= 0 it is
P0, above v2's offset (<= 0.017 here), so the pedal-zero window and the brake-on point do not
move; and it is continuous: its slope at 6 m/s is the table's measured k there, with an offset
above v2's, so it stops binding before 6 m/s (from ~5.3 m/s only past the 2 m/s^2 accel limit).

Closed-loop replay (openpilot's PI, that hinge as the plant, each launch's logged target and
grade; validated against the law that ran to ~0.1 m/s^2) of all 13 clean engaged launches from a
stop on 099..115: aEgo/aTarget at 0.5-4.5 m/s 1.18 -> 1.04 with no lead and 1.11 -> 1.02 behind
one, peak aEgo 2.32 -> 1.85 m/s^2, integrator low point -0.44 -> -0.04, time to 6 m/s 0.07 s and
0.03 s quicker (no sag after the over-shoot). The one-number alternative (3 m/s k = 8.7) made the
launches behind a lead 0.11 s slower to 6 m/s and under-deliver (0.94).

DRIVE MODES. mode_slot(): S if the gear is sport, else ECON if ECON is on, else D (unknown or
P/R/N count as D). MODE_K is a per-slot multiplier on k, all 1.0 today, so the slots change
nothing yet: there is 87 s of engaged ECON and 79 s of engaged S in a month of logs, too little
to fit a table (the ECON hints -- 0.61x achieved vs D, slope 0.72-0.84x -- are 70 s of mostly
manual driving on one day). The dynamic tuner logs time and steady-pedal samples per slot in
its `hondadyn` line so per-mode tables can be fitted offline when the data exists. A slot
change crossfades the pedal linearly over CROSSFADE_FRAMES so a future non-1.0 entry can never
step the throttle.

ElesysGasLaw.update() runs inside CarController.update(), which must NEVER raise (an exception
there means no 0x1FA, the VSA latches BRAKE_ERROR ~1 s later and the 0x1A6 stand-down stops ->
ACC/CMBS fault). Every input is treated as possibly NaN, None or missing; anything that goes
wrong falls back to v1, then to 0.0 (no throttle), never to an exception.
"""

import math

import numpy as np

from opendbc.car import structs

# --- the v1 law (what shipped before v2) --------------------------------------------------------
# Moved here from gas_interceptor.py, which re-exports the names. Unchanged; see the note on
# elesys_gas_multiplier() for its history.
ELESYS_GAS_BP = [0., 3., 6., 10., 15., 20.]
ELESYS_GAS_V = [0.55, 0.85, 1.20, 1.55, 1.95, 2.75]


def elesys_gas_multiplier(v_ego: float) -> float:
  # v1: pedal->accel gain falls off with speed (measured ~4.8 @ 10 m/s, ~3.5 @ 14, ~2.2 @ 18,
  # route ac35d9891f). The previous curve ([0,10,15,20] -> [0.5,1.0,1.4,2.1]) fixed the ends but
  # left the mid band on the old <=1.0 ramp: the Jul-14/15 drives (e64ef42e91/4a64f0ffb2/
  # 2418f2eb2b) delivered only ~50-65% of commanded accel at 3-14 m/s.
  #
  # 2026-08 revision: settled-frame plant identification (~200k frames, three routes, two tunes)
  # put the command needed at 1.38x @ 3-6 m/s ... 1.50-1.75x @ 20+, so the top three breakpoints
  # went up ~1.25x (1.10/1.25/1.55/2.20 -> 1.20/1.55/1.95/2.75), one measured step.
  #
  # 2026-10: that identification regressed demand on the command at "target steady >= 1.5 s",
  # which on this car mostly means cruise, where the offset (wind_brake) and the slope cannot be
  # told apart. Fitting slope and offset separately on 51 routes says the opposite at 6-20 m/s:
  # this curve gives 1.4-1.7x TOO MUCH pedal per m/s^2. v2 (above) uses that fit; this curve is
  # kept as the v1 law, and as the source of v2's offset below 16.9 m/s and of v2's 0-3 m/s gain.
  return float(np.interp(v_ego, ELESYS_GAS_BP, ELESYS_GAS_V))


def elesys_pedal_v1(v_ego: float, gas: float, brake: float, wind_brake: float) -> float:
  """The shipped law, bit-identical to what gas_interceptor.py computed before v2 (the learned
  pedal gain it used to be multiplied by is retired at exactly 1.0)."""
  return float(np.clip(elesys_gas_multiplier(v_ego) * (gas - brake + wind_brake * 3 / 4), 0., 1.))


# --- the v2 law -----------------------------------------------------------------------------------
ELESYS_FF_BP = [0., 3., 6., 10., 15., 20., 25., 30.]
# m/s^2 of net accel per unit of interceptor command. 0 and 3 m/s are v1's (4.8/0.55, 4.8/0.85),
# written that way so the 0-3 m/s segment is exactly v1's; the rest are the measured fit.
ELESYS_FF_K = [4.8 / 0.55, 4.8 / 0.85, 6.8, 4.7, 3.4, 2.9, 2.0, 2.0]
# What the law interpolates: pedal per unit `gas` (= net/4.8), the same quantity as ELESYS_GAS_V,
# so np.interp between 0 and 3 m/s computes v1's multiplier bit for bit. 0.55/0.85 are spelled
# out rather than derived, because 4.8 / (4.8 / 0.55) need not round back to 0.55.
ELESYS_FF_GM = [0.55, 0.85] + [4.8 / k for k in ELESYS_FF_K[2:]]
# Measured flat-cruise pedal (engaged, D, steady). 0 m/s has no data (nothing cruises there); it
# holds the 3 m/s value. Below ~16.9 m/s the min() with v1's offset never picks these.
ELESYS_FF_G0 = [0.045, 0.045, 0.080, 0.102, 0.110, 0.122, 0.150, 0.211]

# v1's offset is OFFSET_PER_WB * wb * gm1, and its pedal reaches 0 at brake == OFFSET_PER_WB * wb,
# i.e. net = -1.95 * wb. v2 keeps that pedal-zero point for whatever offset it uses.
OFFSET_PER_WB = 0.75

# The launch cap (docstring): below LAUNCH_CAP_V_END v2 never sends more than P0 + net / K. K is
# the measured launch response above the pedal the car starts to pull at (13.1 at 1.5-3 m/s, 10.7
# at 3-4.5, 8.1 at 4.5-6 against 12.3 / 10.6 / 8.2 here), ending on the table's measured k at
# 6 m/s so the cap meets v2 there instead of stepping off it.
LAUNCH_CAP_BP = [0., 3., ELESYS_FF_BP[2]]
LAUNCH_CAP_K = [13.0, 12.0, ELESYS_FF_K[2]]
LAUNCH_CAP_P0 = 0.08      # the pedal at which the car starts to pull from a crawl, measured 0.08-0.09
LAUNCH_CAP_V_END = ELESYS_FF_BP[2]


def _finite(x, fallback: float = 0.0) -> float:
  try:
    v = float(x)
  except (TypeError, ValueError):
    return fallback
  return v if math.isfinite(v) else fallback


def elesys_ff_gm(v_ego: float) -> float:
  """v2 pedal per unit `gas` (4.8 / k) at this speed."""
  return float(np.interp(v_ego, ELESYS_FF_BP, ELESYS_FF_GM))


def elesys_ff_offset(v_ego: float, wind_brake: float) -> float:
  """v2 pedal at net = 0: the measured cruise pedal, but never above v1's offset."""
  today = elesys_gas_multiplier(v_ego) * (wind_brake * 3 / 4)
  return min(float(np.interp(v_ego, ELESYS_FF_BP, ELESYS_FF_G0)), today)


def elesys_launch_cap(v_ego: float, gas: float, k_mult: float = 1.0) -> float:
  """The most pedal v2 sends below LAUNCH_CAP_V_END for this `gas` (= net/4.8): P0 + net/K(v)."""
  return LAUNCH_CAP_P0 + gas * 4.8 / (float(np.interp(v_ego, LAUNCH_CAP_BP, LAUNCH_CAP_K)) * k_mult)


def elesys_pedal_v2(v_ego: float, gas: float, brake: float, wind_brake: float, k_mult: float = 1.0) -> float:
  """The v2 law, clipped to [0, 1]. `gas` and `brake` are compute_gb_honda_elesys()'s fractions
  (one of them is 0), `k_mult` the drive-mode multiplier on k. Below LAUNCH_CAP_V_END the launch
  cap applies. Never raises; 0.0 for anything non-finite."""
  try:
    v = float(v_ego)
    g = max(_finite(gas), 0.0)
    b = max(_finite(brake), 0.0)
    m = _finite(k_mult, 1.0)
    if m <= 0.0:
      m = 1.0
    half = OFFSET_PER_WB * _finite(wind_brake)
    if half > 1e-9:
      off = elesys_ff_offset(v, _finite(wind_brake))
      neg_slope = off / half                 # pedal per unit `brake`: 0 at brake == 0.75 * wb
    else:
      off = neg_slope = 0.0                  # no aero term to anchor the window on: no offset
    pedal = off + g * (elesys_ff_gm(v) / m) - b * neg_slope
    if v < LAUNCH_CAP_V_END:
      pedal = min(pedal, elesys_launch_cap(v, g, m))
    pedal = float(np.clip(pedal, 0., 1.))
  except Exception:
    return 0.0
  return pedal if math.isfinite(pedal) else 0.0


# --- drive modes ----------------------------------------------------------------------------------
DRIVE_MODE_SLOTS = ("D", "ECON", "S")
# Per-slot multiplier on k (so pedal per m/s^2 is divided by it). 1.0 everywhere until there is
# engaged ECON/S data to fit; the skeptic review: if ECON gets a prior, use >= 0.85, not 1/1.3.
MODE_K = {"D": 1.0, "ECON": 1.0, "S": 1.0}
# Interceptor frames (50 Hz) over which a slot change is faded in: 2.0 s.
CROSSFADE_FRAMES = 100


def _gear_names() -> dict:
  """ordinal -> name for structs.CarState.GearShifter, whatever backs it."""
  enum = structs.CarState.GearShifter
  try:                                          # capnp
    return {int(v): k for k, v in enum.schema.enumerants.items()}
  except Exception:
    pass
  try:                                          # python Enum
    return {int(m.value): m.name for m in enum}
  except Exception:
    return {}


_GEAR_NAMES = _gear_names()


def gear_name(CS):
  """Gear as a plain name, or None if it cannot be determined.

  gearShifter reaches us in three different shapes depending on who built the message: a bare
  int from a direct assignment, a capnp _DynamicEnum from a reader (str() gives the name, .raw
  the ordinal), or already a string. Getting this wrong fails silently, so normalize all three.
  """
  try:
    gear = getattr(CS.out, "gearShifter", None)
    if gear is None:
      return None
    raw = getattr(gear, "raw", None)
    if raw is not None:
      name = _GEAR_NAMES.get(int(raw))
    elif isinstance(gear, bool):
      return None
    elif isinstance(gear, int):
      name = _GEAR_NAMES.get(int(gear))
    else:
      name = str(gear).rsplit(".", 1)[-1]
    return None if name in (None, "unknown") else name
  except Exception:
    return None


def econ_state(CS):
  """True/False where ECON is decoded (carstate.py sets CS.econ_on on HONDA_ELESYS from 0x221),
  None where it is not observable. Read off the CarState object, not CS.out: ECON is not a
  CarState field."""
  try:
    econ = getattr(CS, "econ_on", None)
    return None if econ is None else bool(econ)
  except Exception:
    return None


def drive_mode_slot(gear, econ) -> str:
  """S if the gear is sport, else ECON if ECON is on, else D. Unknown counts as D. There is no
  S+ECON slot: S overrides ECON's map (and there were 0 s of it in the logs)."""
  if gear == "sport":
    return "S"
  if econ:
    return "ECON"
  return "D"


def mode_slot(CS) -> str:
  try:
    return drive_mode_slot(gear_name(CS), econ_state(CS))
  except Exception:
    return "D"


class ModeCrossfade:
  """Weights over the slots, moved linearly from where they are to the new slot's one-hot over
  CROSSFADE_FRAMES whenever the slot changes -- including in the middle of a fade, which then
  starts from the current blend rather than stepping back to the old slot. So at a constant
  input the blended pedal moves by exactly (p_new - p_start_blend) / CROSSFADE_FRAMES per call."""

  def __init__(self, frames: int = CROSSFADE_FRAMES):
    self.frames = max(int(frames), 1)
    self.slot = None
    self.weights: dict[str, float] = {}
    self._start: dict[str, float] = {}
    self._n = self.frames

  def step(self, slot: str) -> dict[str, float]:
    if self.slot is None:                      # the first frame of a drive: no fade from nothing
      self.slot = slot
      self.weights = {slot: 1.0}
      self._n = self.frames
      return self.weights
    if slot != self.slot:
      self.slot = slot
      self._start = dict(self.weights)
      self._n = 0
    if self._n < self.frames:
      self._n += 1
      a = self._n / self.frames
      w = {s: (1.0 - a) * x for s, x in self._start.items()}
      w[slot] = w.get(slot, 0.0) + a
      self.weights = {s: x for s, x in w.items() if x > 0.0}
    else:
      self.weights = {slot: 1.0}
    return self.weights


# --- the param ------------------------------------------------------------------------------------
GAS_LAW_PARAM = "HondaElesysGasLawV2"
GAS_LAW_DEFAULT = True      # params_keys.h: {PERSISTENT | BACKUP, BOOL, "1"}


def _open_params():
  """Params lives in openpilot, this module lives in opendbc: import lazily so opendbc still
  imports standalone (PC tests, CI)."""
  try:
    from openpilot.common.params import Params
    return Params()
  except Exception:
    return None


def read_gas_law_v2(params=None) -> bool:
  """HondaElesysGasLawV2, read once. Anything that goes wrong -- no openpilot, a registry that
  predates the key (UnknownKeyName), a value that is not a bool -- gives the registered default."""
  try:
    if params is None:
      params = _open_params()
    if params is None:
      return GAS_LAW_DEFAULT
    value = params.get(GAS_LAW_PARAM, return_default=True)
    if value is None:
      return GAS_LAW_DEFAULT
    if isinstance(value, (bool, int, float)):
      return bool(value)
    text = value.decode() if isinstance(value, bytes) else str(value)
    text = text.strip().lower()
    if text in ("1", "true"):
      return True
    if text in ("0", "false"):
      return False
  except Exception:
    pass
  return GAS_LAW_DEFAULT


class ElesysGasLaw:
  """The interceptor command for HONDA_ELESYS. Constructed with CarController (so the param is
  read once per drive, at ignition) and called at the interceptor rate (50 Hz)."""

  def __init__(self, params=None):
    self.v2 = read_gas_law_v2(params)
    self.law = "v2" if self.v2 else "v1"
    self.blend = ModeCrossfade()

  def _v2(self, v_ego, gas, brake, wind_brake) -> float:
    slot = self.blend.slot or "D"
    p = elesys_pedal_v2(v_ego, gas, brake, wind_brake, MODE_K.get(slot, 1.0))
    # p + sum(w_s * (p_s - p)) over the slots still fading out, which is sum(w_s * p_s) because
    # the weights sum to 1 -- but written this way it is EXACTLY p when every slot's law is the
    # same (MODE_K all 1.0 today), whatever the weights are
    out = p
    for s, w in self.blend.weights.items():
      if s != slot and w > 0.0:
        out += w * (elesys_pedal_v2(v_ego, gas, brake, wind_brake, MODE_K.get(s, 1.0)) - p)
    return out

  def update(self, CC, CS, gas: float, brake: float, wind_brake: float) -> float:
    """The pedal to send, in [0, 1]. 0.0 when not longActive. Never raises."""
    try:
      self.blend.step(mode_slot(CS))
      if not CC.longActive:
        return 0.0
      pedal = self._v2(CS.out.vEgo, gas, brake, wind_brake) if self.v2 else \
        elesys_pedal_v1(CS.out.vEgo, gas, brake, wind_brake)
    except Exception:
      try:
        pedal = elesys_pedal_v1(CS.out.vEgo, gas, brake, wind_brake) if CC.longActive else 0.0
      except Exception:
        pedal = 0.0
    if not (isinstance(pedal, float) and math.isfinite(pedal)):
      return 0.0
    # an identity on every value either law produces (both are clipped already); it only catches
    # a crossfade blend whose weights summed to a hair over 1
    return min(max(pedal, 0.0), 1.0)
