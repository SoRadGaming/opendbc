"""
FORK(HONDA_ACCORD_9G_AU): SHADOW longitudinal learners. They measure, and write into the route, what a
learner WOULD learn. They change nothing that is actuated: no command, no gain, no param, no message
anything else reads. The owner decided (2026-10, routes 114/115) that the brake table and the launch
multiplier run like this first, so several drives of data can be judged before anything is applied.

  * L1 BRAKE RESPONSE TABLE. The scalar brake gain (dynamic_tuning.py) cannot represent this brake:
    light-to-mid commands deliver 0.5-0.8x of the /2.6 law and firm ones ~1.0x (A_learning, routes
    10f/115), and its gate admits ~7-18 s of a drive. This measures, per speed band x command band,
    the achieved net deceleration against the command the law was given, and per speed band the
    coast deceleration with neither pedal nor brake (the drag / brake-on point term).
  * L2b LAUNCH MULTIPLIER. Below 6 m/s the gas law over-delivers (route 115 t 511: target 1.6-2.0,
    achieved 2.4-2.7 m/s^2). This measures achieved / requested at 0.5-3 and 3-6 m/s and the bounded
    pedal multiplier (<= 1.0: it may only ever take pedal away) that would cancel it.

THE SIGNAL. Each sample compares the net acceleration the car achieved, aEgo + g*sin(pitch) (gravity
removed), with the command the gas/brake law was handed (actuators.accel + the tuner's pitch term)
pushed through the tuner's first-order plant model (PLANT_TAU 0.3 s, `cmd_ref`). A perfect law gives
zero error; what is left is the law's error at that speed and command. That is the same quantity
openpilot's longitudinal integrator (controlsState.uiAccelCmd) converges to cancel, which is what
A_synth L1 proposed learning from -- but the integrator is not visible in card, it also carries the
planner's own transients, and it is slow; the plant error is visible here, now, and needs no
integrator to have converged. The offline report checks the two agree (shadow_learn_report.py).

WHY NOTHING HERE CAN REACH AN ACTUATOR:
  * the tuner hands it COPIES of numbers (floats) and never reads anything back from it;
  * it is called last in the 50 Hz gas/brake block (from HondaDynamicTuner.update_wind(), after
    brake_gain() and observe_pedal() have produced and recorded this frame's commands);
  * any exception switches it off for the rest of the drive (HondaDynamicTuner._shadow_update);
  * it owns no Params and no writer thread.
The replay check in shadow_learn_report.py's docstring is how that was proven on routes 115 and 10f.

LOG. One `hondashadow` line (carlog -> card -> cloudlog -> logMessage) every LOG_INTERVAL while
something new was admitted, and one at every disengage, so the last line of a route is the drive's
total. The totals are per drive (they start at zero at every ignition); the per-cell counts are in
the line, so routes are combined offline by weighting with them. Format: `key=value` tokens, lists in
[...] (row-major for the brake table: speed band, then command band), `nan` for an empty cell -- the
same shape parse_hondadyn.py already reads.
"""

import math
from collections import deque

from opendbc.car import ACCELERATION_DUE_TO_GRAVITY, structs
from opendbc.car.carlog import carlog

LongCtrlState = structs.CarControl.Actuators.LongControlState

LOG_TAG = "hondashadow"
LOG_VERSION = 1
RATE_HZ = 50                  # called once per 50 Hz gas/brake frame
LOG_INTERVAL = 60 * RATE_HZ   # a summary line at most once a minute

NIDEC_BRAKE_MAX = 256         # CarControllerParams.NIDEC_BRAKE_MAX: brake fraction -> 0x1FA counts

# The command that produced this sample's acceleration went out ~0.25-0.40 s earlier (A_learning:
# best pure delay aTarget -> aEgo 0.40 s on 115, 0.25 s on 10f). So a sample is admitted only when the
# pedal and brake commands have held their state for the whole of that window.
WINDOW = 20                   # 50 Hz samples = 0.4 s
MAX_PITCH = math.radians(2.0) # brake/coast: above this, grade error swamps the measurement (A_synth L1)
# The jerk gate of A_synth L1, on the plant model's own ramp rate (the tuner's cmd_ref): the tuner's 0.5 m/s^3
# (LEARN_MAX_JERK), held for STEADY_HOLD samples. The tuner's own dwell holds it a full second (SETTLE_FRAMES);
# on route 115 that admits 18.5 s of the 107 s of brake-commanded PID time, a 0.2 s hold admits 27 s, and the
# cell means agree within 0.03 m/s^2 -- so the shorter hold, with its own counter here.
STEADY_MAX_RAMP = 0.5         # m/s^3
STEADY_HOLD = 10              # samples (0.2 s)

# --- L1: brake table -------------------------------------------------------------------------------
SPEED_BP = (1., 5., 10., 15., 20., 25.)   # band i is [SPEED_BP[i], SPEED_BP[i+1]); the last is open
BRAKE_MIN_COUNTS = 4.         # below this the command is pump/rounding noise, not a brake request
BRAKE_COUNT_BP = (60., 100.)  # command bands <= 60, 60-100, > 100 counts (A_synth L1)
BRAKE_STEADY_COUNTS = 20.     # the command may move at most this much across WINDOW (~0.2 m/s^2)
COAST_MIN_SPEED = 3.0         # m/s; below this the coast deceleration is mostly creep, not drag
# What an actuating version would be allowed to apply, per cell: an additive accel correction that
# may only ADD braking (never learn below the law's braking -- the asymmetric floor of A_synth L1) and
# at most 0.5 m/s^2 of it, and only from a cell with at least MIN_CELL_SAMPLES behind it.
BRAKE_CORR_MIN = -0.5
BRAKE_CORR_MAX = 0.0
MIN_CELL_SAMPLES = 250        # 5 s of admitted samples
BRAKE_FADE_SPEED = (1.0, 2.0) # the would-apply correction fades in over this speed range

# --- L2b: launch multiplier ------------------------------------------------------------------------
LAUNCH_SPEED_BP = (0.5, 3.0, 6.0)   # two bands: 0.5-3 m/s (the v1 segment) and 3-6 m/s
LAUNCH_MIN_REQ = 0.3          # m/s^2 of (lagged) command: below this the ratio is noise
LAUNCH_MIN_PEDAL = 0.01       # the interceptor must actually be pressing for the whole WINDOW
# A launch ramps faster than the brake channel's dwell (0.5 m/s^3) ever admits, so this uses the plant
# model's own ramp rate with a looser cap: the tau uncertainty (+-0.2 s) times 1.0 m/s^3 leaves at most
# 0.2 m/s^2 of residual against the ~0.6-0.8 m/s^2 over-delivery being measured.
LAUNCH_MAX_RAMP = 1.0         # m/s^3 on cmd_ref
LAUNCH_RAMP_HOLD = 10         # samples (0.2 s) the ramp must have stayed under the cap
# Wider than the brake's 2 deg: launches happen from a standstill, where the device's pitch is at its most
# trustworthy (gravity is the only acceleration), and a 2 deg gate threw away route 115's only launch (a
# -2.7 deg downhill, where gravity alone was +0.46 of the 2.4-2.7 m/s^2 the car showed).
LAUNCH_MAX_PITCH = math.radians(4.0)
LAUNCH_MULT_MIN = 0.6         # the multiplier may take away at most 40% of the pedal...
LAUNCH_MULT_MAX = 1.0         # ...and never add any (A_synth L2b: "<= 1.0 at first")
LAUNCH_MIN_SAMPLES = 100      # 2 s of admitted launch samples


def _finite(x, fallback: float = float("nan")) -> float:
  try:
    v = float(x)
  except (TypeError, ValueError):
    return fallback
  return v if math.isfinite(v) else fallback


def _band(x: float, bp) -> int:
  """How many edges of bp are <= x: 0 below bp[0], len(bp) at or above the last edge."""
  i = 0
  while i < len(bp) and x >= bp[i]:
    i += 1
  return i


def speed_band(v: float) -> int:
  """0..len(SPEED_BP)-1 for v >= SPEED_BP[0]; -1 below it."""
  return _band(v, SPEED_BP) - 1


def count_band(counts: float) -> int:
  """0: <= 60, 1: 60-100, 2: > 100 counts."""
  if counts <= BRAKE_COUNT_BP[0]:
    return 0
  return 1 if counts <= BRAKE_COUNT_BP[1] else 2


def launch_band(v: float) -> int:
  """0: 0.5-3 m/s, 1: 3-6 m/s, -1 outside."""
  if not (LAUNCH_SPEED_BP[0] <= v < LAUNCH_SPEED_BP[-1]):
    return -1
  return _band(v, LAUNCH_SPEED_BP) - 1


def shadow_applicable(CP) -> bool:
  """Only the Elesys Accord: the bands, the window and the brake law they measure are this car's."""
  try:
    from opendbc.car.honda.values import HONDA_ELESYS
    return CP is not None and CP.carFingerprint in HONDA_ELESYS
  except Exception:
    return False


class _Mean:
  """Running sum, sum of squares and count. Means, not EMAs: nothing here acts, so nothing can wind up,
  and a plain mean with its count is what combines exactly across drives."""
  __slots__ = ("n", "s", "ss")

  def __init__(self):
    self.n = 0
    self.s = 0.0
    self.ss = 0.0

  def add(self, x: float) -> None:
    self.n += 1
    self.s += x
    self.ss += x * x

  def mean(self) -> float:
    return self.s / self.n if self.n else float("nan")

  def std(self) -> float:
    if self.n < 2:
      return float("nan")
    m = self.s / self.n
    return math.sqrt(max(self.ss / self.n - m * m, 0.0))


class _Ratio:
  """Least squares through the origin of achieved on requested: sum(r*a) / sum(r*r)."""
  __slots__ = ("n", "ra", "rr")

  def __init__(self):
    self.n = 0
    self.ra = 0.0
    self.rr = 0.0

  def add(self, req: float, ach: float) -> None:
    self.n += 1
    self.ra += req * ach
    self.rr += req * req

  def ratio(self) -> float:
    return self.ra / self.rr if self.n and self.rr > 0.0 else float("nan")


def launch_multiplier(n: int, ra: float, rr: float) -> float:
  """The pedal multiplier the launch learner would apply: 1/ratio, bounded to [0.6, 1.0], and 1.0
  until LAUNCH_MIN_SAMPLES are behind it."""
  if n < LAUNCH_MIN_SAMPLES or rr <= 0.0 or ra <= 0.0:
    return 1.0
  return min(max(rr / ra, LAUNCH_MULT_MIN), LAUNCH_MULT_MAX)


def brake_cell_correction(n: int, mean_err: float) -> float:
  """The additive accel correction one brake cell would apply (m/s^2, <= 0 = more braking). The error is
  achieved - commanded, so an under-braking cell (positive error) asks for more brake; an over-braking
  one is floored at 0 -- the table may never take braking away from the law."""
  if n < MIN_CELL_SAMPLES or not math.isfinite(mean_err):
    return 0.0
  return min(max(-mean_err, BRAKE_CORR_MIN), BRAKE_CORR_MAX)


def _fmt(values, spec: str) -> str:
  return "[" + ",".join(("nan" if isinstance(v, float) and not math.isfinite(v) else format(v, spec)) for v in values) + "]"


class HondaShadowLearners:
  def __init__(self):
    n_speed, n_cmd = len(SPEED_BP), len(BRAKE_COUNT_BP) + 1
    self.brake = [[_Mean() for _ in range(n_cmd)] for _ in range(n_speed)]
    # the command and the achieved accel behind each brake cell, so the table also reads as "delivered
    # against the law" (A_learning's 0.5-0.8x at light commands) without a second pass over the route
    self.brake_counts = [[_Mean() for _ in range(n_cmd)] for _ in range(n_speed)]
    self.brake_acc = [[_Mean() for _ in range(n_cmd)] for _ in range(n_speed)]
    self.coast_err = [_Mean() for _ in range(n_speed)]
    self.coast_acc = [_Mean() for _ in range(n_speed)]
    self.launch = [_Ratio() for _ in range(len(LAUNCH_SPEED_BP) - 1)]
    self.launch_episodes = 0
    self._in_launch = False

    self._counts: deque = deque(maxlen=WINDOW)
    self._gas: deque = deque(maxlen=WINDOW)
    self._prev_ref = 0.0
    self._steady = 0
    self._ramp_ok = 0
    self._long_active = False
    self._frames = 0
    self._dirty = False
    self.lines = 0

  # --- per 50 Hz frame -------------------------------------------------------------------------------

  def update(self, CC, CS, *, pitch: float, pose_fresh: bool, mode_ok: bool,
             cmd_ref: float, brake_frac: float, gas_cmd: float) -> None:
    """One 50 Hz sample. Every argument is a value the tuner already computed for this frame; nothing
    is returned and nothing outside this object is written."""
    long_active = bool(CC.longActive)
    counts = _finite(brake_frac, 0.0) * NIDEC_BRAKE_MAX
    gas = _finite(gas_cmd, 0.0)
    ref = _finite(cmd_ref, 0.0)
    self._counts.append(counts)
    self._gas.append(gas)
    ramp = abs(ref - self._prev_ref) * RATE_HZ
    self._prev_ref = ref
    self._steady = self._steady + 1 if ramp <= STEADY_MAX_RAMP else 0
    self._ramp_ok = self._ramp_ok + 1 if ramp <= LAUNCH_MAX_RAMP else 0

    # the drive's running totals go out at every disengage and once a minute, if anything was added
    disengaged = self._long_active and not long_active
    self._long_active = long_active
    self._frames += 1
    if self._dirty and (disengaged or self._frames % LOG_INTERVAL == 0):
      self.emit()

    launched = self._sample(CC, CS, long_active, pitch, pose_fresh, mode_ok, ref, counts)
    if launched and not self._in_launch:
      self.launch_episodes += 1
    self._in_launch = launched

  def _sample(self, CC, CS, long_active, pitch, pose_fresh, mode_ok, ref, counts) -> bool:
    """Admit at most one sample into one table. Returns True if it was a launch sample."""
    out = CS.out
    if not (long_active and mode_ok and pose_fresh
            and CC.actuators.longControlState == LongCtrlState.pid
            and not out.gasPressed and not out.brakePressed and not out.stockAeb
            and len(self._counts) == WINDOW):
      return False
    pitch = _finite(pitch)
    v = _finite(out.vEgo)
    a = _finite(out.aEgo)
    if not (abs(pitch) < max(MAX_PITCH, LAUNCH_MAX_PITCH) and math.isfinite(v) and math.isfinite(a)):
      return False

    achieved = a + ACCELERATION_DUE_TO_GRAVITY * math.sin(pitch)
    err = achieved - ref
    gas_lo, gas_hi = min(self._gas), max(self._gas)
    cb_lo, cb_hi = min(self._counts), max(self._counts)

    if gas_hi <= 0.0:
      sb = speed_band(v)
      if sb < 0 or self._steady < STEADY_HOLD or not abs(pitch) < MAX_PITCH:
        return False
      if cb_lo >= BRAKE_MIN_COUNTS and cb_hi - cb_lo <= BRAKE_STEADY_COUNTS and count_band(cb_lo) == count_band(cb_hi):
        cb = count_band(counts)
        self.brake[sb][cb].add(err)
        self.brake_counts[sb][cb].add(counts)
        self.brake_acc[sb][cb].add(achieved)
        self._dirty = True
      elif cb_hi <= 0.0 and v >= COAST_MIN_SPEED:
        self.coast_err[sb].add(err)
        self.coast_acc[sb].add(achieved)
        self._dirty = True
      return False

    lb = launch_band(v)
    if (lb >= 0 and abs(pitch) < LAUNCH_MAX_PITCH and cb_hi <= 0.0 and gas_lo >= LAUNCH_MIN_PEDAL and ref >= LAUNCH_MIN_REQ
            and self._ramp_ok >= LAUNCH_RAMP_HOLD):
      self.launch[lb].add(ref, achieved)
      self._dirty = True
      return True
    return False

  # --- what it would learn ---------------------------------------------------------------------------

  def brake_corrections(self) -> list:
    return [[brake_cell_correction(c.n, c.mean()) for c in row] for row in self.brake]

  def brake_correction(self, v_ego: float, counts: float) -> float:
    """What an actuating version would add to the brake request at this speed and command. NOT CALLED
    by anything on the car -- it is here so the bounds, the floor and the fade are tested on the same
    code the log reports."""
    v = _finite(v_ego, 0.0)
    sb = speed_band(v)
    if sb < 0 or _finite(counts, 0.0) < BRAKE_MIN_COUNTS:
      return 0.0
    lo, hi = BRAKE_FADE_SPEED
    fade = min(max((v - lo) / (hi - lo), 0.0), 1.0)
    cell = self.brake[sb][count_band(counts)]
    return fade * brake_cell_correction(cell.n, cell.mean())

  def launch_pooled(self) -> tuple:
    n = sum(r.n for r in self.launch)
    ra = sum(r.ra for r in self.launch)
    rr = sum(r.rr for r in self.launch)
    return n, ra, rr

  # --- telemetry -------------------------------------------------------------------------------------

  def line(self) -> str:
    cells = [c for row in self.brake for c in row]
    n, ra, rr = self.launch_pooled()
    corr = [x for row in self.brake_corrections() for x in row]
    return (f"{LOG_TAG} v={LOG_VERSION} " +
            f"bspd={_fmt(SPEED_BP, 'g')} bcnt={_fmt((BRAKE_MIN_COUNTS,) + BRAKE_COUNT_BP, 'g')} " +
            f"bn={_fmt([c.n for c in cells], 'd')} be={_fmt([c.mean() for c in cells], '+.3f')} " +
            f"bsd={_fmt([c.std() for c in cells], '.3f')} bcorr={_fmt(corr, '+.3f')} " +
            f"bcb={_fmt([c.mean() for row in self.brake_counts for c in row], '.1f')} " +
            f"bacc={_fmt([c.mean() for row in self.brake_acc for c in row], '+.3f')} " +
            f"cn={_fmt([c.n for c in self.coast_acc], 'd')} cacc={_fmt([c.mean() for c in self.coast_acc], '+.3f')} " +
            f"cerr={_fmt([c.mean() for c in self.coast_err], '+.3f')} " +
            f"lspd={_fmt(LAUNCH_SPEED_BP, 'g')} ln={_fmt([r.n for r in self.launch], 'd')} " +
            f"lra={_fmt([r.ra for r in self.launch], '.4g')} lrr={_fmt([r.rr for r in self.launch], '.4g')} " +
            f"lratio={_fmt([r.ratio() for r in self.launch], '.3f')} lep={self.launch_episodes} " +
            f"lmult={launch_multiplier(n, ra, rr):.3f}")

  def emit(self) -> None:
    self._dirty = False
    self.lines += 1
    carlog.info(self.line())
