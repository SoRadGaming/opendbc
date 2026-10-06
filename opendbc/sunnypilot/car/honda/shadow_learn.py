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
    pedal multiplier (<= 1.0: it may only ever take pedal away) that would cancel it. A_synth L2b learns
    from launches with no lead and with the pedal confirmed on 0x17C PEDAL_GAS: both are gates here, and
    launches behind a lead are kept in their own sums (logged, never in the multiplier).

THE SIGNAL. Each sample compares the net acceleration the car achieved, aEgo + g*sin(pitch) (gravity
removed), with the command the gas/brake law was handed (actuators.accel + the tuner's pitch term)
pushed through the tuner's first-order plant model (PLANT_TAU 0.3 s, `cmd_ref`). A perfect law gives
zero error; what is left is the law's error at that speed and command.

THIS DEPARTS FROM A_synth L1, which learns from openpilot's longitudinal integrator
(controlsState.uiAccelCmd) on brake frames. In a steady state the two should agree (the integrator
settles where achieved = target, so error = -(uiAccelCmd + the P term)), but the integrator is not
visible in card, so this uses the plant error. Whether they DO agree is checked offline, not assumed:
the one-off replay in the CAR doc (9.3) bins -uiAccelCmd on exactly these gates beside `be`. On 115 and
10f they agree within 0.06 m/s^2 above 20 m/s and do NOT below 15 m/s (up to 0.18 apart, one cell of
opposite sign), so the brake table is not validated there.

The command it measures against is the LAW'S command at the brake gain then in force: brake_frac is
recorded in HondaDynamicTuner.brake_gain(), before that gain, the soft final stop ceiling and the
release limiter (carcontroller.py), so `bcb` is not the 0x1FA count on the wire and the error already
contains whatever the live scalar gain corrects. Each line logs the mean gain over its brake samples
(`bgain`); a table applied on top of the gain would have to be read against it.

WHY NOTHING HERE CAN REACH AN ACTUATOR:
  * the tuner hands it numbers it already computed and never reads anything back from it; the CC and
    CS it is handed are only read;
  * it is called last in the 50 Hz gas/brake block (from HondaDynamicTuner.update_wind(), after
    brake_gain() and observe_pedal() have produced and recorded this frame's commands);
  * any exception switches it off for the rest of the drive (HondaDynamicTuner._shadow_update), and the
    tuner imports and builds it inside try blocks;
  * it owns no Params and no writer thread.
test_shadow_learn's CarController test and the route replays in the CAR doc (9.3) prove it: byte-
identical CAN with and without it.

THE GATES are on a run of clean frames, not just this one: CLEAN_HOLD (1 s) of engaged PID control
with no driver pedal, no stock AEB, D, a fresh pose. A driver's throttle stays in aEgo after the pedal
is released (route 10f: 0.38 s median, 0.76 s p90 before aEgo is back within 0.15 of the command),
and the pedal/brake WINDOW alone only holds the law's commands, which read zero during an override.
THE LAUNCH counts its clean second through the stop instead (batch 3, learnaudit G6): it needs the same
engaged, pedal-free, AEB-free, D, fresh-pose run, in ANY control state, so a launch from an openpilot-held
stop is sampled from the first wheel motion. Waiting for a second of PID meant waiting for control to
leave the stopping state, and by then the car was at 0.61-1.07 m/s (115 t 511, 10f t 303 and t 2408):
the 0.5-1 m/s slice, where the overshoot is worst, was never sampled.

RUNS WITHOUT THE TUNER'S LIVE PARTS (batch 3). The tuner builds this on the Elesys Accord with the
interceptor and openpilot longitudinal whether or not HondaDynamicTuningEnabled is on: with the toggle off
the tuner still tracks pitch and its plant model for this, and still applies nothing (dynamic_tuning.py).
Never in stock ACC mode: openpilot longitudinal is off there, so the tuner does not apply.

LOG. One `hondashadow` line (carlog -> card -> cloudlog -> logMessage) every LOG_INTERVAL while
something new was admitted, one at every disengage, and one when card exits at ignition-off (flush()),
so the last line of a route is the drive's total. The totals are per drive (they start at zero at every
ignition); the per-cell counts are in the line, so routes are combined offline by weighting with them.
Format: `key=value` tokens, lists in [...] (row-major for the brake table: speed band, then command
band), `nan` for an empty cell -- the same shape parse_hondadyn.py already reads. From v=2 every line
starts with what it was measured on, BUILD_KEYS: `commit` (GitCommit, 9 characters), `gaslaw` (v1/v2),
`cap` (1 = the launch cap is in the gas law), `pump` (v5 = the pump rule up to batch 2, c1b = rule C1b,
"Quiet pump at stops", CP_SP flag 64; v6 = the retired rule C1, flag 16, on routes of 2026-10-05/06),
`blaw` (v1 = the /2.6 brake law, v2 = the measured law, CP_SP flag 32) and `tuner` (Dynamic Tuning's live
parts on or off): shadow_learn_report.py never pools lines that differ in
any of them.
"""

import math
from collections import deque

from opendbc.car import ACCELERATION_DUE_TO_GRAVITY, structs
from opendbc.car.carlog import carlog

LongCtrlState = structs.CarControl.Actuators.LongControlState

LOG_TAG = "hondashadow"
LOG_VERSION = 2               # 2: the build tags (BUILD_KEYS) and the launch window from first wheel motion
RATE_HZ = 50                  # called once per 50 Hz gas/brake frame

# What a line was measured on, in the order the line carries them. "-" = not known.
BUILD_KEYS = ("commit", "gaslaw", "cap", "pump", "blaw", "tuner")
# CP_SP.flags bits (opendbc/sunnypilot/car/honda/values_ext.py HondaFlagsSP: ELESYS_BRAKE_LAW_V2, ELESYS_PUMP_C1B),
# with the fixed values as the fallback so a tag never depends on the import working. Flag 16 (the retired rule C1) is
# never set by this build, so no line it writes says v6.
BRAKE_LAW_V2_FLAG = 32
PUMP_C1B_FLAG = 64
LOG_INTERVAL = 60 * RATE_HZ   # a summary line at most once a minute

NIDEC_BRAKE_MAX = 256         # CarControllerParams.NIDEC_BRAKE_MAX: brake fraction -> 0x1FA counts

# The command that produced this sample's acceleration went out ~0.25-0.40 s earlier (A_learning:
# best pure delay aTarget -> aEgo 0.40 s on 115, 0.25 s on 10f). So a sample is admitted only when the
# pedal and brake commands have held their state for the whole of that window.
WINDOW = 20                   # 50 Hz samples = 0.4 s
# ...and only after this many consecutive clean frames (engaged, PID, no driver pedal, no stock AEB, D, fresh pose):
# the driver's throttle, or the frames before an engagement, must have left aEgo (docstring: 10f p90 0.76 s)
CLEAN_HOLD = 50               # 50 Hz samples = 1.0 s
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
LAUNCH_MIN_PEDAL = 0.01       # the interceptor must actually be pressing for the whole WINDOW...
LAUNCH_MIN_PCM_PEDAL = 1.0    # ...and the PCM must see it: 0x17C PEDAL_GAS (0-255) >= this over the WINDOW
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


def shadow_applicable(CP, CP_SP) -> bool:
  """Only the Elesys Accord with the gas interceptor: the bands, the window and the brake law they measure
  are this car's, and without the interceptor the pedal command reads 0 on every frame (all 'no pedal')."""
  try:
    from opendbc.car.honda.values import HONDA_ELESYS
    return CP is not None and CP.carFingerprint in HONDA_ELESYS and bool(getattr(CP_SP, "enableGasInterceptor", False))
  except Exception:
    return False


def pcm_pedal(CS) -> float:
  """0x17C POWERTRAIN_DATA PEDAL_GAS as CarStateExt records it: the pedal the PCM sees (with the interceptor,
  openpilot's command as the interceptor passes it on). nan when not recorded."""
  return _finite(getattr(CS, "pcm_pedal_gas", float("nan")))


def lead_visible(CC) -> bool:
  try:
    return bool(CC.hudControl.leadVisible)
  except Exception:
    return True   # not known: kept out of the no-lead sums


def _flag_value(name: str, fallback: int) -> int:
  try:
    from opendbc.sunnypilot.car.honda.values_ext import HondaFlagsSP
    return int(getattr(HondaFlagsSP, name, fallback))
  except Exception:
    return fallback


def pump_rule_tag(CP_SP) -> str:
  """'c1b' when the controller runs pump rule C1b (CP_SP flag ELESYS_PUMP_C1B), else 'v5'; '-' without CP_SP."""
  try:
    return "c1b" if int(CP_SP.flags) & _flag_value("ELESYS_PUMP_C1B", PUMP_C1B_FLAG) else "v5"
  except Exception:
    return "-"


def brake_law_tag(CP_SP) -> str:
  """'v2' when the controller runs the measured brake law (CP_SP flag ELESYS_BRAKE_LAW_V2), else 'v1'; '-' without CP_SP."""
  try:
    return "v2" if int(CP_SP.flags) & _flag_value("ELESYS_BRAKE_LAW_V2", BRAKE_LAW_V2_FLAG) else "v1"
  except Exception:
    return "-"


def launch_cap_tag(gas_law: str) -> str:
  """'1' when the gas law has the launch cap (elesys_gas.py: v2 below LAUNCH_CAP_V_END), '0' when it does not (v1, or
  another car's law), '-' while the law is not known yet (it arrives with the first interceptor frame)."""
  law = str(gas_law or "-")
  if law in ("-", ""):
    return "-"
  try:
    from opendbc.sunnypilot.car.honda.elesys_gas import LAUNCH_CAP_V_END
    return "1" if law == "v2" and LAUNCH_CAP_V_END > 0.0 else "0"
  except Exception:
    return "-"


def git_commit_tag(params) -> str:
  """The running build's commit (Params GitCommit, which manager writes at every start), 9 characters, or '-'."""
  try:
    raw = params.get("GitCommit") if params is not None else None
    if isinstance(raw, bytes):
      raw = raw.decode(errors="replace")
    txt = str(raw or "").strip()
    return txt[:9] if txt and all(c.isalnum() for c in txt[:9]) else "-"
  except Exception:
    return "-"


def build_tags(CP_SP, params=None, tuner_on: bool = False) -> dict:
  """Every BUILD_KEYS tag but the gas law and its cap, which the interceptor path reports frame by frame."""
  return {"commit": git_commit_tag(params), "gaslaw": "-", "cap": "-", "pump": pump_rule_tag(CP_SP),
          "blaw": brake_law_tag(CP_SP), "tuner": "1" if tuner_on else "0"}


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
  def __init__(self, build: dict | None = None):
    # what every line says it was measured on (BUILD_KEYS); the gas law and cap fill in from update()
    self.build = {k: "-" for k in BUILD_KEYS}
    self.build.update({k: str(v) for k, v in (build or {}).items() if k in BUILD_KEYS})
    n_speed, n_cmd = len(SPEED_BP), len(BRAKE_COUNT_BP) + 1
    self.brake = [[_Mean() for _ in range(n_cmd)] for _ in range(n_speed)]
    # the command and the achieved accel behind each brake cell, so the table also reads as "delivered
    # against the law" (A_learning's 0.5-0.8x at light commands) without a second pass over the route
    self.brake_counts = [[_Mean() for _ in range(n_cmd)] for _ in range(n_speed)]
    self.brake_acc = [[_Mean() for _ in range(n_cmd)] for _ in range(n_speed)]
    self.coast_err = [_Mean() for _ in range(n_speed)]
    self.coast_acc = [_Mean() for _ in range(n_speed)]
    self.brake_gain = _Mean()     # the live brake gain in force over the brake samples (docstring)
    # no lead: what the multiplier learns from; behind a lead: logged on its own, never in the multiplier
    self.launch = [_Ratio() for _ in range(len(LAUNCH_SPEED_BP) - 1)]
    self.launch_lead = [_Ratio() for _ in range(len(LAUNCH_SPEED_BP) - 1)]
    self.launch_episodes = 0
    self.launch_episodes_lead = 0
    self._in_launch = False

    self._counts: deque = deque(maxlen=WINDOW)
    self._gas: deque = deque(maxlen=WINDOW)
    self._pcm_pedal: deque = deque(maxlen=WINDOW)
    self._prev_ref = 0.0
    self._steady = 0
    self._ramp_ok = 0
    self._clean = 0
    self._clean_launch = 0   # the same run of clean frames, in any control state (the launch's window, docstring)
    self._long_active = False
    self._frames = 0
    self._dirty = False
    self.lines = 0

  # --- per 50 Hz frame -------------------------------------------------------------------------------

  def update(self, CC, CS, *, pitch: float, pose_fresh: bool, mode_ok: bool,
             cmd_ref: float, brake_frac: float, gas_cmd: float, brake_gain: float = 1.0, gas_law: str = "") -> None:
    """One 50 Hz sample. Every keyword argument is a value the tuner already computed for this frame, CC
    and CS are only read; nothing is returned and nothing outside this object is written."""
    if gas_law:
      self.build["gaslaw"] = str(gas_law)
      self.build["cap"] = launch_cap_tag(gas_law)
    long_active = bool(CC.longActive)
    out = CS.out
    counts = _finite(brake_frac, 0.0) * NIDEC_BRAKE_MAX
    gas = _finite(gas_cmd, 0.0)
    ref = _finite(cmd_ref, 0.0)
    self._counts.append(counts)
    self._gas.append(gas)
    self._pcm_pedal.append(pcm_pedal(CS))
    ramp = abs(ref - self._prev_ref) * RATE_HZ
    self._prev_ref = ref
    self._steady = self._steady + 1 if ramp <= STEADY_MAX_RAMP else 0
    self._ramp_ok = self._ramp_ok + 1 if ramp <= LAUNCH_MAX_RAMP else 0
    clean_any_state = (long_active and mode_ok and pose_fresh
                       and not out.gasPressed and not out.brakePressed and not out.stockAeb)
    clean = clean_any_state and CC.actuators.longControlState == LongCtrlState.pid
    self._clean = self._clean + 1 if clean else 0
    self._clean_launch = self._clean_launch + 1 if clean_any_state else 0

    # the drive's running totals go out at every disengage and once a minute, if anything was added
    disengaged = self._long_active and not long_active
    self._long_active = long_active
    self._frames += 1
    if self._dirty and (disengaged or self._frames % LOG_INTERVAL == 0):
      self.emit()

    launched = self._sample(CC, CS, pitch, ref, counts, brake_gain)
    if launched and not self._in_launch:
      if lead_visible(CC):
        self.launch_episodes_lead += 1
      else:
        self.launch_episodes += 1
    self._in_launch = launched

  def _sample(self, CC, CS, pitch, ref, counts, brake_gain) -> bool:
    """Admit at most one sample into one table. Returns True if it was a launch sample."""
    out = CS.out
    # the brake and coast tables need CLEAN_HOLD of PID; a launch the same run in any state (docstring)
    if self._clean_launch < CLEAN_HOLD or len(self._counts) < WINDOW:
      return False
    pid_clean = self._clean >= CLEAN_HOLD
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
      if not pid_clean or sb < 0 or self._steady < STEADY_HOLD or not abs(pitch) < MAX_PITCH:
        return False
      if cb_lo >= BRAKE_MIN_COUNTS and cb_hi - cb_lo <= BRAKE_STEADY_COUNTS and count_band(cb_lo) == count_band(cb_hi):
        cb = count_band(counts)
        self.brake[sb][cb].add(err)
        self.brake_counts[sb][cb].add(counts)
        self.brake_acc[sb][cb].add(achieved)
        self.brake_gain.add(_finite(brake_gain, 1.0))
        self._dirty = True
      elif cb_hi <= 0.0 and v >= COAST_MIN_SPEED:
        self.coast_err[sb].add(err)
        self.coast_acc[sb].add(achieved)
        self._dirty = True
      return False

    lb = launch_band(v)
    # the PCM saw the pedal over the whole window (nan - not recorded - counts as not seen)
    pcm_ok = all(p >= LAUNCH_MIN_PCM_PEDAL for p in self._pcm_pedal)
    if (lb >= 0 and abs(pitch) < LAUNCH_MAX_PITCH and cb_hi <= 0.0 and gas_lo >= LAUNCH_MIN_PEDAL and pcm_ok
            and ref >= LAUNCH_MIN_REQ and self._ramp_ok >= LAUNCH_RAMP_HOLD):
      (self.launch_lead if lead_visible(CC) else self.launch)[lb].add(ref, achieved)
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
    # tags carry no spaces or '=' (a value that did would split the line's tokens)
    tags = " ".join(f"{k}={str(self.build.get(k, '-')).replace(' ', '_').replace('=', '_') or '-'}" for k in BUILD_KEYS)
    return (f"{LOG_TAG} v={LOG_VERSION} {tags} " +
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
            f"lmult={launch_multiplier(n, ra, rr):.3f} " +
            f"lnl={_fmt([r.n for r in self.launch_lead], 'd')} lral={_fmt([r.ra for r in self.launch_lead], '.4g')} " +
            f"lrrl={_fmt([r.rr for r in self.launch_lead], '.4g')} " +
            f"lratiol={_fmt([r.ratio() for r in self.launch_lead], '.3f')} lepl={self.launch_episodes_lead} " +
            f"bgain={_fmt([self.brake_gain.mean()], '.3f')}")

  def emit(self) -> None:
    self._dirty = False
    self.lines += 1
    carlog.info(self.line())

  def flush(self) -> None:
    """The drive's last totals, if anything was admitted since the last line: called once when card exits
    (HondaDynamicTuner.flush_at_exit), so a drive that ends engaged still ends on its total."""
    if self._dirty:
      self.emit()
