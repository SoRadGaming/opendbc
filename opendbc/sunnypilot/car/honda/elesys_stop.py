"""
FORK(HONDA_ACCORD_9G_AU): the soft final stop on this car (2013-15 Accord AU, HONDA_ELESYS).

CarController (opendbc/car/honda/carcontroller.py) builds ElesysSoftStop only on HONDA_ELESYS
with HondaDynamicTuningEnabled on -- the same gate as the stopping debounce and the 32-count
release limiter -- and calls it in the Nidec brake block after the learned brake gain and before
the release limiter. Every other car, and this car with the toggle off, never builds it, so their
BRAKE_COMMAND is bit-identical.

WHY (braking audit and its skeptic review, 45 stops openpilot completed with no driver input,
routes 3e..103): the harsh final jerk is the stopping ramp reaching the full standstill-hold brake
before the car has stopped. Stopping state is entered a median 0.90 s before the stop, at
0.62 m/s, with a near-zero command. longcontrol then ramps toward stopAccel -0.8 at 0.8 m/s^3 and
the creep table adds its own +35 counts as the speed falls, so the brake is already at the hold
level when the wheels stop (median 185 counts, hold 189) while the planner asks for only -0.16 at
the stop. Deceleration felt at the stop: median 0.92 m/s^2 (lead stops 1.02, stops without a lead
0.48), jerk on settling a median 7.4 m/s^3.

WHAT: a brake CEILING while the car is still rolling in the stopping state.

  output       min(apply_brake, ceiling) -- it can only LOWER the command, never raise it
  at entry     ceiling = max(cap, apply_brake at entry): never below what was being commanded
               when stopping began
  cap          125 + max(0, -sin(pitch)) * 9.81 / 2.6 * 256 counts, ~17 per degree of downhill;
               pitch is the dynamic tuner's filtered pitch, and no grade term when it has none
  rolling      ceiling = max(ceiling, cap): a steeper downhill raises it, nothing lowers it
  RISES        at 250 counts/s (125 -> 189 in ~0.26 s), monotone, from the first of
                 SETTLE   0.55 s after the WHEELS first read zero (vEgoRaw == 0 / standstill)
                 MOVING   the wheels turn again after reading zero
                 WEAK     aEgo > -0.25 m/s^2 for 0.4 s, counted only once the command has sat AT
                          the ceiling for 0.3 s
                 MAX_ROLL 1.9 s since entry, absolute: rolling or settling, whatever the wheels do
  done         at 255 counts the ceiling is out of the way, so the standstill hold is today's
               (189), byte for byte
  no ceiling   stopping entered with the wheels already at zero, or faster than 1.2 m/s (vEgo;
               unknown counts as faster); outside the stopping state; longActive false; gas or
               brake pressed. The last three reset the state.

The bound: the rise starts 1.9 s after entry at the latest and the ceiling is gone 0.52 s after
that, so nothing of it is left 2.42 s after the stopping state began. On the replayed stops every
rise was SETTLE, at most 1.82 s after entry.

The ENTRY SPEED bound (review, 2026-10): the weak-decel escape cannot fire while the car is
decelerating harder than 0.25 m/s^2, so a stop entered fast with the brake still building would
sit at the cap until MAX_ROLL. Measured entries reach 1.08 m/s (median 0.58); above 1.2 m/s the
ceiling stays off and the stop is today's.

Leaving the stopping state for the PID drops the ceiling in one frame: the PID's command passes
untouched (on a stopping<->pid flicker the largest upward step on that frame was 6 counts in the
replay). Carrying a ceiling into the PID state would cap braking the planner asked for, so it does not.

The four changes the skeptic review made to the audit's version, and why:

  1. SETTLE STARTS WHEN THE WHEELS READ ZERO, AND LASTS 0.55 s. vEgoRaw (XMISSION_SPEED below
     1 m/s) drops from ~0.30 m/s straight to 0 while the camera still sees 0.17 m/s; at today's
     0.9 m/s^2 the car stops 0.14 s later, at this ceiling's softer ~0.6 m/s^2 the last 0.3 m/s
     takes 0.3-0.5 s. vEgo < 0.15 (the audit's trigger) fires 0.05 s before the wheels read zero
     and a 0.25 s settle put the brake back near 175-189 by the real stop.
  2. WEAK DECELERATION ONLY COUNTS AT THE CEILING. On today's traces the plain 0.4 s escape fired
     in 6 of 45 stops, every time 0.38 s after entry, in stops that entered with 9-53 counts and
     were still building pressure -- including three of the harshest (ad 394.3 2.74 m/s^2,
     b0 391.0 1.60, d3 829.4 1.41). It would have handed exactly those back to today's ramp.
  3. THE CAP IS GRADE-AWARE. Below 2 m/s and in the stopping state the pitch feedforward is 0
     (dynamic_tuning._pitch_feedforward), so a flat 125 on a 2.5 deg downhill leaves ~0.1 m/s^2
     of net deceleration. -2.5 deg adds 42 counts.
  4. MOVING AGAIN AFTER WHEEL-ZERO RISES AT ONCE, and MAX_ROLL is 1.9 s (the audit's 2.5 s was
     wider than the ~1.4 s a 0.8 m/s stop takes at ~0.6 m/s^2; 4.1 m median to a stopped lead).

Expected (inferred; the audit's simulator was never validated, so measure it): decel at the stop
~0.5-0.6 m/s^2 instead of 0.92, about +0.1-0.2 m and +0.3-0.4 s per stop. Measure decel at the
camera/IMU-confirmed stop and the peak jerk in the 0-0.8 s after wheel-zero; the brake count at
wheel-zero is 125 by construction, so it proves nothing.

The pump is unaffected in kind: 125 > 100, so brake_pump_hysteresis_elesys()'s continuous
final-approach branch still runs. The learners are unaffected: the brake learner needs the PID
state and vEgo > 1 m/s, the per-mode pedal counter needs the PID state (_learn_ok), and both read
the command BEFORE this ceiling. Panda forwards stock AEB whenever its brake is >= ours, so a
lower command can only start that forwarding earlier, never later.

One line per stop goes to carlog (`hondastop ...`, which reaches the route as a logMessage): why
and when the ceiling started to rise, that the stop ended before it did, or (`skip=speed v=`)
that it was entered faster than 1.2 m/s and got no ceiling.
"""
import math
from dataclasses import dataclass

from opendbc.car import DT_CTRL, structs
from opendbc.car.carlog import carlog

LongCtrlState = structs.CarControl.Actuators.LongControlState

SOFT_STOP_DT = 2 * DT_CTRL       # s; the Nidec brake block runs on every other 100 Hz frame
SOFT_STOP_ROLL_CB = 125.         # counts: the ceiling while rolling on the flat
SOFT_STOP_GRADE_CB = 9.81 / 2.6 * 256   # counts per unit of -sin(pitch): full brake is 2.6 m/s^2 at 256
SOFT_STOP_SETTLE = 0.55          # s the ceiling holds after the wheels first read zero
SOFT_STOP_RISE = 250.            # counts/s once rising: 5 counts per 50 Hz frame
SOFT_STOP_AT_CEILING = 0.3       # s the command must sit at the ceiling before weak decel counts
SOFT_STOP_WEAK_DECEL = 0.25      # m/s^2: aEgo above -this is "not slowing"
SOFT_STOP_WEAK_TIME = 0.4        # s of not slowing, at the ceiling, before the rise
SOFT_STOP_MAX_ROLL = 1.9         # s since entry, absolute: the rise starts by then whatever the wheels do
SOFT_STOP_MAX_ENTRY_V = 1.2      # m/s (vEgo): stopping entered faster gets no ceiling; measured entries max 1.08
SOFT_STOP_DONE_CB = 255.         # the highest command the brake block can send: the ceiling is gone
WHEELS_ZERO_SPEED = 1e-3         # m/s; vEgoRaw at or below this reads as the wheels stopped
LOG_TAG = "hondastop"

_EPS = 1e-6                      # timers accumulate SOFT_STOP_DT; compare with a little slack


def _finite(x, fallback: float = 0.0) -> float:
  try:
    v = float(x)
  except (TypeError, ValueError):
    return fallback
  return v if math.isfinite(v) else fallback


@dataclass
class SoftStopState:
  ceiling: float
  armed: bool                    # False: entered at standstill or too fast, or the ceiling has risen out of the way
  entry_brake: int = 0
  entry_cap: float = SOFT_STOP_ROLL_CB
  entry_v: float = math.nan      # vEgo on the entry frame
  skip: str = ""                 # "speed": entered faster than SOFT_STOP_MAX_ENTRY_V (or at an unknown speed)
  wheels_zero_seen: bool = False
  rising: bool = False
  reason: str = ""               # what started the rise: settle / moving / weak / max_roll
  t: float = 0.                  # s since entry
  roll_t: float = 0.             # s since entry with the wheels turning
  still_t: float = 0.            # s with the wheels at zero, since they first read zero
  at_ceiling_t: float = 0.       # s the command has been at (or above) the ceiling, continuously
  weak_t: float = 0.             # s of weak deceleration counted at the ceiling, continuously


def soft_stop_cap(pitch) -> float:
  """The ceiling while rolling, in counts: 125 on the flat, plus ~17 per degree of downhill.
  No grade term without a usable pitch (None, NaN, or not a pitch at all)."""
  p = _finite(pitch, float("nan"))
  if not (math.isfinite(p) and abs(p) < math.pi / 2):
    return SOFT_STOP_ROLL_CB
  return SOFT_STOP_ROLL_CB + max(0., -math.sin(p)) * SOFT_STOP_GRADE_CB


def soft_stop_ceiling(stopping: bool, wheels_zero: bool, v_ego: float, a_ego: float, pitch, apply_brake: int,
                      st: SoftStopState | None, dt: float = SOFT_STOP_DT) -> tuple[int, SoftStopState | None]:
  """One 50 Hz brake frame. Returns (the command to send, the state to pass back next frame).

  `stopping` is the caller's whole gate: longActive, the stopping state, and no gas or brake
  pedal. The output is never above `apply_brake`; the state is None whenever there is no stop.
  `v_ego` matters on the entry frame only.
  """
  if not stopping:
    return apply_brake, None

  cmd = _finite(apply_brake)
  cap = soft_stop_cap(pitch)
  if st is None:
    # entered with the wheels already at zero: nothing to soften, the hold is today's. Entered faster
    # than any measured stop (or at an unknown speed): today's ramp, not a ceiling nothing has tested.
    v0 = _finite(v_ego, math.nan)
    too_fast = not (v0 <= SOFT_STOP_MAX_ENTRY_V)
    st = SoftStopState(ceiling=max(cap, cmd), armed=not wheels_zero and not too_fast, entry_brake=int(cmd),
                       entry_cap=cap, entry_v=v0, skip="speed" if (too_fast and not wheels_zero) else "")
  if not st.armed:
    return apply_brake, st

  st.t += dt
  st.ceiling = max(st.ceiling, cap)       # a steeper downhill raises it; nothing lowers it

  if wheels_zero:
    st.wheels_zero_seen = True
    st.still_t += dt
    st.at_ceiling_t = st.weak_t = 0.
    if st.still_t >= SOFT_STOP_SETTLE - _EPS:
      _rise(st, "settle")
  else:
    if st.wheels_zero_seen:
      _rise(st, "moving")                 # the wheels read zero and are turning again
    st.roll_t += dt
    at_ceiling = cmd >= int(st.ceiling)
    # weak deceleration only means something once the brake has had 0.3 s AT the ceiling to build
    # pressure: before that the car is slow to decelerate because the brake is still coming on
    settled = at_ceiling and st.at_ceiling_t >= SOFT_STOP_AT_CEILING - _EPS
    st.at_ceiling_t = st.at_ceiling_t + dt if at_ceiling else 0.
    # an unknown aEgo counts as not slowing: the fallback is today's ramp, not a longer cap
    weak = settled and not (_finite(a_ego, math.nan) <= -SOFT_STOP_WEAK_DECEL)
    st.weak_t = st.weak_t + dt if weak else 0.
    if st.weak_t >= SOFT_STOP_WEAK_TIME - _EPS:
      _rise(st, "weak")
  if st.t >= SOFT_STOP_MAX_ROLL - _EPS:
    _rise(st, "max_roll")                 # absolute: counted from entry, rolling or settling

  if st.rising:
    st.ceiling += SOFT_STOP_RISE * dt     # monotone: it never falls back during this stop
  if st.ceiling >= SOFT_STOP_DONE_CB:
    st.armed = False                      # out of the way; the hold is today's
    return apply_brake, st
  return min(apply_brake, int(st.ceiling)), st


def _rise(st: SoftStopState, reason: str) -> None:
  if not st.rising:
    st.rising = True
    st.reason = reason


def wheels_read_zero(CS) -> bool:
  """True when the wheels read zero: CarState.standstill (XMISSION_SPEED == 0) or vEgoRaw at zero.
  Below 1 m/s vEgoRaw IS the transmission speed, which reads 0 below ~0.3 m/s. A missing or
  non-finite reading counts as zero, which falls back to today's behavior (no ceiling at entry,
  the settle timer running after it)."""
  out = CS.out
  if bool(getattr(out, "standstill", False)):
    return True
  return not (_finite(getattr(out, "vEgoRaw", math.nan), math.nan) > WHEELS_ZERO_SPEED)


class ElesysSoftStop:
  """CarController's handle on soft_stop_ceiling(): reads its inputs off CC, CS and the tuner's
  filtered pitch, keeps the state, logs one line per stop, and never raises -- an exception in
  CarController.update() means no 0x1FA, and the VSA latches BRAKE_ERROR about 1 s later."""

  def __init__(self):
    self.state: SoftStopState | None = None

  def update(self, CC, CS, apply_brake: int, tuner=None) -> int:
    try:
      stopping = bool(CC.longActive and CC.actuators.longControlState == LongCtrlState.stopping
                      and not CS.out.gasPressed and not CS.out.brakePressed)
      pitch = None
      if stopping and tuner is not None:
        try:
          pitch = tuner.filtered_pitch()
        except Exception:
          pitch = None                    # no grade term; the flat cap still applies
      prev = self.state
      prev_rising = prev is not None and prev.rising
      # a missing vEgo reads as unknown, which the entry-speed bound treats as too fast: no ceiling
      v_ego = getattr(CS.out, "vEgo", math.nan)
      out, self.state = soft_stop_ceiling(stopping, wheels_read_zero(CS), v_ego, CS.out.aEgo, pitch, apply_brake, self.state)
      self._log(prev, prev_rising, self.state)
      out = int(out)
      # belt and braces: whatever happened above, the ceiling can only ever lower the command
      return out if 0 <= out <= apply_brake else apply_brake
    except Exception:
      self.state = None
      return apply_brake

  @staticmethod
  def _log(prev: SoftStopState | None, prev_rising: bool, st: SoftStopState | None) -> None:
    try:
      if st is not None and st.rising and not prev_rising:
        carlog.info(f"{LOG_TAG} rise={st.reason} t={st.t:.2f} roll={st.roll_t:.2f} still={st.still_t:.2f} " +
                    f"entry={st.entry_brake} cap={st.entry_cap:.0f} ceil={st.ceiling:.0f}")
      elif st is None and prev is not None and prev.armed and not prev_rising:
        carlog.info(f"{LOG_TAG} end=left t={prev.t:.2f} roll={prev.roll_t:.2f} still={prev.still_t:.2f} " +
                    f"entry={prev.entry_brake} cap={prev.entry_cap:.0f} ceil={prev.ceiling:.0f}")
      elif prev is None and st is not None and st.skip:
        carlog.info(f"{LOG_TAG} skip={st.skip} v={st.entry_v:.2f} entry={st.entry_brake}")
    except Exception:
      pass
