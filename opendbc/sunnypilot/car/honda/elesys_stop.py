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
  cap          105 + max(0, -sin(pitch)) * 9.81 / 2.6 * 256 counts, ~17 per degree of downhill;
               pitch is the dynamic tuner's filtered pitch, and no grade term when it has none
  rolling      ceiling = max(ceiling, cap): a steeper downhill raises it, nothing lowers it
  RISES        at 250 counts/s (105 -> 189 in ~0.34 s), monotone, from the first of
                 SETTLE   0.55 s after the WHEELS first read zero (vEgoRaw == 0 / standstill)
                 MOVING   the wheels turn again after reading zero
                 WEAK     aEgo > -0.25 m/s^2 for 0.4 s, counted only once the command has sat AT
                          the ceiling for 0.3 s
                 MAX_ROLL since entry, absolute: rolling or settling, whatever the wheels do. 1.9 s up to a
                          0.6 m/s entry, +2.5 s per m/s above it (round 10): 2.4 s at 0.8, 3.4 s at most (1.2)
  done         at the highest command the brake block can send (255; 320 with brake320, 255 again
               after its VSA fallback) the ceiling is out of the way, so the standstill hold is
               today's (189), byte for byte
  no ceiling   stopping entered with the wheels already at zero, or faster than 1.2 m/s (vEgo;
               unknown counts as faster); outside the stopping state; longActive false; gas or
               brake pressed. The last three reset the state.

The bound: the rise starts at MAX_ROLL after entry at the latest - 1.9 s up to a 0.6 m/s entry, at
most 3.4 s at the 1.2 m/s entry bound (round 10) - and the ceiling is gone at most 0.60 s after that
(105 -> 255 at 250 counts/s; a firmed-up ceiling starts higher and is gone sooner), so nothing of it
is left 2.50 s after the stopping state began for an entry up to 0.6 m/s, 3.0 s at 0.8 m/s and
4.0 s at 1.2 m/s (measured: still lowering the command 3.70 s after a 1.2 m/s entry at aEgo -0.5).
With brake320 (brake_ceiling.py) the ceiling rises on to 320, 0.86 s, so 2.76 s / 4.26 s after entry; between 255
and 320 it can only bind on a command above 255, which the brake block cannot send in the stopping
state: only the pid state earns beyond 255 (review 2026-10-24, which found stops entered from a
hard brake holding LongControl's -2.6 to -3.0 at 267-317 counts for the whole standstill before
that gate). A stop's hold is 189 from stopAccel, or at most 255 after a hard brake, as before.
The release limiter, after this ceiling, still walks a higher approach command down 32 a send.
On the replayed stops every rise was SETTLE, at most 1.82 s after entry.

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
     (dynamic_tuning._pitch_feedforward), so a flat 125 (the cap then) on a 2.5 deg downhill leaves ~0.1 m/s^2
     of net deceleration. -2.5 deg adds 42 counts.
  4. MOVING AGAIN AFTER WHEEL-ZERO RISES AT ONCE, and MAX_ROLL is 1.9 s (the audit's 2.5 s was
     wider than the ~1.4 s a 0.8 m/s stop takes at ~0.6 m/s^2; 4.1 m median to a stopped lead).
     Round 10 (below) keeps 1.9 s up to a 0.6 m/s entry and lets it grow above that.

Expected (inferred; the audit's simulator was never validated, so measure it): decel at the stop
~0.5-0.6 m/s^2 instead of 0.92, about +0.1-0.2 m and +0.3-0.4 s per stop. Measure decel at the
camera/IMU-confirmed stop and the peak jerk in the 0-0.8 s after wheel-zero; the brake count at
wheel-zero is the cap by construction, so it proves nothing.

THE CAP WAS 125 until round 8 (2026-10-20, proposal 5). Measured on the 10 rlog stops of drives
13d-142 (17-18 Oct, push9): the ceiling did what it was built for (9 of 10 hondastop lines
rise=settle, the cap at wheel-zero), but the car still met the stop at a median -0.74 m/s^2
(IMU -0.86) with a median 4.6 m/s^3 snap on the IMU, against this file's own target of <= 0.6.
While rolling at the cap the IMU read a median 0.82 m/s^2, so the cap itself set that decel.
At ~0.012 m/s^2 per count (the batch-3 plant's brake gain at a crawl), 105 takes ~0.24 off it:
estimated ~0.58 at the cap, the roll 0.80 -> 0.99 s (median), 0.1-0.2 m longer (inferred). The
closed-loop replay of the same stops through the real controller agrees in direction (true decel
at the stop 0.08-0.22 lower where the cap binds; median 0.70 -> 0.58) but the snap barely moves:
the jerk after wheel-zero is a median 4.78 -> 4.64 m/s^3, about 3% smaller, not the ~25% the
analysis hoped for. And it shows the cost: a stop that already rolled long at 125 (13d t380.8:
1.74 s, only 0.35 m/s^2 at the cap) has too little decel at 105, its escape changes from settle to
weak, and the old ramp it is handed to makes it marginally harsher (4.49 -> 4.54 m/s^3) - one of
the ten. 141 t234.7 escapes by max_roll at either cap and comes out softer (4.56 -> 4.35). Those
escapes are the designed fallback; the hondastop line says which one fired.

The pump is unaffected: the cap stays above C1b's and v5's crawl threshold (cb > 100), so for any
command c, min(c, cap) > 100 exactly when c > 100 - the crawl continuous run (0.15 <= v < 2.5 m/s)
pumps the same frames as with 125, and the rise to the hold is still more than ELESYS_PUMP_BIG_RISE
over what was delivered. That is why the cap must stay above 100: brake_pump_hysteresis_elesys()'s
and brake_pump_c1b_elesys()'s continuous final-approach branch still runs. The learners are unaffected: the brake learner needs the PID
state and vEgo > 1 m/s, the per-mode pedal counter needs the PID state (_learn_ok), and both read
the command BEFORE this ceiling. Panda forwards stock AEB whenever its brake is >= ours, so a
lower command can only start that forwarding earlier, never later.

ROUND 10 (2026-11, the red-light stops): the planner now asks to stop at 0.8 m/s at a red light too (openpilot
e2e_stop.py), so a stop that used to enter here at 0.28 m/s, 0.01-0.08 s before wheel-zero, enters at ~0.78 m/s and
1.5-2 s of it is rolling at this ceiling. Two changes for that, both measured on the four hands-off red-light stops
(12b 470.7, 142 263.7, 161 147.6, 170 233.0) and the 50 clean queue stops, closed loop through this file:

  FIRM-UP      while the wheels turn, once the command has sat AT the ceiling for 0.3 s, aEgo above -0.5 m/s^2 raises
               the ceiling by 150 counts/s per m/s^2 of shortfall (at most 75/s): it only ever rises, never past the
               command, and freezes at wheel-zero (aEgo means nothing there). It replaces the guess that 105 counts is
               0.5-0.6 m/s^2 on every road: on 142 105 gave ~0.43, on 161's -2.2 deg 162 counts gave 0.31 (measured).
  MAX_ROLL     1.9 s up to 0.6 m/s at entry, +2.5 s per m/s above it (2.4 s at 0.8 m/s). At 1.9 s the backstop fired
               in 17 of 24 replays of a 0.8 m/s red-light entry, inside the last 0.3 m/s; at 2.4 s, none.

Replayed (brake gain 1.5-2.9 m/s^2 per 256 counts, lag 0.3-0.5 s): decel at wheel-zero 0.36-0.59 m/s^2 (median 0.49),
at rest at most 0.64; on the queue stops the change moves the rest point a median 0.00 m (at most 0.22 m shorter) and
the decel at wheel-zero not at all (median 0.57 both), and MAX_ROLL never fires (48 of 300 replays today).

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
SOFT_STOP_ROLL_CB = 105.         # counts: the ceiling while rolling on the flat (125 until round 8); keep > 100
SOFT_STOP_GRADE_CB = 9.81 / 2.6 * 256   # counts per unit of -sin(pitch): the brake law's 2.6 m/s^2 per 256 counts
SOFT_STOP_SETTLE = 0.55          # s the ceiling holds after the wheels first read zero
SOFT_STOP_RISE = 250.            # counts/s once rising: 5 counts per 50 Hz frame
SOFT_STOP_AT_CEILING = 0.3       # s the command must sit at the ceiling before weak decel counts
SOFT_STOP_WEAK_DECEL = 0.25      # m/s^2: aEgo above -this is "not slowing"
SOFT_STOP_WEAK_TIME = 0.4        # s of not slowing, at the ceiling, before the rise
SOFT_STOP_MAX_ROLL = 1.9         # s since entry, absolute: the rise starts by then whatever the wheels do
SOFT_STOP_MAX_ENTRY_V = 1.2      # m/s (vEgo): stopping entered faster gets no ceiling; measured entries max 1.08
# FIRM-UP (round 10): while rolling AT the ceiling, a car slowing less than SOFT_STOP_FIRM_DECEL lets the ceiling climb in
# proportion to the shortfall, so the brake firms toward the stop instead of holding a count that is too light for this
# road (a downhill, a weak plant at a crawl). Only up, only while the wheels turn, only once the command has sat at the
# ceiling for SOFT_STOP_AT_CEILING (the brake's lag), never after the rise has started. 0.5 is round 8's softer stop.
SOFT_STOP_FIRM_DECEL = 0.5       # m/s^2: the deceleration the ceiling holds the car to while rolling, at least
SOFT_STOP_FIRM_GAIN = 150.       # counts/s per m/s^2 of shortfall: a 0.6-1.1 s time constant at 1.5-2.9 m/s^2 per 256 counts
SOFT_STOP_FIRM_MAX_ERR = 0.5     # m/s^2: the most shortfall counted, so the firm-up is at most 75 counts/s (the rise is 250)
# MAX_ROLL grows with the entry speed: a red-light stop enters at ~0.8 m/s (the planner's e2e stop entry) and at 0.5 m/s^2
# needs ~1.6 s plus the ramp-in; at 1.9 s the backstop would fire inside the last 0.3 m/s. 1.9 s up to 0.6 m/s, as before.
SOFT_STOP_MAX_ROLL_FROM_V = 0.6  # m/s
SOFT_STOP_MAX_ROLL_PER_MS = 2.5  # s per m/s of entry speed above SOFT_STOP_MAX_ROLL_FROM_V (0.8 m/s: 2.4 s; 1.2: 3.4 s)
# the highest command the brake block can send: the ceiling is gone. The default; brake320's CarController passes its
# own (320, or 255 after the VSA fallback) as done_cb
SOFT_STOP_DONE_CB = 255.
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
  firm_cb: float = 0.            # counts the firm-up has added to the ceiling
  max_roll: float = SOFT_STOP_MAX_ROLL   # this stop's MAX_ROLL (soft_stop_max_roll of the entry speed)


def soft_stop_cap(pitch) -> float:
  """The ceiling while rolling, in counts: 105 on the flat, plus ~17 per degree of downhill.
  No grade term without a usable pitch (None, NaN, or not a pitch at all)."""
  p = _finite(pitch, float("nan"))
  if not (math.isfinite(p) and abs(p) < math.pi / 2):
    return SOFT_STOP_ROLL_CB
  return SOFT_STOP_ROLL_CB + max(0., -math.sin(p)) * SOFT_STOP_GRADE_CB


def soft_stop_max_roll(v_entry) -> float:
  """MAX_ROLL for a stop entered at v_entry (m/s): 1.9 s up to 0.6 m/s, +2.5 s per m/s above it."""
  v = _finite(v_entry, 0.0)
  return SOFT_STOP_MAX_ROLL + SOFT_STOP_MAX_ROLL_PER_MS * max(min(v, SOFT_STOP_MAX_ENTRY_V) - SOFT_STOP_MAX_ROLL_FROM_V, 0.)


def soft_stop_ceiling(stopping: bool, wheels_zero: bool, v_ego: float, a_ego: float, pitch, apply_brake: int,
                      st: SoftStopState | None, dt: float = SOFT_STOP_DT,
                      done_cb: float = SOFT_STOP_DONE_CB) -> tuple[int, SoftStopState | None]:
  """One 50 Hz brake frame. Returns (the command to send, the state to pass back next frame).

  `stopping` is the caller's whole gate: longActive, the stopping state, and no gas or brake
  pedal. The output is never above `apply_brake`; the state is None whenever there is no stop.
  `v_ego` matters on the entry frame only. `done_cb` is the highest command the brake block can send.
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
                       entry_cap=cap, entry_v=v0, skip="speed" if (too_fast and not wheels_zero) else "",
                       max_roll=soft_stop_max_roll(v0))
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
    if settled and not st.rising:
      # FIRM-UP: slowing less than SOFT_STOP_FIRM_DECEL at the ceiling raises it in proportion; unknown aEgo is the most
      a = _finite(a_ego, math.nan)
      short = SOFT_STOP_FIRM_MAX_ERR if not math.isfinite(a) else min(max(a + SOFT_STOP_FIRM_DECEL, 0.), SOFT_STOP_FIRM_MAX_ERR)
      st.ceiling += SOFT_STOP_FIRM_GAIN * short * dt
      st.firm_cb += SOFT_STOP_FIRM_GAIN * short * dt
    # an unknown aEgo counts as not slowing: the fallback is today's ramp, not a longer cap
    weak = settled and not (_finite(a_ego, math.nan) <= -SOFT_STOP_WEAK_DECEL)
    st.weak_t = st.weak_t + dt if weak else 0.
    if st.weak_t >= SOFT_STOP_WEAK_TIME - _EPS:
      _rise(st, "weak")
  if st.t >= st.max_roll - _EPS:
    _rise(st, "max_roll")                 # absolute: counted from entry, rolling or settling

  if st.rising:
    st.ceiling += SOFT_STOP_RISE * dt     # monotone: it never falls back during this stop
  if st.ceiling >= done_cb:
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

  def update(self, CC, CS, apply_brake: int, tuner=None, done_cb: float = SOFT_STOP_DONE_CB) -> int:
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
      out, self.state = soft_stop_ceiling(stopping, wheels_read_zero(CS), v_ego, CS.out.aEgo, pitch, apply_brake, self.state,
                                          done_cb=done_cb)
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
                    f"entry={st.entry_brake} cap={st.entry_cap:.0f} ceil={st.ceiling:.0f} firm={st.firm_cb:.0f} v={st.entry_v:.2f}")
      elif st is None and prev is not None and prev.armed and not prev_rising:
        carlog.info(f"{LOG_TAG} end=left t={prev.t:.2f} roll={prev.roll_t:.2f} still={prev.still_t:.2f} " +
                    f"entry={prev.entry_brake} cap={prev.entry_cap:.0f} ceil={prev.ceiling:.0f}")
      elif prev is None and st is not None and st.skip:
        carlog.info(f"{LOG_TAG} skip={st.skip} v={st.entry_v:.2f} entry={st.entry_brake}")
    except Exception:
      pass
