"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Dynamic (self-learning) longitudinal tuning for Honda Nidec.

Ported from MVL's ACURA_MDX_3G dynamic branch, restructured for this fork:

  * The MDX 3G drives gas purely through the Nidec PCM, so MVL has a single set
    of gas learners that all learn against pcm_accel. This car has a comma pedal
    interceptor, which owns the gas at every speed, so none of that is ported.
    An earlier revision crossfaded part of the request back to the PCM above
    ~30 km/h; that is gone. See the note in carcontroller.py where the request
    is built.

  * THERE IS NO GAS LEARNER ANY MORE (2026-10). The per-band pedal gain and the
    aero ("wind") scale are retired -- see "Retired channels" below for why --
    and the gas side is the measured, static law in elesys_gas.py. What is left
    on the gas side here is bookkeeping: time and steady-pedal samples per drive
    mode, so per-mode pedal tables can be fitted offline from routes.

  * MVL's per-5mph lateral latFactors are deliberately NOT ported. Longitudinal
    only, by request.

  * MVL reads/writes Params directly at the top of opendbc/car/honda/
    carcontroller.py. opendbc has to stay importable without openpilot on the PC
    side, so the import is lazy and every failure path degrades to
    in-memory-only learning.

Design rules that differ from MVL's original, all of them the result of failure
modes found while reviewing the port. Do not relax these without re-deriving why
they are here:

  1. NOTHING LEARNS DURING TRANSIENTS. Every learner is gated behind a dwell
     timer. Brake hydraulics take ~0.5 s to build pressure; without the gate the
     integrator eats pure actuator lag and rails within a few seconds of normal
     driving.

  2. ONLY SETTLED, OFF-RAIL VALUES ARE PERSISTED. Each learner keeps a long-EMA
     estimate that advances only while the live value is off its clamps. A
     railed excursion therefore never reaches disk to become the next drive's
     starting point, which is the worst failure mode a persisted learner has.
     Note this deliberately does NOT gate on instantaneous error being small --
     see the CONVERGED_TAU comment for why that version was biased.

  3. EVERY LOADED VALUE IS RE-CLAMPED. A hand-edited or corrupted param must not
     be able to produce an out-of-range actuator command.

  4. LIMITS ARE PRODUCTS, NOT INDIVIDUALS. The brake gain and the brake rise
     rate both multiply the same command, so the ceiling is chosen against their
     product, not against either one alone.

Everything here is inert unless HondaDynamicTuningEnabled is set; with the
toggle off, callers get pass-through values identical to stock.
"""

import math
import threading
from collections import deque
from queue import Empty, Queue

import numpy as np

from opendbc.car import ACCELERATION_DUE_TO_GRAVITY, structs
from opendbc.car.carlog import carlog
from opendbc.car.common.filter_simple import FirstOrderFilter
from opendbc.car.common.pid import PIDController
# Deliberately NOT gas_interceptor.py: that module is what imports this one's caller, and the old
# `ELESYS_GAS_BP as PEDAL_GAIN_BP` import was a trap -- renaming the gas grid took out
# CarController.__init__, i.e. the car's longitudinal control, for a rename.
from opendbc.sunnypilot.car.honda.elesys_gas import DRIVE_MODE_SLOTS, drive_mode_slot, econ_state, gear_name

LongCtrlState = structs.CarControl.Actuators.LongControlState

DT_CTRL = 0.01

# --- dwell / lag compensation ------------------------------------------------
# update_state() runs every control frame (100 Hz).
#
# 2026-10: the pedal and aero learners this ramp dwell was built for are RETIRED (see "Retired
# channels" below). It now gates only the per-mode steady-pedal counter in observe_pedal(), and
# accel_ref/accel_error are still logged. The history below is kept because it is why the brake
# channel has a different dwell, and because it is what any future gas learner has to answer.
#
# The thing every learner here has to avoid is mistaking ACTUATOR LAG for a gain
# error. The plant tracks the target late, so during any ramp aEgo sits below a
# rising target and (target - aEgo) is positive with nothing wrong with the gain.
# There are two ways to deal with that:
#
#   (a) refuse to learn until the target has been STEADY long enough for the lag
#       to have decayed, or
#   (b) model the lag and subtract it.
#
# This used to do (a): a 1.5 s dwell that reset whenever the commanded accel drifted
# more than 0.20 m/s^2 from the start of the window. Measured against two weeks of
# driving (routes 99/9b/9c/9d/9e, ~45 min engaged), that gate is very nearly closed
# at all times, because (a) is in direct conflict with LEARN_MIN_CMD below. Real
# accel demands RAMP. The median demand episode lasts 0.37-0.90 s, and of the few
# that last over 1.5 s the target swings a median 0.67-0.94 m/s^2 across the episode
# -- 3-5x the tolerance -- so the dwell kept resetting inside the very episodes it
# existed to admit. Engaged+PID+clean time that cleared BOTH gates: 0.4-2.6% of the
# drive, i.e. 0.7 to 19.6 SECONDS per route.
#
# Worse, the exclusion was biased rather than merely sparse. On route a2d078f21b,
# demand in 0.6-1.0 m/s^2 got 28.2 s of which 1.7 s was settled, and demand above
# 1.0 m/s^2 got 20.8 s of which ZERO was settled. The learner could only ever see a
# thin 0.4-0.6 m/s^2 sliver -- exactly where pedal gain is least identifiable
# against grade and wind. The result after two weeks of persistence: the six pedal
# breakpoints read 1.000/1.000/1.000/1.004/1.004/1.000. The channel was dead.
#
# So this now does (b). accel_ref is the planner target pushed through a
# first-order lag matching the plant, and accel_error is measured against THAT
# rather than against the raw target. For a plant that tracks perfectly but late,
# the error is then ~0 by construction on a ramp exactly as it is on a plateau,
# which is what makes it safe to admit ramps at all. The dwell keeps watching the
# COMMANDED value (target + grade) so a pitch transient still resets it; the two
# references are filtered separately for exactly that reason.
#
# PLANT_TAU is a pooled least-squares fit of aEgo against lowpass(target, tau) over
# engaged PID time on four routes. The basin is real but SHALLOW -- residual RMS
# 0.242 at tau=0.30 against 0.256 at tau=0.0 -- because the residual is dominated
# by grade and wind, not by lag. Per route the optimum ranges 0.0-0.5 s. So treat
# the uncertainty on this as +-0.2 s, not +-0.05, and see LEARN_MAX_JERK.
PLANT_TAU = 0.30            # s
# A first-order reference lags a ramp by tau * rate, so an error of dtau in the tau
# estimate leaves dtau * rate of residual. With dtau = 0.2 s, holding that residual
# under 0.1 m/s^2 -- comfortably below the 0.2-0.35 m/s^2 biases this exists to find
# -- caps the admissible ramp rate at 0.5 m/s^3.
#
# This is measured on the FILTERED reference, not on raw target jerk; see the note
# in update_state() for why that distinction is the whole gate. On the filtered
# signal the demand-time distribution is p50 0.21, p75 0.45, p90 0.92 m/s^3, so 0.5
# sits just above the median ramp and admits the bulk of ordinary demand while
# excluding steps, lead cut-ins and the pitch feedforward fading.
#
# Admitted pedal-learn time over the four routes, old gate against this one:
#     9927285994   3.7 s -> 7.7 s      cb866d7303   0.7 s -> 1.5 s
#     e0d590c25c   2.2 s -> 7.7 s      a2d078f21b  19.4 s -> 49.4 s
#     TOTAL       25.9 s -> 66.3 s  (2.6x), and no longer biased against the
# strong-demand bands that carry the actual gain information.
#
# Loosening to 0.8-1.0 m/s^3 buys 3.3-3.6x instead of 2.6x, and was NOT taken: it
# puts the tau-mismatch residual at 0.16-0.20 m/s^2, the same size as the bias being
# learned. Revisit only alongside a tighter PLANT_TAU.
LEARN_MAX_JERK = 0.5        # m/s^3 on the filtered reference; above this the dwell resets
# The filtered ramp rate stays above LEARN_MAX_JERK until the reference has settled
# after a step, so this no longer has to cover the plant's settling time on its own
# -- it is the margin AFTER the model has caught up.
SETTLE_FRAMES = 100

# --- the brake channel keeps the OLD dwell -----------------------------------
# The ramp dwell above is safe for the pedal and aero learners because those
# are small-rate EMAs (2e-4 and below): 2-3x the samples buys 2-3x the convergence
# speed and nothing else. The brake learner is not an EMA. It is a PID integrator
# at BRAKE_KI = 0.5 whose error is WEIGHTED BY BRAKE COMMAND, so it grows on the
# ramp-in and barely unwinds on the release -- the ratchet the original dwell was
# written against. Feeding it the extra ramp samples rails it.
#
# Measured, replaying the real tuner over the same four routes, time spent pinned
# at the 1.60x ceiling:
#     old dwell:  0.0 s / 0.0 s / 0.0 s / 0.0 s
#     ramp dwell: 0.0 s / 0.0 s / 3.3 s / 10.7 s   (and 71.5 s at the 0.85 floor)
# so the brake channel additionally requires the steady-target dwell it has always
# used, and its behaviour is bit-for-bit what shipped. Revisit only by lowering
# BRAKE_KI at the same time, and only against replay.
STEADY_SETTLE_FRAMES = 150
# Drift from the START of the window, NOT frame-to-frame. Compared against the
# previous frame this is a 20 m/s^3 jerk threshold, and real planner targets ramp
# far slower than that, so the gate would never close and the integrator would eat
# pure actuator lag.
STEADY_SETTLE_TOLERANCE = 0.20
# The persisted estimate is a long EMA of the live value that only advances
# while the live value is off its rails. An earlier version gated it on
# |instantaneous error| < 0.3 instead, which is biased: with a real -0.35 m/s^2
# bias and 0.2 m/s^2 of noise, that gate only opens when noise happens to cancel
# the very bias being corrected, so it systematically persists "no correction
# needed". Measured convergence hit-rates were 19-75% depending on band -- i.e.
# the gate was sampling the error distribution, not the learner's state.
CONVERGED_TAU = 3e-4        # EMA rate; ~67 s of learnable time to 1 tau

# Below this the accel command is too small to identify an actuator gain from --
# at settled cruise the residual is dominated by grade and wind, not by pedal
# gain (per-band R^2 of aEgo vs command collapses to ~0.00 above 10 m/s, while
# the error correlates r=0.23-0.56 with the measured pitch term). Learning there
# would just chase the disturbance the pitch feedforward already handles.
LEARN_MIN_CMD = 0.4         # m/s^2

# --- drive-mode gating -------------------------------------------------------
# The pedal -> accel map is NOT one curve. S holds lower gears, so the same
# interceptor command lands a different accel; ECON remaps the pedal outright.
# One set of tables averaged across modes is wrong in every mode, and worse than
# not learning at all, because the error it converges on is a blend that matches
# no actual driving.
#
# The rule here is deliberately the cheap one: LEARN IN THE REFERENCE MODE ONLY
# (D, ECON off), and treat any mode change as a transient. Since the gas learners
# were retired this gates only the brake learner, which sees a different engine
# braking in S and was never characterized outside D.
#
# MEASURED, all 86 routes logged to 2026-09 (1406 min, 0x188 and 0x221 decoded):
#
#   mode          moving     engaged    routes
#   D, ECON off   65,085 s   33,808 s   most
#   D, ECON on       467 s       87 s   be, bf, c0
#   S                148 s       79 s   b1, dd, fc
#   S + ECON           0 s        0 s   -
#
# S decodes through GEAR = 26 (16,164 frames; the 2,975 frames of raw-0 shift
# transients read GEAR = 0, see carstate.py), and ECON through ECON_STATUS. The
# brake learner was correctly frozen in ECON (modeok=0 on bf and c0). What the
# data says about the modes, for whoever fits per-mode pedal tables next:
#   * ECON delivers 0.35-0.48 m/s^2 LESS at the same speed and PCM pedal (>= 40
#     counts), separately in be, bf and c0; achieved/D-predicted median 0.61,
#     slope ratio 0.72-0.84. Same rpm per km/h as D, so the same gearing. Mostly
#     manual driving, one day.
#   * S runs 1.4-2.5x D's rpm per km/h (lower gears). Within 0.04 m/s^2 of D at
#     light pedal (30-70 counts), +0.17..+0.57 at 70-100 counts (2-5 s per bin).
#   * Neither has enough ENGAGED data to fit anything: a steady-pedal admission
#     rule over the whole month collects ~11 s in ECON and ~28 s in S. So the
#     per-mode multipliers in elesys_gas.py (MODE_K) are all 1.0, and this module
#     logs per-mode time and samples (observe_pedal, the hondadyn line) so the
#     tables can be fitted offline once there is data.
#
# UNKNOWN gear deliberately still learns. Gear is only decoded on this platform
# (see the transmissionType note in interface.py); on any other Nidec car
# gearShifter stays `unknown` and gating on it would silently disable the whole
# feature there.
LEARN_GEARS = ("drive",)    # names from structs.CarState.GearShifter

# ECON is decoded on HONDA_ELESYS from 0x221 (honda_accord_au_2015_can.dbc). ECON_ON
# is bit 23 of a 3-BYTE ECON_STATUS -- not the 6-byte frame the other Honda DBCs
# describe. carstate.py decodes it into CS.econ_on; _econ_state() below reads it;
# the dwell reset on change, the reference-mode gate, the econ= log field and the
# drive-mode slot all key off _drive_mode().
#
# Why it looked dead for so long, recorded so nobody re-runs the analysis: across
# 628,053 logged bus-0 frames the message only ever took four values
# (000003/000012/000021/000030) with bytes 0-1 always zero, because ECON was
# simply OFF for the entire 7 hours. An all-zero data field cannot distinguish
# "no state here" from "state that was off", and the owner settled it by pressing
# the button. With ECON on the payloads are 000083-family, which the parser
# accepts as 00008b/00009a/0000a9/0000b8 once the checksum is right.
#
# The ECON BUTTON is separate, at 0x37C bit 48, and is deliberately NOT used:
# it is a momentary press, and what matters is the resulting STATE. 0x37C is
# also CRUISE_PARAMS in the shared _nidec_common.dbc, so defining a signal there
# would reach every Nidec Honda for no benefit here.
LEARN_ECON = (False,)       # learn with ECON off only; ECON remaps the throttle


def _econ_tag(econ) -> str:
  """'-' where ECON is not observable (every other platform), so the line reads the same."""
  return "-" if econ is None else ("on" if econ else "off")


# --- retired channels (2026-10) ------------------------------------------------
# The per-band PEDAL GAIN learner and the AERO ("wind") learner are gone. Measured
# over ~50 tuner-on drives (9f..103), replaying this module over the logs:
#
#   * The pedal learner could not persist anything. The live gain moved 2e-4*err
#     per admitted sample, the persisted estimate 3e-4 of the difference per
#     sample, and every ignition reset live to persisted -- so each drive's
#     progress was thrown away. Admitted pedal time was ~1.6% of engaged time;
#     route 103 moved live by +0.0009 and persisted by 0.0000. Six weeks took the
#     gains from [1,1,1,1.004,1.004,1] to [1,1,.996,.996,.999,.993].
#   * And had it persisted, it would have learned the wrong way. Its gate read
#     actuators.accel, which includes openpilot's own integrator: at 12-30 m/s the
#     admitted frames were ones where the integrator had already lifted a ~0.36
#     m/s^2 target over the 0.4 gate, so it read UNDER-delivery (+0.04..+0.12)
#     where the car really OVER-delivers (-0.19..-0.33 gated on the planner's
#     target). One multiplier was also being asked to fit both slope and offset.
#   * The aero scale was a random walk: median 0.37 of its 0.7-1.5 range inside a
#     drive, rails hit on b0/c8/ce (1.5) and de/fd (0.7), drift uncorrelated with
#     speed mix (r = -0.09), consecutive drives drifting opposite ways. One scalar
#     cannot fit the cruise pedal shape anyway (1.90 at 8-12 m/s, 0.66 at 21-27).
#     It also scaled the brake-side credit, so it moved the brake-on point.
#
# What replaced them: the measured slope table in elesys_gas.py (HondaElesysGasLawV2),
# a constant aero scale of 1.0 (wind_scale), and pedal gain fixed at 1.0. The
# params HondaDynPedalGain0-5 and HondaDynWindFactor are no longer registered, read
# or written; a device that had them keeps the files on disk, unread.

# --- per-mode data, for fitting pedal tables offline -----------------------------
# Counted only while the tuner is enabled, logged in the hondadyn line per drive,
# and (seconds only) persisted as running totals so the UI can show how much there is.
MODE_MOVING_SPEED = 1.0     # m/s; "engaged time" counts longActive frames above this
# The steady-pedal admission rule the audit fitted its tables with: on top of the
# ramp dwell, PID state and no driver/AEB override,
ADMIT_MIN_SPEED = 3.0       # m/s
ADMIT_WINDOW = 50           # interceptor frames (50 Hz) = 1 s of pedal history
ADMIT_PEDAL_MIN = 0.02      # the whole window inside these: off the brake, off the top
ADMIT_PEDAL_MAX = 0.9
ADMIT_PEDAL_STD = 0.008     # and steady
ADMIT_MAX_PITCH = 0.08      # rad, with a fresh pose
MODE_SEC_MAX = 1e9          # clamp on a loaded total; a corrupt value cannot overflow the log

# --- brake channel -----------------------------------------------------------
# MVL runs k_i=2.0 / pos_limit=4.0, i.e. up to 5x the base brake command, on top
# of a 3x faster brake rise. Their product reaches full brake authority from a
# gentle request in about a tenth of a second. This keeps the stock rise rate and
# caps the learned gain at 1.6x; raise only against replay data.
BRAKE_KI = 0.5
BRAKE_POS_LIMIT = 0.6
# The learned gain is TWO-SIDED, because the thing it exists to trim is the /2.6
# divisor in compute_gb_honda_elesys(), which was hand-set to cure severe
# under-braking and is meant to be fine-tuned on the road rather than by edit. A
# one-sided gain can only ever add brake, so if the hand value ends up slightly
# hot there is no way back and the only correction available is another edit.
#
# The two sides are deliberately NOT symmetric. Over-braking on this car is a
# comfort problem the driver feels immediately and can back out of; under-braking
# is the failure mode that produced the original bug. So the learner may add up
# to 60% and remove at most 15%, and the negative side is small enough that a
# wrong learn is a slightly soft stop, not a missed one. Anything below ~0.8x
# would put it in range of undoing the /2.6 fix outright, which is not something
# a self-tuner should be able to do without the driver knowing.
BRAKE_NEG_LIMIT = 0.15        # gain floor 0.85x; ceiling stays 1.60x
BRAKE_LEARN_MIN_SPEED = 1.0   # m/s below which the brake learner is frozen

# --- pitch -------------------------------------------------------------------
PITCH_RC = 0.5              # s, matches the spirit of toyota's filtered pitch
PITCH_ACCEL_LIMIT = 1.5     # m/s^2, hard ceiling on the feedforward
PITCH_STALE_FRAMES = 100    # 1 s without a fresh pose -> ramp the feedforward out
PITCH_FADE_MIN_SPEED = 2.0  # m/s below which the feedforward is fully faded out
PITCH_FADE_FULL_SPEED = 5.0

PERSIST_INTERVAL = 6000     # frames at 100 Hz -> every 60 s
# carlog is forwarded to cloudlog by selfdrive/car/card.py, so these lines land in
# logMessage and come straight back out of a route as plain text -- no capnp
# change, no openpilot import, and nothing hijacked out of carControl.actuators
# (which is what MVL does, at the cost of corrupting everything downstream that
# reads them).
LOG_INTERVAL = 500          # frames at 100 Hz -> every 5 s
LOG_TAG = "hondadyn"

# key -> (default, lo, hi). Every load is re-clamped against this table.
_PARAM_SPEC = {
  # starts at zero gain: flipping the toggle must not change braking until
  # something has actually been learned
  "HondaDynBrakeGain": (0.0, -BRAKE_NEG_LIMIT, BRAKE_POS_LIMIT),
  # running totals of engaged, moving seconds per drive-mode slot -- counters, not learned
  # values, so rule 2 (converged only) does not apply to them
  "HondaDynModeSecD": (0.0, 0.0, MODE_SEC_MAX),
  "HondaDynModeSecECON": (0.0, 0.0, MODE_SEC_MAX),
  "HondaDynModeSecS": (0.0, 0.0, MODE_SEC_MAX),
}
MODE_SEC_KEYS = {slot: f"HondaDynModeSec{slot}" for slot in DRIVE_MODE_SLOTS}


def _open_params():
  """Params lives in openpilot, this module lives in opendbc. Import lazily so
  opendbc still imports standalone (PC tests, CI) with no openpilot on the path."""
  try:
    from openpilot.common.params import Params
    return Params()
  except Exception:
    return None


class _ParamWriter:
  """Background writer. Param puts hit disk; doing that from the 100 Hz control
  loop would add jitter, so snapshots are queued and coalesced off-thread."""

  def __init__(self, params):
    self._params = params
    self._queue: Queue = Queue()
    self.write_errors = 0
    self._thread = threading.Thread(target=self._run, name="honda-dyn-param-writer", daemon=True)
    self._thread.start()

  def put_many(self, values: dict) -> None:
    # floats are immutable and the dict is rebuilt per call, so the writer thread
    # never observes a torn value
    self._queue.put({k: float(v) for k, v in values.items()})

  def _run(self) -> None:
    while True:
      pending = self._queue.get()
      # Collapse anything that piled up so a slow disk keeps only the newest value per key.
      try:
        while True:
          pending.update(self._queue.get_nowait())
      except Empty:
        pass

      for key, value in pending.items():
        try:
          self._params.put(key, value)
        except Exception:
          # counted rather than silently dropped, so a missing params_keys.h entry
          # shows up in debug_values() instead of looking like "nothing to learn"
          self.write_errors += 1


def _finite(x, fallback: float = 0.0) -> float:
  """Params.get() may hand back a numpy scalar, a Decimal, or bytes depending on
  the build, so convert rather than isinstance-check -- an isinstance gate would
  silently drop every stored value and make persistence a no-op across boots."""
  try:
    v = float(x)
  except (TypeError, ValueError):
    return fallback
  return v if math.isfinite(v) else fallback


class HondaDynamicTuner:
  def __init__(self, CP, CP_SP):
    self.CP = CP
    self.CP_SP = CP_SP

    # Only Nidec cars running openpilot longitudinal have anything to learn.
    # Bosch and stock-long cars skip the params read and the writer thread
    # entirely rather than paying for a feature that can never apply to them.
    self.applicable = self._is_applicable(CP, CP_SP)
    self.interceptor = bool(getattr(CP_SP, "enableGasInterceptor", False))

    self._params = _open_params() if self.applicable else None
    self.enabled = self._get_bool("HondaDynamicTuningEnabled")

    self._writer = _ParamWriter(self._params) if (self._params is not None and self.enabled) else None

    # per-mode data: this drive's engaged seconds and admitted steady-pedal samples per slot, and
    # the running totals of seconds loaded from the params
    self.slot = "D"
    self.mode_seconds = dict.fromkeys(DRIVE_MODE_SLOTS, 0.0)
    self.mode_admitted = dict.fromkeys(DRIVE_MODE_SLOTS, 0)
    self.mode_seconds_loaded = {s: self._get_float(k) for s, k in MODE_SEC_KEYS.items()}
    self._pedal_window: deque = deque(maxlen=ADMIT_WINDOW)
    # which gas law the interceptor path is running ("v1"/"v2" on HONDA_ELESYS), for the log
    self.gas_law = "-"

    # brake channel
    self.brake_pid = PIDController(k_p=0.0, k_i=BRAKE_KI, pos_limit=BRAKE_POS_LIMIT,
                                   neg_limit=-BRAKE_NEG_LIMIT, rate=50)
    self.brake_pid.reset()
    self.brake_gain_converged = self._get_float("HondaDynBrakeGain")
    self.brake_pid.i = self.brake_gain_converged
    self.brake_pid_factor = self.brake_gain_converged

    # pitch
    self.pitch_filter = FirstOrderFilter(0.0, PITCH_RC, DT_CTRL)
    self.pitch = 0.0
    self._pose_stale = PITCH_STALE_FRAMES

    # dwell tracking
    self.accel_target = 0.0
    self.accel_error = 0.0
    # First-order model of the plant's own lag. Every learner compares aEgo
    # against this rather than the raw target, so a plant that tracks perfectly
    # but late reads as zero error instead of as a gain shortfall.
    self.accel_ref_filter = FirstOrderFilter(0.0, PLANT_TAU, DT_CTRL)
    self.accel_ref = 0.0
    # The same model applied to the COMMANDED value (target + grade). Only the
    # dwell reads this: it has to see a pitch transient as a transient, while the
    # error above must stay measured against what the planner actually asked for.
    self.cmd_ref_filter = FirstOrderFilter(0.0, PLANT_TAU, DT_CTRL)
    self.cmd_ref = 0.0
    self._settle = 0
    # the brake channel's separate, stricter dwell
    self._dwell_ref = 0.0
    self._settle_steady = 0

    # drive-mode tracking
    # (gear, econ); either element None means "not observable, do not gate on it"
    self.drive_mode: tuple = (None, None)
    self.mode_ok = True
    self.long_active = False

  @staticmethod
  def _is_applicable(CP, CP_SP) -> bool:
    try:
      from opendbc.car.honda.values import HONDA_BOSCH
      return bool(CP.openpilotLongitudinalControl) and CP.carFingerprint not in HONDA_BOSCH
    except Exception:
      # no CP (unit tests) or an unexpected shape: let the toggle decide
      return True

  # --- param plumbing --------------------------------------------------------

  def _get_float(self, key: str) -> float:
    # A key with no _PARAM_SPEC entry (a slot added to DRIVE_MODE_SLOTS without its
    # HondaDynModeSec<slot>, say) must not take out CarController.__init__ -- that is
    # a car with no longitudinal control, for a typo. Fall back to a neutral value
    # and let the writer's error count surface it instead.
    spec = _PARAM_SPEC.get(key)
    if spec is None:
      carlog.error(f"{LOG_TAG} no _PARAM_SPEC entry for {key}; using 0.0 and not learning it")
      return 0.0
    default, lo, hi = spec
    if self._params is None or not self.enabled:
      return default
    try:
      value = self._params.get(key, return_default=True)
    except Exception:
      return default
    value = _finite(value, default)
    # rule 3: a hand-edited or corrupted param must not reach an actuator
    return float(np.clip(value, lo, hi))

  def _get_bool(self, key: str) -> bool:
    if self._params is None:
      return False
    try:
      return bool(self._params.get_bool(key))
    except Exception:
      return False

  def persist(self, frame: int) -> None:
    """Rule 2: writes the *converged* brake estimate, never the live one. A value
    that railed during a transient is never written, so it cannot come back as
    next drive's starting point. The per-mode seconds are counters, not learned
    values, and are written as running totals."""
    if self._writer is None or not self.enabled or frame % PERSIST_INTERVAL != 0:
      return
    values = {"HondaDynBrakeGain": self.brake_gain_converged}
    # plain floats only: put_many() does float(v) in THIS thread, the control thread
    values.update({key: self.mode_seconds_total(slot) for slot, key in MODE_SEC_KEYS.items()})
    self._writer.put_many(values)

  def mode_seconds_total(self, slot: str) -> float:
    """The running total for a slot: what was loaded at ignition plus this drive."""
    total = _finite(self.mode_seconds_loaded.get(slot, 0.0)) + _finite(self.mode_seconds.get(slot, 0.0))
    return float(min(max(total, 0.0), MODE_SEC_MAX))

  # --- per-frame bookkeeping -------------------------------------------------

  def update_state(self, CC, CS) -> float:
    """Call once per control frame, before gas/brake are computed. Returns the
    pitch feedforward in m/s^2 to add to the accel target (0.0 when disabled).

    Sign convention matches ford/toyota: orientationNED[1] is pitch, nose-up is
    positive, so climbing adds positive demand.
    """
    actuators = CC.actuators
    target = _finite(actuators.accel if CC.longActive else 0.0)

    # Computed BEFORE the dwell, because the dwell has to watch the value that is
    # actually commanded. The learners see `target + hill_accel` (carcontroller.py
    # builds adjust_accel that way), but the dwell used to compare only `target`.
    # A planner target is perfectly steady through a stop approach while the pitch
    # term swings by its whole magnitude across the 2-5 m/s fade, so that swing
    # walked straight past rule 1: measured over repeated closed-loop stop
    # approaches it fed the brake learner +0.0074 per downhill stop against
    # -0.0015 per uphill one, a one-signed artifact of the fade rather than
    # anything about the brakes. Watching the commanded value makes any pitch
    # transient -- fade, grade change, or filter lag -- reset the dwell exactly as
    # a target transient does. With the tuner off hill_accel is 0.0, so this is
    # bit-identical to watching `target` alone.
    hill_accel = self._pitch_feedforward(CC, CS)
    commanded = target + hill_accel

    # A gear change swaps the pedal map underneath the learners mid-sample, so it
    # is a transient in exactly the sense rule 1 means -- folded into the same
    # reset as a target transient so the edge frame behaves identically to one.
    # mode_ok then holds learning off entirely outside the reference mode.
    mode = self._drive_mode(CS)
    mode_changed = mode != self.drive_mode
    self.drive_mode = mode
    self.mode_ok = self._mode_learnable(mode)
    # stashed purely for the log line: a dwell-open percentage computed over
    # disengaged frames is meaningless (target pins at 0.0, so the gate opens
    # trivially) and that is exactly how the old telemetry read 88% while the
    # learner was getting ~1 s of real samples per drive.
    self.long_active = bool(CC.longActive)

    # per-mode engaged time, the same slots the gas law uses (elesys_gas.drive_mode_slot)
    self.slot = drive_mode_slot(*mode)
    if self.enabled and self.long_active and _finite(CS.out.vEgo) > MODE_MOVING_SPEED:
      self.mode_seconds[self.slot] = self.mode_seconds.get(self.slot, 0.0) + DT_CTRL

    # The dwell now watches JERK, not drift from the window start. A sustained
    # ramp is exactly what the lag model below is for, so admitting it is the
    # point; what still has to be excluded is a transient fast enough that the
    # +-0.2 s uncertainty on PLANT_TAU matters. A mode change is a step in the
    # pedal map itself, which no lag model covers, so it resets as it always did.
    # Advance the plant model first -- every frame, engaged or not, so it is
    # already tracking when the dwell opens rather than converging from stale
    # state. The dwell below keys off ITS ramp rate, so the two must describe the
    # same frame.
    # Filtered from `target`, NOT from `commanded`, for the same reason the error
    # below has always been measured against the planner target: the grade term is
    # added to the command precisely so the car still achieves `target`, so folding
    # it into the reference would ask every learner to correct for a hill the pitch
    # feedforward is already handling. The DWELL still watches `commanded` -- a
    # pitch transient is a transient whatever it is added to.
    prev_ref = self.cmd_ref
    self.cmd_ref = _finite(self.cmd_ref_filter.update(commanded))
    self.accel_ref = _finite(self.accel_ref_filter.update(target))

    # The dwell watches the ramp rate of the FILTERED reference, not the raw
    # frame-to-frame target jerk. That distinction is the whole gate. The planner
    # target is noisy enough on its own that ~12% of frames exceed 0.5 m/s^3, so a
    # raw-jerk dwell almost never strings together SETTLE_FRAMES clean frames and
    # measures WORSE than the gate it replaces (2.3-5.8 s admitted against 25.9 s
    # over the same four routes). The filtered rate is also the quantity that
    # actually multiplies the tau error, and it has a second useful property: after
    # a step it stays above the threshold until the reference itself has settled,
    # so the dwell cannot start counting while the lag model is still catching up.
    ramp = abs(self.cmd_ref - prev_ref) / DT_CTRL
    if mode_changed or ramp > LEARN_MAX_JERK:
      self._settle = 0
    else:
      self._settle = min(self._settle + 1, SETTLE_FRAMES)

    # The brake channel's own, stricter dwell -- unchanged from what shipped. See
    # STEADY_SETTLE_FRAMES for why that channel does not get the ramp dwell.
    if mode_changed or abs(commanded - self._dwell_ref) > STEADY_SETTLE_TOLERANCE:
      self._settle_steady = 0
      self._dwell_ref = commanded
    else:
      self._settle_steady = min(self._settle_steady + 1, STEADY_SETTLE_FRAMES)

    # the planner target, NOT the commanded value: the command-magnitude gates
    # (LEARN_MIN_CMD) are about what the planner asked for, not about grade
    self.accel_target = target
    # Lag-compensated. The brake learner reads this (and the log line reports it);
    # the pedal and aero channels that also read it are retired.
    self.accel_error = self.accel_ref - _finite(CS.out.aEgo)

    # Rule 5: NOTHING WOUND UP IN ONE ENGAGEMENT CROSSES INTO THE NEXT.
    # The brake integrator's only unwind paths are the stopping clamp and the
    # standstill reset, and neither fires on a plain disengage at speed. Without
    # this, winding to the 0.6 rail on a downgrade, disengaging, and re-engaging
    # applies 1.6x the requested brake open-loop for the whole SETTLE_FRAMES
    # dwell before the PID is allowed to correct it -- in a condition (fresh
    # engagement, possibly different grade or load) that is not the one it was
    # learned in. _settle re-arms itself, since target snaps to 0.0 here.
    if not CC.longActive:
      self.brake_pid.i = self.brake_gain_converged
      self.brake_pid_factor = self.brake_gain_converged

    return hill_accel

  def _pitch_feedforward(self, CC, CS) -> float:
    """The grade term to add to the accel target, in m/s^2. 0.0 when disabled or
    outside the PID state. Advances the pitch filter, so it must be called
    exactly once per control frame."""
    if not self.enabled:
      self.pitch = 0.0
      return 0.0

    # A stale or non-finite pose must decay the feedforward out rather than
    # freezing the last slope in forever.
    raw = None
    if len(CC.orientationNED) == 3:
      candidate = _finite(CC.orientationNED[1], float("nan"))
      if math.isfinite(candidate) and abs(candidate) < math.pi / 2:
        raw = candidate

    if raw is None:
      self._pose_stale = min(self._pose_stale + 1, PITCH_STALE_FRAMES)
      if self._pose_stale >= PITCH_STALE_FRAMES:
        raw = 0.0                      # ramp out through the filter
      else:
        raw = self.pitch_filter.x      # brief dropout: hold
    else:
      self._pose_stale = 0

    self.pitch = float(self.pitch_filter.update(raw))
    accel = math.sin(self.pitch) * ACCELERATION_DUE_TO_GRAVITY

    # Fade out at low speed and outside the PID state. On an uphill the term is
    # positive, which correctly asks for less brake while decelerating -- but at a
    # standstill it would release the brake that is holding the car against
    # gravity and let it roll back. openpilot's own longitudinal PI already
    # integrates out steady grade, so the value here is faster response to grade
    # CHANGES at speed; there is nothing to gain by keeping it near zero.
    #
    # KNOWN ISSUE, deliberately not changed here: the 2-5 m/s fade band sits
    # entirely inside the PID state (stopping only begins below 0.8 m/s, stopping_tune.py),
    # so on a real stop approach this hands the whole grade term back to
    # openpilot's longitudinal integrator at a rate it cannot follow -- modelled
    # shortfall 0.115 m/s^2 on a 4% downhill rising to 0.249 on a 10%. Toyota
    # avoids this by adding a HIGH-pass pitch (HighPassFilter, carcontroller.py
    # :250) so the term is transient by construction and never has to be handed
    # back; this uses the low-pass, which is MVL's shape. Fixing it means either
    # switching to a high-pass or dropping the speed fade (the `!= pid` gate below
    # already covers the standstill-rollback case the comment above worries
    # about) -- both change how the car brakes on grades, so they want road data
    # first. The learner-corruption half of the problem is already handled: the
    # dwell in update_state() watches target + this term, so the fade transient
    # resets it instead of being learned.
    if not CC.longActive or CC.actuators.longControlState != LongCtrlState.pid:
      return 0.0
    fade = float(np.interp(_finite(CS.out.vEgo), [PITCH_FADE_MIN_SPEED, PITCH_FADE_FULL_SPEED], [0.0, 1.0]))
    return float(np.clip(accel * fade, -PITCH_ACCEL_LIMIT, PITCH_ACCEL_LIMIT))

  def filtered_pitch(self):
    """The filtered pitch in radians (nose-up positive) for a consumer outside the PID state --
    the soft final stop's grade-aware cap (elesys_stop.py) -- or None when there is no usable pose:
    the tuner is off, no pose has arrived yet, or it has been stale for PITCH_STALE_FRAMES (the
    filter is then decaying toward 0, which is not a measurement). update_state() advances it, so
    read it after that call in the same frame."""
    if not self.enabled or self._pose_stale >= PITCH_STALE_FRAMES:
      return None
    p = _finite(self.pitch, float("nan"))
    return p if math.isfinite(p) else None

  @staticmethod
  def _gear_name(CS):
    """Gear as a plain name, or None if it cannot be determined. Shared with the gas law
    (elesys_gas.gear_name), which needs the same slot; an unrecognised value would read as
    "not drive" and freeze the brake learner forever, so it normalizes every shape."""
    return gear_name(CS)

  @staticmethod
  def _econ_state(CS):
    """True/False where ECON is decoded, None where it is not observable (elesys_gas.econ_state).
    carstate.py sets CS.econ_on from ECON_STATUS (0x221) on HONDA_ELESYS and leaves it None
    everywhere else, so gating stays inert on other platforms."""
    return econ_state(CS)

  @classmethod
  def _drive_mode(cls, CS) -> tuple:
    """The pedal map currently in force, as (gear, econ).

    Either element may be None, meaning "not observable, do not gate on it".
    A change in EITHER element is a mode change, because both swap the
    pedal -> accel map underneath the learners.
    """
    return cls._gear_name(CS), cls._econ_state(CS)

  @staticmethod
  def _mode_learnable(mode) -> bool:
    gear, econ = mode
    if gear is not None and gear not in LEARN_GEARS:
      return False
    if econ is not None and econ not in LEARN_ECON:
      return False
    return True

  def _learn_ok(self, CC, CS, reference_mode_only: bool = True) -> bool:
    """Rule 1: no learning during transients, overrides, or non-PID states. The per-mode
    data counter passes reference_mode_only=False: counting the other modes is its point."""
    return (self.enabled
            and CC.longActive
            and (self.mode_ok or not reference_mode_only)
            and self._settle >= SETTLE_FRAMES
            and CC.actuators.longControlState == LongCtrlState.pid
            and not CS.out.gasPressed
            and not CS.out.brakePressed
            and not CS.out.stockAeb)

  @staticmethod
  def _track(converged: float, live: float) -> float:
    return converged + CONVERGED_TAU * (live - converged)

  # --- per-mode pedal data (the pedal learner is retired) ---------------------

  def observe_pedal(self, CC, CS, gas_cmd: float, law: str = "") -> None:
    """Called once per interceptor frame (50 Hz) with the command actually sent. Learns
    nothing: it counts, per drive-mode slot, the samples a steady-pedal plant fit could use,
    so the hondadyn line says how much ECON/S data a drive collected. Runs inside
    CarController.update(), so it never raises."""
    try:
      if law:
        self.gas_law = str(law)
      self._pedal_window.append(_finite(gas_cmd))
      if self._admit_pedal_sample(CC, CS):
        self.mode_admitted[self.slot] = self.mode_admitted.get(self.slot, 0) + 1
    except Exception:
      pass

  def _admit_pedal_sample(self, CC, CS) -> bool:
    """_learn_ok() minus the reference-mode gate (the point is to count the other modes), plus
    the steady-pedal rule the offline tables were fitted with."""
    if not self._learn_ok(CC, CS, reference_mode_only=False):
      return False
    if _finite(CS.out.vEgo) < ADMIT_MIN_SPEED:
      return False
    if self._pose_stale != 0 or not abs(self.pitch) < ADMIT_MAX_PITCH:
      return False
    window = self._pedal_window
    if len(window) < ADMIT_WINDOW or min(window) < ADMIT_PEDAL_MIN or max(window) > ADMIT_PEDAL_MAX:
      return False
    return bool(float(np.std(window)) < ADMIT_PEDAL_STD)

  # --- aero (frozen) ------------------------------------------------------------

  def wind_scale(self) -> float:
    """The aero scale on wind_brake, which carcontroller.py applies to BOTH the interceptor
    offset and the brake-side credit (apply_brake -= wind_brake * scale). FROZEN at 1.0 since
    2026-10 (see "Retired channels"): the learner was a random walk on its 0.7-1.5 range, and
    under the v2 gas law it would lose its lever on the cruise pedal and rail, moving the
    brake-on point by up to +-50%. Persisted values were ~0.80, so with the tuner on the
    brake-on point moves slightly later: about 5 counts less brake at 25 m/s for light requests."""
    return 1.0

  def update_wind(self, CC, CS, wind_brake_ms2: float) -> None:
    """Retired (see wind_scale). Kept so carcontroller.py's call needs no upstream edit."""
    return

  # --- brake channel ---------------------------------------------------------

  def brake_gain(self, CC, CS, apply_brake_frac: float) -> float:
    """Learned multiplier on the base brake command. This is what replaces
    hand-editing the brake divisor in compute_gb_honda_elesys()."""
    if not self.enabled:
      return 1.0

    stopping = CC.actuators.longControlState == LongCtrlState.stopping

    # Rule 1: the dwell gate is what stops the integrator from eating brake
    # hydraulic lag. Without it a normal deceleration rails the gain in seconds.
    active = (apply_brake_frac > 0.0
              and self.accel_target < -LEARN_MIN_CMD
              and self._settle_steady >= STEADY_SETTLE_FRAMES
              and self.mode_ok
              and CC.longActive
              and CC.actuators.longControlState == LongCtrlState.pid
              and CS.out.vEgo > BRAKE_LEARN_MIN_SPEED
              and not CS.out.stockAeb
              and not CS.out.brakePressed
              and not CS.out.gasPressed)

    if active:
      self.brake_pid_factor = float(self.brake_pid.update(error=-self.accel_error * apply_brake_frac,
                                                          speed=CS.out.vEgo))
      # rule 2, now with two rails to stay off rather than one
      if -BRAKE_NEG_LIMIT + 1e-6 < self.brake_pid_factor < BRAKE_POS_LIMIT - 1e-6:
        self.brake_gain_converged = float(np.clip(self._track(self.brake_gain_converged, self.brake_pid_factor),
                                                  -BRAKE_NEG_LIMIT, BRAKE_POS_LIMIT))

    # During the stopping phase the PID is frozen, so the gain would otherwise be
    # applied open-loop for the whole approach. Hold it no higher than the value
    # that was actually converged at speed -- and clamp the INTEGRATOR too, not
    # just the output, or a rolling stop (stopping entered and left without ever
    # reaching standstill) hands the wound value straight back on the next frame.
    if stopping:
      self.brake_pid.i = min(self.brake_pid.i, self.brake_gain_converged)
      self.brake_pid_factor = min(self.brake_pid_factor, self.brake_gain_converged)

    # Each standstill resets to the converged estimate, so nothing a single
    # episode wound up survives into the next stop. Note this ASSIGNS rather
    # than min()s -- it has to re-arm the estimate for the next stop, not just
    # clamp this one -- which is exactly why the hold below must not use it.
    if CS.out.vEgo < 1e-3:
      self.brake_pid.i = self.brake_gain_converged
      self.brake_pid_factor = self.brake_gain_converged

    # FADE the learned gain out below the speed it can actually be learned at.
    #
    # The learner is frozen at or below BRAKE_LEARN_MIN_SPEED (see `active` above), so
    # anything applied down there is pure open loop -- and the standstill hold in
    # particular is the command interface.py tuned by hand, where stopAccel = -0.8 was
    # chosen to land cb 189 instead of the railed 255.
    #
    # A hard cutoff at standstill is NOT enough, and the first version of this fix got
    # that wrong: returning 1.0 only below 1e-3 m/s left a cliff immediately above it.
    # Measured on the real controller with a converged gain of 0.60, holding in the
    # stopping state:
    #     vEgo 0.0005 m/s -> cb 189
    #     vEgo 0.0020 m/s -> cb 255
    # a 66-count step across a 1.5 mm/s change in speed, with no hysteresis -- so a car
    # dithering around zero on a rough surface would chatter the hold between 189 and the
    # rail. And through the whole 0.001-1.0 m/s band the gain was applied with no feedback
    # at all; above 0.8 m/s (the stopping speed) not even the `stopping` clamp applies, so a
    # live railed 1.60x could reach the wire while the converged estimate was still ~0.
    #
    # Ramping instead of cutting fixes both: continuous everywhere, and the correction is
    # only applied in proportion to how much of it was actually learned at that speed.
    fade = float(np.clip(_finite(CS.out.vEgo) / BRAKE_LEARN_MIN_SPEED, 0.0, 1.0))
    gain = 1.0 + fade * float(np.clip(_finite(self.brake_pid_factor), -BRAKE_NEG_LIMIT, BRAKE_POS_LIMIT))
    return gain

  # --- telemetry -------------------------------------------------------------

  def log_state(self, frame: int) -> None:
    """Emit the learned state into the route log. Without this there is no way to
    watch convergence, or to work out after the fact why a learn went wrong."""
    if not self.enabled or frame % LOG_INTERVAL != 0:
      return
    try:
      v = self.debug_values()
      # lists in [...] in DRIVE_MODE_SLOTS order (D, ECON, S), so parse_hondadyn.py reads them as
      # float lists: modesec = engaged moving seconds this drive, modeadm = steady-pedal samples
      # (50 Hz) this drive, modetot = running total of seconds, as persisted
      sec = ",".join(f"{v['mode_seconds'][s]:.1f}" for s in DRIVE_MODE_SLOTS)
      adm = ",".join(f"{v['mode_admitted'][s]:d}" for s in DRIVE_MODE_SLOTS)
      tot = ",".join(f"{v['mode_seconds_total'][s]:.0f}" for s in DRIVE_MODE_SLOTS)
      carlog.info(
        f"{LOG_TAG} gaslaw={v['gas_law']} slot={v['slot']} " +
        f"modesec=[{sec}] modeadm=[{adm}] modetot=[{tot}] " +
        f"brake={v['brake_gain']:.3f} brakec={v['brake_gain_converged']:.3f} " +
        f"pitch={v['pitch']:+.4f} " +
        f"settle={self._settle} settles={self._settle_steady} eng={int(v['long_active'])} " +
        f"aref={v['accel_ref']:+.3f} aerr={v['accel_error']:+.3f} " +
        f"stale={v['pose_stale']} werr={v['write_errors']} " +
        f"gear={v['drive_mode'][0] or '-'} econ={_econ_tag(v['drive_mode'][1])} " +
        f"modeok={int(v['mode_ok'])}")
    except Exception:
      pass

  def debug_values(self) -> dict:
    return {
      "enabled": self.enabled,
      "gas_law": self.gas_law,
      "slot": self.slot,
      "mode_seconds": {s: float(self.mode_seconds.get(s, 0.0)) for s in DRIVE_MODE_SLOTS},
      "mode_admitted": {s: int(self.mode_admitted.get(s, 0)) for s in DRIVE_MODE_SLOTS},
      "mode_seconds_total": {s: self.mode_seconds_total(s) for s in DRIVE_MODE_SLOTS},
      # BOTH as gains. brake_gain_converged is stored as an OFFSET (the param
      # defaults to 0.0 so the toggle changes nothing until something is learned),
      # but reporting the offset next to a gain made the log unreadable: a settled
      # brake channel logged "brake=1.015 brakec=0.015", which reads as a
      # disagreement rather than as the same number twice.
      "brake_gain": 1.0 + self.brake_pid_factor,
      "brake_gain_converged": 1.0 + self.brake_gain_converged,
      "pitch": self.pitch,
      "pose_stale": self._pose_stale,
      "drive_mode": self.drive_mode,
      "mode_ok": self.mode_ok,
      "long_active": self.long_active,
      "accel_ref": self.accel_ref,
      "accel_error": self.accel_error,
      "write_errors": self._writer.write_errors if self._writer is not None else -1,
    }
