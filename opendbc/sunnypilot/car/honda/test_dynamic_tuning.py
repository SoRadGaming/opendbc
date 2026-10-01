#!/usr/bin/env python3
"""Offline checks for HondaDynamicTuner. Runs against a stubbed opendbc surface.

These do not prove the tuning is *good* -- only road data can do that. They prove
the things that have to hold before it is ever flashed:
  1. with the toggle off, everything the car controller consumes is stock,
  2. the brake learner moves the right way and settles,
  3. the clamps hold under adversarial input,
  4. no learner can rail on actuator lag, latch, or persist a transient,
  5. the retired pedal and aero learners stay retired (pedal gain 1.0, aero 1.0),
     and the per-drive-mode data counter counts the right thing in the right slot.

Several checks below are named for the specific failure mode they regression-test.
"""

import sys
from dataclasses import dataclass, field
from unittest import mock

import numpy as np

from opendbc.car import structs
from opendbc.sunnypilot.car.honda import dynamic_tuning as dt
from opendbc.sunnypilot.car.honda.dynamic_tuning import (
  HondaDynamicTuner, SETTLE_FRAMES, BRAKE_POS_LIMIT, PITCH_ACCEL_LIMIT, PITCH_STALE_FRAMES,
)

LongCtrlState = structs.CarControl.Actuators.LongControlState

MAX_SETTLE_FRAMES = 1000   # generous cap for settle(); real waits are 135-177

NIDEC_BRAKE_MAX = 256
NIDEC_GAS_MAX = 198

FAILURES = []


def check(name, cond, detail=""):
  if cond:
    print(f"  PASS  {name}")
  else:
    print(f"  FAIL  {name}  {detail}")
    FAILURES.append(name)


@dataclass
class Actuators:
  accel: float = 0.0
  longControlState: object = LongCtrlState.pid


@dataclass
class CC:
  longActive: bool = True
  orientationNED: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
  actuators: Actuators = field(default_factory=Actuators)


@dataclass
class Out:
  vEgo: float = 10.0
  aEgo: float = 0.0
  gasPressed: bool = False
  brakePressed: bool = False
  stockAeb: bool = False


@dataclass
class CS:
  out: Out = field(default_factory=Out)


class FakeParams:
  """Stands in for openpilot Params, including the UnknownKeyName behaviour."""

  def __init__(self, values=None, known=None):
    self.values = dict(values or {})
    self.known = known
    self.written = {}

  def _check(self, key):
    if self.known is not None and key not in self.known:
      raise KeyError(f"UnknownKeyName: {key}")

  def get(self, key, return_default=False):
    self._check(key)
    return self.values.get(key)

  def get_bool(self, key):
    self._check(key)
    return bool(self.values.get(key, False))

  def put(self, key, value):
    self._check(key)
    self.written[key] = value


def make_tuner(enabled=True, params=None):
  if params is not None:
    dt._open_params = lambda: params
  else:
    dt._open_params = lambda: None
  t = HondaDynamicTuner(CP=None, CP_SP=None)
  if params is None:
    t.enabled = enabled
  return t


def settle(t, cc, cs, n=None):
  """Drive update_state until the dwell is actually open.

  This used to run a fixed SETTLE_FRAMES + 5 frames, which was enough when the
  dwell started counting from the first steady frame. It now counts only once the
  lag model has caught up with the commanded step, so the wait depends on the step
  size -- measured 135 frames for a 0.5 m/s^2 step, 156 for 1.0, 177 for 2.0.
  Waiting on the real condition keeps the tests honest about that instead of
  encoding a constant that has to be re-derived every time PLANT_TAU moves.
  """
  if n is not None:
    for _ in range(n):
      t.update_state(cc, cs)
    return
  for _ in range(MAX_SETTLE_FRAMES):
    t.update_state(cc, cs)
    if t._settle >= SETTLE_FRAMES:
      return
  raise AssertionError(f"dwell never opened within {MAX_SETTLE_FRAMES} frames (settle={t._settle})")


def base(v=10.0, target=0.0, a=0.0):
  cc, cs = CC(), CS()
  cc.actuators.accel = target
  cs.out.vEgo, cs.out.aEgo = v, a
  return cc, cs


# --- 1. disabled is a strict no-op ------------------------------------------

print("\n[1] toggle off -> stock behaviour")
t = make_tuner(enabled=False)
cc, cs = base(12.0, 1.0)
cc.orientationNED = [0.0, 0.30, 0.0]
check("pitch feedforward is exactly 0.0", t.update_state(cc, cs) == 0.0)
check("wind scale is exactly 1.0", t.wind_scale() == 1.0)
check("brake gain is exactly 1.0", t.brake_gain(cc, cs, 0.5) == 1.0)

for _ in range(500):
  t.update_state(cc, cs)
  t.observe_pedal(cc, cs, 0.3, "v2")
  t.update_wind(cc, cs, 0.2)
check("nothing is counted when disabled",
      all(x == 0 for x in t.mode_admitted.values()) and all(x == 0.0 for x in t.mode_seconds.values()),
      f"{t.mode_admitted} {t.mode_seconds}")

# a scale of exactly 1.0 must be an identity on the command, not just close
gas = 0.4213
check("disabled wind scale is an exact identity", gas * t.wind_scale() == gas)


# --- 2. pitch feedforward ----------------------------------------------------

print("\n[2] pitch feedforward")
t = make_tuner()
cc, cs = base()
cc.orientationNED = [0.0, 0.10, 0.0]
for _ in range(500):
  up = t.update_state(cc, cs)
check("uphill adds positive demand", up > 0)
cc.orientationNED = [0.0, -0.10, 0.0]
for _ in range(500):
  down = t.update_state(cc, cs)
check("downhill subtracts demand", down < 0)

t = make_tuner()
cc, cs = base()
cc.orientationNED = [0.0, 1.4, 0.0]     # absurd 80-degree slope
for _ in range(2000):
  v = t.update_state(cc, cs)
check("pitch feedforward is clamped", abs(v) <= PITCH_ACCEL_LIMIT + 1e-9, f"{v}")

t = make_tuner()
cc, cs = base()
cc.orientationNED = [0.0, float("nan"), 0.0]
vals = [t.update_state(cc, cs) for _ in range(200)]
check("NaN pitch never reaches the output", all(np.isfinite(x) for x in vals))
cc.orientationNED = [0.0, float("inf"), 0.0]
vals = [t.update_state(cc, cs) for _ in range(200)]
check("inf pitch never reaches the output", all(np.isfinite(x) for x in vals))

# stale pose must ramp out, not freeze forever  (review finding 8)
t = make_tuner()
cc, cs = base()
cc.orientationNED = [0.0, 0.15, 0.0]
for _ in range(1000):
  held = t.update_state(cc, cs)
cc.orientationNED = []                    # pose lost
for _ in range(PITCH_STALE_FRAMES + 2000):
  after = t.update_state(cc, cs)
check("stale pose ramps the feedforward out", abs(after) < 0.05 * abs(held), f"held={held:.3f} after={after:.3f}")


# --- 3. the retired channels stay retired ------------------------------------
#
# The per-band pedal gain could not persist anything and was gate-biased; the aero scale was a
# random walk that also moved the brake-on point. Both are gone: no API, no params, and the aero
# scale is a constant 1.0 whatever the error does.

print("\n[3] retired pedal and aero learners")
check("no pedal-gain API is left on the tuner",
      not any(hasattr(HondaDynamicTuner, a) for a in ("pedal_gain_at", "update_pedal")))
check("no retired key is read or written",
      not any(k.startswith(("HondaDynPedalGain", "HondaDynWindFactor")) for k in dt._PARAM_SPEC),
      f"{sorted(dt._PARAM_SPEC)}")
check("the tuner no longer imports the gas grid (the PEDAL_GAIN_BP trap)",
      not hasattr(dt, "PEDAL_GAIN_BP") and not hasattr(dt, "ELESYS_GAS_BP"))

# adversarial: a huge, one-signed error in the old aero learner's own band, tuner ON
t = make_tuner()
cc, cs = base(30.0, 0.1, -3.0)
settle(t, cc, cs)
scales = set()
for _ in range(8000):
  t.update_state(cc, cs)
  t.update_wind(cc, cs, 0.441)
  scales.add(t.wind_scale())
check("wind scale is frozen at exactly 1.0 with the tuner on", scales == {1.0}, f"{scales}")


# --- 4. per-drive-mode data counter -------------------------------------------
#
# Nothing learns on the gas side. observe_pedal() counts, per slot (D / ECON / S), the samples a
# steady-pedal plant fit could use, and update_state() the engaged moving seconds, so the hondadyn
# line says how much ECON and S data a drive collected.

print("\n[4] per-mode data counter")
GearShifter = structs.CarState.GearShifter


def counted(gear=None, econ=None, frames=3000, pedal=lambda i: 0.20, mutate=None, v=12.0, target=0.5):
  t = make_tuner()
  cc, cs = base(v, target, target)
  if gear is not None:
    cs.out.gearShifter = gear
  cs.econ_on = econ
  settle(t, cc, cs)
  if mutate is not None:
    mutate(cc, cs)
  for i in range(frames):
    t.update_state(cc, cs)
    if i % 2 == 0:                        # the interceptor runs at 50 Hz
      t.observe_pedal(cc, cs, pedal(i), "v2")
  return t


t = counted(GearShifter.drive, False)
check("a steady pedal in D is counted in D", t.mode_admitted["D"] > 1000 and t.mode_admitted["ECON"] == 0
      and t.mode_admitted["S"] == 0, f"{t.mode_admitted}")
check("and its engaged seconds land in D", 30.0 <= t.mode_seconds["D"] <= 32.0
      and t.mode_seconds["ECON"] == 0.0, f"{t.mode_seconds}")
check("the gas law tag reaches the log fields", t.debug_values()["gas_law"] == "v2")
t = counted(GearShifter.drive, True)
check("ECON on counts in ECON", t.mode_admitted["ECON"] > 1000 and t.mode_admitted["D"] == 0, f"{t.mode_admitted}")
check("...even though the brake learner is frozen there", not t.mode_ok)
t = counted(GearShifter.sport, True)
check("S overrides ECON", t.mode_admitted["S"] > 1000 and t.mode_admitted["ECON"] == 0, f"{t.mode_admitted}")
t = counted(GearShifter.unknown, None)
check("unknown gear counts as D", t.mode_admitted["D"] > 1000, f"{t.mode_admitted}")

t = counted(pedal=lambda i: 0.20 + 0.05 * ((i // 20) % 2))
check("an unsteady pedal is not counted", t.mode_admitted["D"] == 0, f"{t.mode_admitted}")
t = counted(pedal=lambda i: 0.01)
check("a pedal at the brake end is not counted", t.mode_admitted["D"] == 0, f"{t.mode_admitted}")
t = counted(pedal=lambda i: 0.95)
check("a pedal near the top is not counted", t.mode_admitted["D"] == 0, f"{t.mode_admitted}")
t = counted(v=2.0)
check("below 3 m/s nothing is counted", t.mode_admitted["D"] == 0, f"{t.mode_admitted}")
for label, mutate in [
  ("gas pressed", lambda c, s: setattr(s.out, "gasPressed", True)),
  ("brake pressed", lambda c, s: setattr(s.out, "brakePressed", True)),
  ("stock AEB", lambda c, s: setattr(s.out, "stockAeb", True)),
  ("long inactive", lambda c, s: setattr(c, "longActive", False)),
  ("not in PID state", lambda c, s: setattr(c.actuators, "longControlState", LongCtrlState.stopping)),
  ("pose stale", lambda c, s: setattr(c, "orientationNED", [])),
  ("steep pitch", lambda c, s: setattr(c, "orientationNED", [0.0, 0.15, 0.0])),
]:
  t = counted(mutate=mutate)
  check(f"nothing is counted while {label}", t.mode_admitted["D"] == 0, f"{t.mode_admitted}")
t = counted(mutate=lambda c, s: setattr(c, "longActive", False))
check("and no engaged seconds while disengaged", t.mode_seconds["D"] < 3.0, f"{t.mode_seconds}")

# robustness: it runs inside CarController.update(), so odd input must never raise
t = make_tuner()
cc, cs = base(12.0, 0.5)
for bad in (float("nan"), float("inf"), None, "x", -5.0):
  try:
    t.observe_pedal(cc, cs, bad, "v2")
    t.observe_pedal(None, None, bad, None)
    ok = True
  except Exception:
    ok = False
  check(f"observe_pedal never raises on {bad!r}", ok)


# --- 5. brake learner --------------------------------------------------------

print("\n[5] brake learner")
t = make_tuner()
cc, cs = base(10.0, -2.0, -1.0)
settle(t, cc, cs)
for _ in range(1500):
  t.update_state(cc, cs)
  g = t.brake_gain(cc, cs, 0.5)
check("brake gain rises when under-braking", g > 1.0)
check("brake gain respects pos_limit", g <= 1.0 + BRAKE_POS_LIMIT + 1e-6, f"{g}")

# REGRESSION (review finding 1): a normal deceleration must not rail the gain on
# pure hydraulic lag. Model the actuator as a first-order lag on aEgo.
t = make_tuner()
cc, cs = base(15.0, 0.0, 0.0)
tau_alpha = 0.02          # ~0.5 s time constant at 100 Hz
peak = 1.0
for i in range(6000):
  cc.actuators.accel = -1.5 if (i // 600) % 2 == 0 else 0.0   # repeated brake applications
  t.update_state(cc, cs)
  cs.out.aEgo += tau_alpha * (cc.actuators.accel - cs.out.aEgo)   # plant tracks perfectly, just late
  peak = max(peak, t.brake_gain(cc, cs, 0.35))
check("lag alone does not rail the brake gain", peak < 1.0 + BRAKE_POS_LIMIT - 0.05, f"peak={peak:.3f}")
check("lag alone does not poison the converged estimate", t.brake_gain_converged < 0.2,
      f"conv={t.brake_gain_converged:.3f}")

# REGRESSION: the worst case, expressed in brake counts on the wire
worst_frac = 0.35
worst_counts = int(np.clip(worst_frac * (1.0 + BRAKE_POS_LIMIT) * NIDEC_BRAKE_MAX, 0, NIDEC_BRAKE_MAX - 1))
stock_counts = int(worst_frac * NIDEC_BRAKE_MAX)
check("max learned gain cannot saturate the brake command",
      worst_counts < NIDEC_BRAKE_MAX - 1, f"{stock_counts} -> {worst_counts} of {NIDEC_BRAKE_MAX}")

# REGRESSION (review finding 1): a railed episode must not survive a stop
t = make_tuner()
cc, cs = base(15.0, -2.0, 0.5)     # huge sustained error, drives the integrator to the rail
settle(t, cc, cs)
for _ in range(3000):
  t.update_state(cc, cs)
  t.brake_gain(cc, cs, 0.9)
railed = t.brake_pid.i
cs.out.vEgo, cs.out.aEgo = 0.0, 0.0
t.update_state(cc, cs)
after_stop = t.brake_gain(cc, cs, 0.9)
check("railed integrator does not survive a standstill",
      after_stop < 1.0 + 0.1, f"railed={railed:.3f} after_stop={after_stop:.3f}")
check("railed value is never written to the converged estimate", t.brake_gain_converged < 0.1,
      f"conv={t.brake_gain_converged:.3f}")

# REGRESSION (review finding 3): the stopping phase must not apply a wound gain open-loop
t = make_tuner()
cc, cs = base(15.0, -2.0, 0.5)
settle(t, cc, cs)
for _ in range(3000):
  t.update_state(cc, cs)
  t.brake_gain(cc, cs, 0.9)
cc.actuators.longControlState = LongCtrlState.stopping
t.update_state(cc, cs)
g_stopping = t.brake_gain(cc, cs, 0.9)
check("stopping caps the gain at the converged value", g_stopping <= 1.0 + t.brake_gain_converged + 1e-9,
      f"{g_stopping:.3f}")

# over-braking. The gain is two-sided on purpose -- it exists to trim the /2.6
# divisor in compute_gb_honda_elesys() live, and a one-sided gain could only ever
# add brake -- but the floor is deliberately much tighter than the ceiling.
t = make_tuner()
cc, cs = base(10.0, -1.0, -3.0)
settle(t, cc, cs)
for _ in range(3000):
  t.update_state(cc, cs)
  g = t.brake_gain(cc, cs, 0.5)
check("sustained over-braking does reduce the gain", g < 1.0, f"{g:.4f}")
check("but never below the floor", g >= 1.0 - dt.BRAKE_NEG_LIMIT - 1e-9,
      f"{g:.4f} vs floor {1.0 - dt.BRAKE_NEG_LIMIT:.2f}")
check("the floor is much tighter than the ceiling (under-braking is the worse failure)",
      dt.BRAKE_NEG_LIMIT < BRAKE_POS_LIMIT / 2,
      f"neg {dt.BRAKE_NEG_LIMIT} vs pos {BRAKE_POS_LIMIT}")
check("a railed-low gain does not reach the converged estimate",
      t.brake_gain_converged > -dt.BRAKE_NEG_LIMIT + 1e-6, f"{t.brake_gain_converged:.5f}")
# and it comes back up when the car is under-braking again
cc, cs = base(10.0, -1.0, -0.2)
settle(t, cc, cs)
for _ in range(3000):
  t.update_state(cc, cs)
  g_up = t.brake_gain(cc, cs, 0.5)
check("and it recovers upward once the error flips", g_up > g, f"{g:.4f} -> {g_up:.4f}")

for label, mutate in [
  ("stock AEB", lambda c, s: setattr(s.out, "stockAeb", True)),
  ("brake pressed", lambda c, s: setattr(s.out, "brakePressed", True)),
  ("gas pressed", lambda c, s: setattr(s.out, "gasPressed", True)),
  ("not in PID state", lambda c, s: setattr(c.actuators, "longControlState", LongCtrlState.starting)),
  ("long inactive", lambda c, s: setattr(c, "longActive", False)),
]:
  t = make_tuner()
  cc, cs = base(10.0, -2.0, -1.0)
  settle(t, cc, cs)
  mutate(cc, cs)
  before = t.brake_pid.i
  for _ in range(1000):
    t.update_state(cc, cs)
    t.brake_gain(cc, cs, 0.5)
  check(f"brake integrator frozen while {label}", abs(t.brake_pid.i - before) < 1e-9)

t = make_tuner()
cc, cs = base(0.5, -2.0, -1.0)      # below BRAKE_LEARN_MIN_SPEED
settle(t, cc, cs)
before = t.brake_pid.i
for _ in range(2000):
  t.update_state(cc, cs)
  t.brake_gain(cc, cs, 0.5)
check("brake integrator frozen below the learn floor", abs(t.brake_pid.i - before) < 1e-9)

t = make_tuner()
cc, cs = base(10.0, -2.0, -1.0)
before = t.brake_pid.i
for _ in range(30):                 # dwell not yet satisfied
  t.update_state(cc, cs)
  t.brake_gain(cc, cs, 0.5)
check("brake integrator frozen before the dwell opens", abs(t.brake_pid.i - before) < 1e-9)


# --- 10. params: load clamping, unknown keys, persistence -------------------

print("\n[10] params handling")
known = set(dt._PARAM_SPEC) | {"HondaDynamicTuningEnabled"}

# REGRESSION (review finding 14): corrupted / hand-edited values must be clamped
p = FakeParams({"HondaDynamicTuningEnabled": True, "HondaDynBrakeGain": 10.0,
                "HondaDynModeSecD": -99.0, "HondaDynModeSecS": 1e30}, known)
t = make_tuner(params=p)
check("out-of-range brake gain is clamped on load", t.brake_pid.i <= BRAKE_POS_LIMIT + 1e-9,
      f"{t.brake_pid.i}")
check("negative and absurd mode totals are clamped on load",
      t.mode_seconds_loaded["D"] == 0.0 and t.mode_seconds_loaded["S"] <= dt.MODE_SEC_MAX,
      f"{t.mode_seconds_loaded}")
cc, cs = base(10.0, -2.0, -1.0)
check("clamped brake gain reaches the output bounded",
      t.brake_gain(cc, cs, 1.0) <= 1.0 + BRAKE_POS_LIMIT + 1e-9)

p = FakeParams({"HondaDynamicTuningEnabled": True, "HondaDynBrakeGain": float("nan"),
                "HondaDynModeSecECON": float("inf")}, known)
t = make_tuner(params=p)
check("NaN param falls back to the default", t.brake_gain_converged == 0.0, f"{t.brake_gain_converged}")
check("inf param falls back to the default", t.mode_seconds_loaded["ECON"] == 0.0, f"{t.mode_seconds_loaded}")

# unknown keys (params_keys.h not updated) must degrade, not crash
p = FakeParams({}, known=set())
t = make_tuner(params=p)
check("unregistered keys degrade to disabled without raising", t.enabled is False)
check("no writer thread started when disabled", t._writer is None)

# persistence writes the CONVERGED brake value only, and the mode totals as loaded + this drive
p = FakeParams({"HondaDynamicTuningEnabled": True, "HondaDynModeSecECON": 87.0}, known)
t = make_tuner(params=p)
t.brake_pid_factor, t.brake_gain_converged = 0.6, 0.12
t.mode_seconds["ECON"] = 13.0
t.persist(dt.PERSIST_INTERVAL)
# the writer is a daemon thread with no task_done()/join() contract; poll instead
import time as _time
for _ in range(100):
  if "HondaDynBrakeGain" in p.written and "HondaDynModeSecS" in p.written:
    break
  _time.sleep(0.01)
check("persist writes the converged brake gain, not the live one",
      abs(p.written.get("HondaDynBrakeGain", -1) - 0.12) < 1e-9, str(p.written.get("HondaDynBrakeGain")))
check("persist writes the mode totals as loaded + this drive",
      p.written.get("HondaDynModeSecECON") == 100.0 and p.written.get("HondaDynModeSecD") == 0.0,
      str({k: v for k, v in p.written.items() if k.startswith("HondaDynModeSec")}))
check("every persisted value is a plain float (put_many runs float() in the control thread)",
      all(type(v) is float for v in p.written.values()), str({k: type(v).__name__ for k, v in p.written.items()}))
check("no retired key is written", not any(k.startswith(("HondaDynPedalGain", "HondaDynWindFactor")) for k in p.written))
t2 = make_tuner(params=FakeParams({"HondaDynamicTuningEnabled": True}, known))
t2.persist(dt.PERSIST_INTERVAL + 1)
check("persist is a no-op off the interval", len(t2._writer._queue.queue) == 0)


# --- 11. standalone / no-openpilot ------------------------------------------

print("\n[11] standalone import")
dt._open_params = lambda: None
t = HondaDynamicTuner(CP=None, CP_SP=None)   # no helper override: the real default path
check("constructs with no openpilot Params available", t is not None)
check("defaults to disabled without params", t.enabled is False)
check("no writer thread without params", t._writer is None)
check("persist() is a no-op without params", t.persist(dt.PERSIST_INTERVAL) is None)
check("debug_values() works", isinstance(t.debug_values(), dict))


# --- 12. second-review regressions ------------------------------------------

print("\n[12] second-review regressions")

# The original dwell gate compared frame-to-frame, i.e. a 20 m/s^3 jerk threshold,
# so it never closed on a realistic RAMPED target and the brake integrator ate
# pure hydraulic lag all the way to the rail. Step targets pass either way -- this
# uses the ramp shape a real planner produces.
t = make_tuner()
cc, cs = base(15.0, 0.0, 0.0)
peak, target = 1.0, 0.0
for i in range(60000):
  want = -2.0 if (i // 3000) % 2 == 0 else 0.0
  target += float(np.clip(want - target, -0.02, 0.02))     # 2 m/s^3 ramp
  cc.actuators.accel = target
  t.update_state(cc, cs)
  cs.out.aEgo += 0.02 * (target - cs.out.aEgo)             # perfect plant, 0.5 s lag only
  peak = max(peak, t.brake_gain(cc, cs, max(0.0, -target) * 0.2))
check("ramped target: lag alone does not rail the brake gain",
      peak < 1.0 + BRAKE_POS_LIMIT - 0.05, f"peak={peak:.4f}")
check("ramped target: lag alone does not reach disk",
      t.brake_gain_converged < 0.1, f"conv={t.brake_gain_converged:.4f}")

# A rolling stop enters and leaves `stopping` without ever reaching standstill.
# Clamping only the output left the integrator wound and handed it straight back.
t = make_tuner()
cc, cs = base(15.0, -2.0, 0.5)
settle(t, cc, cs)
for _ in range(3000):
  t.update_state(cc, cs)
  t.brake_gain(cc, cs, 0.9)
cc.actuators.longControlState = LongCtrlState.stopping
cs.out.vEgo = 1.5
for _ in range(200):
  t.update_state(cc, cs)
  t.brake_gain(cc, cs, 0.9)
cc.actuators.longControlState = LongCtrlState.pid
cs.out.vEgo = 2.0
t.update_state(cc, cs)
rolling = t.brake_gain(cc, cs, 0.9)
check("rolling stop does not hand back the wound gain", rolling <= 1.0 + 0.1, f"{rolling:.3f}")

# Pitch must fade out at standstill: on an uphill the term asks for less brake,
# which is right while decelerating and wrong while holding the car on a hill.
t = make_tuner()
cc, cs = base(0.0, -0.5)
cc.orientationNED = [0.0, 0.10, 0.0]          # ~10% uphill
for _ in range(2000):
  hill_stopped = t.update_state(cc, cs)
check("pitch feedforward is zero at standstill", abs(hill_stopped) < 1e-9, f"{hill_stopped}")
cc.actuators.longControlState = LongCtrlState.stopping
cs.out.vEgo = 8.0
for _ in range(500):
  hill_stopping = t.update_state(cc, cs)
check("pitch feedforward is zero during stopping", abs(hill_stopping) < 1e-9, f"{hill_stopping}")
cc.actuators.longControlState = LongCtrlState.pid
for _ in range(2000):
  hill_moving = t.update_state(cc, cs)
check("pitch feedforward is live while cruising", hill_moving > 0.5, f"{hill_moving}")

# Params that come back as a numpy scalar / string must still load.
p = FakeParams({"HondaDynamicTuningEnabled": True,
                "HondaDynBrakeGain": np.float64(0.25),
                "HondaDynModeSecS": "12.5"}, known)
t = make_tuner(params=p)
check("numpy scalar param loads", abs(t.brake_gain_converged - 0.25) < 1e-9, f"{t.brake_gain_converged}")
check("string param loads", abs(t.mode_seconds_loaded["S"] - 12.5) < 1e-9, f"{t.mode_seconds_loaded}")
check("registry default of 0.0 means no day-one brake gain",
      make_tuner(params=FakeParams({"HondaDynamicTuningEnabled": True}, known)).brake_gain_converged == 0.0)


# --- 13. log-derived regressions --------------------------------------------

print("\n[13] log-derived regressions")

# Measured plant lag on the real routes is 0.23-0.34 s. Re-run the ramp case at
# the measured value as well as at the more pessimistic 0.5 s already covered.
for tau_s, alpha in (("0.30 s (measured)", 0.033), ("0.50 s (pessimistic)", 0.02)):
  t = make_tuner()
  cc, cs = base(15.0, 0.0, 0.0)
  peak, target = 1.0, 0.0
  for i in range(60000):
    want = -2.0 if (i // 3000) % 2 == 0 else 0.0
    target += float(np.clip(want - target, -0.02, 0.02))
    cc.actuators.accel = target
    t.update_state(cc, cs)
    cs.out.aEgo += alpha * (target - cs.out.aEgo)
    peak = max(peak, t.brake_gain(cc, cs, max(0.0, -target) * 0.2))
  check(f"lag of {tau_s} does not rail the brake gain",
        peak < 1.0 + BRAKE_POS_LIMIT - 0.05, f"peak={peak:.4f}")

# Small commands carry no identifiable gain information -- at settled cruise the
# residual is grade, not brake gain.
t = make_tuner()
cc, cs = base(10.0, -(dt.LEARN_MIN_CMD - 0.05), 0.5)
settle(t, cc, cs)
before = t.brake_pid.i
for _ in range(5000):
  t.update_state(cc, cs)
  t.brake_gain(cc, cs, 0.5)
check("no brake learning below the command floor", abs(t.brake_pid.i - before) < 1e-9)

# The persisted estimate must never advance while the live value sits on a clamp,
# no matter how long the excursion lasts.
t = make_tuner()
cc, cs = base(15.0, -2.0, 0.5)
settle(t, cc, cs)
for _ in range(40000):
  t.update_state(cc, cs)
  t.brake_gain(cc, cs, 0.9)
check("brake gain reached its rail", t.brake_pid_factor >= BRAKE_POS_LIMIT - 1e-6)
# the few frames spent climbing to the rail are legitimately off-rail, so a tiny
# amount leaks in; what matters is that it stays negligible and then stops
conv_at_rail = t.brake_gain_converged
check("railed brake gain stays negligible in the converged estimate",
      conv_at_rail < 0.05 * BRAKE_POS_LIMIT, f"{conv_at_rail:.5f} of {BRAKE_POS_LIMIT}")
for _ in range(80000):
  t.update_state(cc, cs)
  t.brake_gain(cc, cs, 0.9)
check("converged brake estimate stops advancing once railed",
      abs(t.brake_gain_converged - conv_at_rail) < 1e-9,
      f"{conv_at_rail:.6f} -> {t.brake_gain_converged:.6f}")


# --- 14. third-review regressions --------------------------------------------
#
# Each of these reproduced against the real module before the fix; they are the
# failure, not the feature. Do not relax one without re-deriving why it is here.

print("\n[14] third-review regressions")

# (a) nothing wound up in one engagement may cross into the next. brake_gain()'s
# only unwind paths were the stopping clamp and the standstill reset, and neither
# fires on a plain disengage at speed -- so a gain railed on a downgrade came back
# open-loop on the next engagement and stayed for the whole 1.5 s dwell.
t = make_tuner()
cc, cs = base(15.0, -1.5, -1.0)
settle(t, cc, cs)
for _ in range(2000):
  t.update_state(cc, cs)
  t.brake_gain(cc, cs, 0.5)
wound = t.brake_pid_factor
check("brake gain does wind up while learning", wound > 0.5 * BRAKE_POS_LIMIT, f"{wound:.4f}")

off_cc, off_cs = base(15.0, 0.0, 0.0)
off_cc.longActive = False
off_cc.actuators.longControlState = LongCtrlState.off
for _ in range(300):
  t.update_state(off_cc, off_cs)
  t.brake_gain(off_cc, off_cs, 0.0)
re_cc, re_cs = base(15.0, -0.5, 0.0)
first = t.brake_gain(re_cc, re_cs, 0.4)
check("a disengage unwinds the brake integrator to the converged estimate",
      abs(first - (1.0 + t.brake_gain_converged)) < 1e-9,
      f"first frame after re-engage {first:.4f}, converged says {1 + t.brake_gain_converged:.4f}")
check("the railed value does not survive the disengage",
      first < 1.0 + 0.1 * BRAKE_POS_LIMIT, f"{first:.4f}")

# (b) the standstill hold is the one brake command with no feedback, and it is
# what interface.py's stopAccel was hand-tuned against. A learned gain must not
# reach it -- 0.33 put the measured hold back at cb 251 and 0.50 railed it.
t = make_tuner()
t.brake_gain_converged = 0.5
t.brake_pid.i = t.brake_pid_factor = 0.5
hold_cc, hold_cs = base(0.0, -0.8, 0.0)
hold_cc.actuators.longControlState = LongCtrlState.stopping
t.update_state(hold_cc, hold_cs)
check("no learned gain is applied to the standstill hold",
      t.brake_gain(hold_cc, hold_cs, 0.75) == 1.0,
      f"{t.brake_gain(hold_cc, hold_cs, 0.75):.4f}")
check("the standstill reset still re-arms the estimate for the next stop",
      abs(t.brake_pid.i - t.brake_gain_converged) < 1e-9, f"{t.brake_pid.i:.4f}")
# ... and the gain is still live on the approach, which is where it earns its keep
t2 = make_tuner()
t2.brake_gain_converged = 0.5
t2.brake_pid.i = t2.brake_pid_factor = 0.5
app_cc, app_cs = base(3.0, -1.5, -1.0)
app_cc.actuators.longControlState = LongCtrlState.stopping
t2.update_state(app_cc, app_cs)
check("the gain is still applied while still moving in the stopping phase",
      t2.brake_gain(app_cc, app_cs, 0.6) > 1.4, f"{t2.brake_gain(app_cc, app_cs, 0.6):.4f}")

# --- 15. drive-mode gating ----------------------------------------------------
#
# S holds lower gears (more engine braking) and ECON remaps the throttle. One gain
# pooled across modes converges on a blend that matches no actual driving, which is
# worse than not learning -- so the brake learner learns in D with ECON off only.

print("\n[15] drive-mode gating")


def learn_in(gear, econ=None, frames=3000):
  t = make_tuner()
  cc, cs = base(15.0, -1.5, -1.0)          # under-braking: the brake gain rises where it may
  cs.out.gearShifter = gear
  cs.econ_on = econ
  settle(t, cc, cs)
  for _ in range(frames):
    t.update_state(cc, cs)
    t.brake_gain(cc, cs, 0.5)
  return t


t_d = learn_in(GearShifter.drive)
check("learning runs in D", t_d.brake_pid_factor > 0.0, f"{t_d.brake_pid_factor:.4f}")
t_s = learn_in(GearShifter.sport)
check("and mode_ok reports why S is frozen", not t_s.mode_ok and t_s.drive_mode == ("sport", None),
      f"mode_ok={t_s.mode_ok} mode={t_s.drive_mode}")
t_e = learn_in(GearShifter.drive, econ=True)
check("learning is frozen in ECON (decoded on this car since 0x221 was mapped)",
      t_e.brake_pid_factor == 0.0 and t_e.drive_mode == ("drive", True), f"{t_e.brake_pid_factor} {t_e.drive_mode}")
check("ECON off learns", learn_in(GearShifter.drive, econ=False).brake_pid_factor > 0.0)
check("ECON gating keys off CS.econ_on, and is inert where it is not decoded",
      dt.HondaDynamicTuner._econ_state(base(10.0)[1]) is None
      and not dt.HondaDynamicTuner._mode_learnable(("drive", True))
      and dt.HondaDynamicTuner._mode_learnable(("drive", False))
      and dt.HondaDynamicTuner._mode_learnable(("drive", None)))

# unknown gear must NOT gate -- every other Nidec car leaves gearShifter unknown,
# and gating there would silently disable the whole feature on those platforms
t_u = learn_in(GearShifter.unknown)
check("unknown gear still learns (other Nidec platforms)",
      t_u.brake_pid_factor > 0.0 and t_u.mode_ok, f"{t_u.brake_pid_factor:.4f} mode_ok={t_u.mode_ok}")

# brake channel is gated too
t = make_tuner()
cc, cs = base(15.0, -1.5, -1.0)
cs.out.gearShifter = GearShifter.sport
settle(t, cc, cs)
for _ in range(3000):
  t.update_state(cc, cs)
  t.brake_gain(cc, cs, 0.5)
check("brake learning is frozen in S too", t.brake_pid_factor == 0.0, f"{t.brake_pid_factor:.4f}")

# a mode change is a transient: the dwell must re-arm, not carry across
t = make_tuner()
cc, cs = base(10.0, 1.0, 0.7)
cs.out.gearShifter = GearShifter.drive
settle(t, cc, cs)
check("settled in D before the shift", t._settle >= SETTLE_FRAMES, f"{t._settle}")
cs.out.gearShifter = GearShifter.sport
t.update_state(cc, cs)
check("a gear change resets the dwell", t._settle == 0, f"{t._settle}")
cs.out.gearShifter = GearShifter.drive
t.update_state(cc, cs)
check("and again on the way back", t._settle == 0, f"{t._settle}")


# --- 16. the hondadyn log line ------------------------------------------------

print("\n[16] hondadyn log line")
lines = []
with mock.patch.object(dt.carlog, "info", lambda msg, *a, **k: lines.append(msg)):
  t = counted(GearShifter.drive, True, frames=600)
  t.log_state(0)
line = lines[-1] if lines else ""
check("one line, tagged and carrying the gas law and the slot",
      line.startswith("hondadyn gaslaw=v2 slot=ECON "), line[:80])
check("per-mode lists in D, ECON, S order", all(f"{k}=[" in line for k in ("modesec", "modeadm", "modetot")), line)
check("the retired fields are gone", "pedal=" not in line and "wind=" not in line, line)


print("\n" + "=" * 60)
if FAILURES:
  print(f"{len(FAILURES)} FAILED: {FAILURES}")
  sys.exit(1)
print("ALL CHECKS PASSED")
