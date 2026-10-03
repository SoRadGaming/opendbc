"""FORK(HONDA_ACCORD_9G_AU): the shadow longitudinal learners (shadow_learn.py).

What has to hold before this goes in a car:
  * NOTHING ACTUATED CHANGES. The real CarController, driven through braking, coasting, launches, disengages and
    noise, sends byte-identical CAN with the shadow on and with it removed, and the tuner's own state and outputs
    are identical too.
  * It never raises into CarController, whatever it is fed.
  * The gates admit what they are meant to and nothing else, into the right cell.
  * What it reports it would apply stays inside its bounds: the brake table only ever adds braking, at most
    0.5 m/s^2, the launch multiplier only ever takes pedal away, at most 40%.
  * The log line is bounded, parseable, and comes out once a minute and at a disengage, not per frame.
"""
import math
import random
import unittest
from dataclasses import dataclass, field
from unittest import mock

from opendbc.car import structs
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.carcontroller import CarController
from opendbc.car.honda.values import CAR
from opendbc.sunnypilot.car.honda import dynamic_tuning as dt
from opendbc.sunnypilot.car.honda import elesys_gas as eg
from opendbc.sunnypilot.car.honda import shadow_learn as sl
from opendbc.sunnypilot.car.honda.shadow_learn import (
  HondaShadowLearners, brake_cell_correction, launch_multiplier, speed_band, count_band, launch_band,
)

LongCtrlState = structs.CarControl.Actuators.LongControlState
G = 9.81
PLATFORM = CAR.HONDA_ACCORD_9G_AU


@dataclass
class Act:
  accel: float = 0.0
  longControlState: object = LongCtrlState.pid


@dataclass
class FakeCC:
  longActive: bool = True
  actuators: Act = field(default_factory=Act)


@dataclass
class Out:
  vEgo: float = 10.0
  aEgo: float = 0.0
  gasPressed: bool = False
  brakePressed: bool = False
  stockAeb: bool = False


@dataclass
class FakeCS:
  out: Out = field(default_factory=Out)


def feed(sh, n, v=10.0, a=-1.0, ref=-1.0, counts=80.0, gas=0.0, pitch=0.0, pose_fresh=True,
         mode_ok=True, long_active=True, state=LongCtrlState.pid, **out):
  cc = FakeCC(long_active, Act(ref, state))
  cs = FakeCS(Out(vEgo=v, aEgo=a, **out))
  for _ in range(n):
    sh.update(cc, cs, pitch=pitch, pose_fresh=pose_fresh, mode_ok=mode_ok, cmd_ref=ref,
              brake_frac=counts / sl.NIDEC_BRAKE_MAX, gas_cmd=gas)


def totals(sh):
  return (sum(c.n for row in sh.brake for c in row), sum(c.n for c in sh.coast_acc), sum(r.n for r in sh.launch))


def parse(line):
  out = {}
  for tok in line.split()[1:]:
    k, v = tok.split("=", 1)
    out[k] = [float(x) for x in v.strip("[]").split(",")] if v.startswith("[") else float(v)
  return out


class TestBandsAndBounds(unittest.TestCase):
  def test_bands(self):
    self.assertEqual(speed_band(0.99), -1)
    self.assertEqual(speed_band(1.0), 0)
    self.assertEqual(speed_band(4.99), 0)
    self.assertEqual(speed_band(5.0), 1)
    self.assertEqual(speed_band(40.0), len(sl.SPEED_BP) - 1)
    self.assertEqual([count_band(c) for c in (4, 60, 60.1, 100, 100.1, 255)], [0, 0, 1, 1, 2, 2])
    self.assertEqual([launch_band(v) for v in (0.4, 0.5, 2.99, 3.0, 5.99, 6.0)], [-1, 0, 0, 1, 1, -1])

  def test_brake_correction_only_adds_braking_and_is_capped(self):
    n = sl.MIN_CELL_SAMPLES
    self.assertAlmostEqual(brake_cell_correction(n, 0.2), -0.2)   # under-braking -> more brake
    self.assertEqual(brake_cell_correction(n, 2.0), sl.BRAKE_CORR_MIN)
    self.assertEqual(brake_cell_correction(n, -0.3), 0.0)          # over-braking: never less than the law
    self.assertEqual(brake_cell_correction(n - 1, 0.3), 0.0)       # not enough evidence
    self.assertEqual(brake_cell_correction(n, float("nan")), 0.0)

  def test_launch_multiplier_only_takes_pedal_away(self):
    n = sl.LAUNCH_MIN_SAMPLES
    self.assertAlmostEqual(launch_multiplier(n, 1.4, 1.0), 1 / 1.4)
    self.assertEqual(launch_multiplier(n, 3.0, 1.0), sl.LAUNCH_MULT_MIN)
    self.assertEqual(launch_multiplier(n, 0.7, 1.0), 1.0)           # under-delivery: never adds pedal
    self.assertEqual(launch_multiplier(n - 1, 1.4, 1.0), 1.0)
    self.assertEqual(launch_multiplier(n, 0.0, 0.0), 1.0)

  def test_applicable_only_on_the_elesys_accord(self):
    self.assertTrue(sl.shadow_applicable(CarInterface.get_non_essential_params(PLATFORM)))
    self.assertFalse(sl.shadow_applicable(CarInterface.get_non_essential_params(CAR.HONDA_CIVIC)))
    self.assertFalse(sl.shadow_applicable(None))


class TestGates(unittest.TestCase):
  def test_steady_brake_lands_in_its_cell_with_gravity_removed(self):
    sh = HondaShadowLearners()
    # 2% downhill (nose-down, pitch < 0): the car shows -1.0 but the brakes delivered -1.0 + 0.196
    pitch = -0.02
    feed(sh, sl.WINDOW + 49, v=12.0, a=-1.0, ref=-1.0, counts=80.0, pitch=pitch)
    cell = sh.brake[speed_band(12.0)][count_band(80.0)]
    self.assertEqual(cell.n, 50)
    self.assertAlmostEqual(cell.mean(), G * math.sin(pitch), places=2)
    self.assertEqual(totals(sh), (50, 0, 0))

  def test_every_gate_closes(self):
    closed = {
      "gas pressed": dict(gasPressed=True), "brake pressed": dict(brakePressed=True), "stock AEB": dict(stockAeb=True),
      "pose stale": dict(pose_fresh=False), "ECON/S": dict(mode_ok=False),
      "steep": dict(pitch=0.05), "disengaged": dict(long_active=False), "stopping": dict(state=LongCtrlState.stopping),
      "below 1 m/s": dict(v=0.9), "pedal and brake both": dict(gas=0.1),
    }
    for name, kw in closed.items():
      with self.subTest(name):
        sh = HondaShadowLearners()
        feed(sh, 200, **kw)
        self.assertEqual(totals(sh)[0], 0, name)

  def test_jerk_gate_on_the_plant_model(self):
    sh = HondaShadowLearners()
    cc, cs = FakeCC(), FakeCS(Out(vEgo=12.0, aEgo=-1.0))
    for i in range(300):     # the command ramping at 1 m/s^3 (0.02 per 50 Hz sample): never steady
      sh.update(cc, cs, pitch=0.0, pose_fresh=True, mode_ok=True, cmd_ref=-0.5 - 0.02 * (i % 50),
                brake_frac=80.0 / sl.NIDEC_BRAKE_MAX, gas_cmd=0.0)
    self.assertEqual(totals(sh), (0, 0, 0))
    feed(sh, sl.STEADY_HOLD - 1)   # held steady, but not yet for STEADY_HOLD samples
    self.assertEqual(totals(sh), (0, 0, 0))
    feed(sh, 2)
    self.assertEqual(totals(sh)[0], 1)

  def test_brake_must_hold_steady_for_the_whole_window(self):
    sh = HondaShadowLearners()
    cc, cs = FakeCC(), FakeCS(Out(vEgo=12.0, aEgo=-1.0))
    for i in range(300):     # alternates 70 / 95 counts: same band, but 25 counts apart
      counts = 70.0 if i % 2 else 95.0
      sh.update(cc, cs, pitch=0.0, pose_fresh=True, mode_ok=True, cmd_ref=-1.0,
                brake_frac=counts / sl.NIDEC_BRAKE_MAX, gas_cmd=0.0)
    self.assertEqual(totals(sh), (0, 0, 0))
    sh = HondaShadowLearners()
    feed(sh, sl.WINDOW - 1, counts=50.0)     # the window still holds a <=60 command...
    feed(sh, 10, counts=65.0)                # ...so a 65-count command crosses bands: not admitted
    self.assertEqual(totals(sh), (0, 0, 0))

  def test_coast(self):
    sh = HondaShadowLearners()
    feed(sh, sl.WINDOW + 99, v=20.0, a=-0.35, ref=-0.2, counts=0.0)
    b = speed_band(20.0)
    self.assertEqual(sh.coast_acc[b].n, 100)
    self.assertAlmostEqual(sh.coast_acc[b].mean(), -0.35)
    self.assertAlmostEqual(sh.coast_err[b].mean(), -0.15)
    sh = HondaShadowLearners()
    feed(sh, 200, v=2.5, a=-0.3, ref=-0.2, counts=0.0)   # below COAST_MIN_SPEED
    self.assertEqual(totals(sh), (0, 0, 0))

  def test_launch(self):
    sh = HondaShadowLearners()
    feed(sh, sl.WINDOW + 149, v=2.0, a=1.4, ref=1.0, counts=0.0, gas=0.15)
    self.assertEqual(sh.launch[0].n, 150)
    self.assertAlmostEqual(sh.launch[0].ratio(), 1.4)
    self.assertAlmostEqual(launch_multiplier(*sh.launch_pooled()), 1 / 1.4)
    self.assertEqual(sh.launch_episodes, 1)
    for name, kw in {"weak demand": dict(ref=0.2), "brake in the window": dict(counts=5.0),
                     "pedal not pressing": dict(gas=0.0), "above 6 m/s": dict(v=6.5), "creep": dict(v=0.3),
                     "steeper than 4 deg": dict(pitch=0.075)}.items():
      with self.subTest(name):
        sh = HondaShadowLearners()
        args = dict(v=2.0, a=1.4, ref=1.0, counts=0.0, gas=0.15) | kw
        feed(sh, 200, **args)
        self.assertEqual(totals(sh)[2], 0, name)

  def test_launch_on_a_grade_is_measured_with_gravity_removed(self):
    sh = HondaShadowLearners()     # route 115's launch: -2.7 deg, the car shows 1.88 for a 1.04 command
    pitch = math.radians(-2.7)
    feed(sh, sl.WINDOW + 149, v=2.0, a=1.88, ref=1.04, counts=0.0, gas=0.15, pitch=pitch)
    self.assertAlmostEqual(sh.launch[0].ratio(), (1.88 + G * math.sin(pitch)) / 1.04, places=3)
    self.assertEqual(totals(sh)[0], 0)   # and the same grade keeps it out of the brake table

  def test_launch_ramp_gate(self):
    sh = HondaShadowLearners()
    cc, cs = FakeCC(), FakeCS(Out(vEgo=2.0, aEgo=1.0))
    for i in range(200):     # cmd_ref ramping at 2 m/s^3: too fast for the lag model to be trusted
      sh.update(cc, cs, pitch=0.0, pose_fresh=True, mode_ok=True, cmd_ref=0.3 + 0.04 * i,
                brake_frac=0.0, gas_cmd=0.2)
    self.assertEqual(totals(sh), (0, 0, 0))

  def test_launch_episodes_counted_once_each(self):
    sh = HondaShadowLearners()
    for _ in range(3):
      feed(sh, 60, v=2.0, a=1.4, ref=1.0, counts=0.0, gas=0.15)
      feed(sh, 60, v=8.0, a=0.0, ref=0.0, counts=0.0, gas=0.1)
    self.assertEqual(sh.launch_episodes, 3)

  def test_brake_correction_fades_in_with_speed(self):
    sh = HondaShadowLearners()
    feed(sh, sl.WINDOW + sl.MIN_CELL_SAMPLES, v=4.0, a=-0.7, ref=-1.0, counts=50.0)
    self.assertAlmostEqual(sh.brake_correction(4.0, 50.0), -0.3, places=6)
    self.assertAlmostEqual(sh.brake_correction(1.5, 50.0), -0.15, places=6)
    self.assertEqual(sh.brake_correction(0.5, 50.0), 0.0)
    self.assertEqual(sh.brake_correction(4.0, 2.0), 0.0)


class TestLog(unittest.TestCase):
  def test_line_cadence_and_shape(self):
    lines = []
    with mock.patch.object(sl.carlog, "info", lambda msg, *a, **k: lines.append(msg)):
      sh = HondaShadowLearners()
      feed(sh, 2 * sl.LOG_INTERVAL, v=12.0, a=-0.8, ref=-1.0, counts=80.0)
      self.assertEqual(len(lines), 2)                        # once a minute while learning...
      feed(sh, 2 * sl.LOG_INTERVAL, long_active=False)
      self.assertEqual(len(lines), 3)                        # ...once at the disengage, then silent
      feed(sh, 5, v=12.0, a=-0.8, ref=-1.0, counts=80.0, gasPressed=True)   # engaged, nothing admitted
      feed(sh, 5, long_active=False)
      self.assertEqual(len(lines), 3)                        # nothing admitted in the window: no line
    d = parse(lines[-1])
    self.assertTrue(lines[-1].startswith("hondashadow v=1 "))
    self.assertLess(max(len(x) for x in lines), 1200)
    n_cells = len(sl.SPEED_BP) * (len(sl.BRAKE_COUNT_BP) + 1)
    for k in ("bn", "be", "bsd", "bcorr", "bcb", "bacc"):
      self.assertEqual(len(d[k]), n_cells, k)
    i = speed_band(12.0) * (len(sl.BRAKE_COUNT_BP) + 1) + count_band(80.0)
    self.assertEqual(d["bn"][i], 2 * sl.LOG_INTERVAL - sl.WINDOW + 1)
    self.assertAlmostEqual(d["be"][i], 0.2, places=3)
    self.assertAlmostEqual(d["bcorr"][i], -0.2, places=3)
    self.assertTrue(math.isnan(d["be"][0]))
    self.assertEqual(d["lmult"], 1.0)


# --- through the real CarController --------------------------------------------------------------

class _Params:
  def __init__(self, store):
    self.store = store

  def get(self, key, block=False, return_default=False):
    return self.store.get(key, dt._PARAM_SPEC[key][0] if key in dt._PARAM_SPEC else None)

  def get_bool(self, key, block=False):
    return bool(self.store.get(key, False))

  def put(self, key, val, block=False):
    self.store[key] = val


class _CS:
  def __init__(self):
    self.out = structs.CarState.new_message()
    self.out.cruiseState.speed = 30.0
    self.out.cruiseState.available = True
    self.v_cruise_factor = 1.0
    self.stock_brake = {"CHIME": 0, "AEB_REQ_1": 0, "AEB_REQ_2": 0, "AEB_STATUS": 0}
    self.acc_hud = {"FCM_OFF": 0, "FCM_OFF_2": 0, "FCM_PROBLEM": 0, "ICONS": 0}
    self.lkas_hud = {}
    self.scm_buttons = {"CRUISE_BUTTONS": 0, "CRUISE_SETTING": 0}
    self.is_metric = True


def _build(shadow: bool):
  params = _Params({"HondaDynamicTuningEnabled": True, eg.GAS_LAW_PARAM: True})
  with mock.patch.object(dt, "_open_params", lambda: params), mock.patch.object(eg, "_open_params", lambda: params):
    CP = CarInterface.get_non_essential_params(PLATFORM)
    CP_SP = CarInterface.get_non_essential_params_sp(CP, PLATFORM)
    CP.openpilotLongitudinalControl = True
    CP_SP.enableGasInterceptor = True
    cc = CarController(PLATFORM.config.dbc_dict, CP, CP_SP)
  if not shadow:
    cc.dynamic_tuner.shadow = None
  return cc


def _make_cc(accel, state, long_active, pitch):
  cc = structs.CarControl.new_message()
  cc.enabled = long_active
  cc.longActive = long_active
  cc.latActive = long_active
  cc.orientationNED = [0.0, pitch, 0.0]
  cc.actuators.accel = accel
  cc.actuators.torque = 0.2 * math.sin(accel)
  cc.actuators.longControlState = state
  cc.hudControl.speedVisible = True
  cc.hudControl.setSpeed = 30.0
  return cc.as_reader()


def _scenario(seed: int, frames: int = 9000):
  """Launches, cruise, coasting, braking, a stop, disengages, pedal presses, pitch, noise."""
  rng = random.Random(seed)
  out, v, a = [], 0.0, 0.0
  target, state, active, pitch = 0.0, LongCtrlState.pid, True, 0.0
  for i in range(frames):
    if i % 300 == 0:
      target = rng.choice([1.8, 1.0, 0.5, 0.0, -0.3, -0.8, -1.5, -2.5])
      state = LongCtrlState.stopping if (target < -1 and v < 2) else LongCtrlState.pid
      active = rng.random() > 0.1
      pitch = rng.uniform(-0.04, 0.04)
    a += ((target if active else -0.3) * rng.uniform(0.6, 1.4) - a) * 0.03
    v = max(0.0, v + a * 0.01)
    out.append(dict(accel=target + rng.gauss(0, 0.05), state=state, active=active, pitch=pitch + rng.gauss(0, 0.002),
                    v=v, a=a + rng.gauss(0, 0.05), gas=rng.random() < 0.01, brake=rng.random() < 0.01))
  return out


class TestNothingActuatedChanges(unittest.TestCase):
  def test_can_and_tuner_identical_with_and_without_shadow(self):
    lines = []
    with mock.patch.object(sl.carlog, "info", lambda msg, *a, **k: lines.append(msg)):
      for seed in (1, 2, 3):
        with_sh, without = _build(True), _build(False)
        self.assertIsNotNone(with_sh.dynamic_tuner.shadow)
        cs_a, cs_b = _CS(), _CS()
        for i, f in enumerate(_scenario(seed)):
          for cs in (cs_a, cs_b):
            cs.out.vEgo, cs.out.aEgo, cs.out.standstill = f["v"], f["a"], f["v"] < 0.01
            cs.out.gasPressed, cs.out.brakePressed = f["gas"], f["brake"]
          cc = _make_cc(f["accel"], f["state"], f["active"], f["pitch"])
          act_a, sends_a = with_sh.update(cc, structs.CarControlSP(), cs_a, i * int(1e7))
          act_b, sends_b = without.update(cc, structs.CarControlSP(), cs_b, i * int(1e7))
          self.assertEqual([(s[0], bytes(s[1]), s[2]) for s in sends_a], [(s[0], bytes(s[1]), s[2]) for s in sends_b], i)
          self.assertEqual((act_a.gas, act_a.brake, act_a.accel, act_a.torque), (act_b.gas, act_b.brake, act_b.accel, act_b.torque))
        ta, tb = with_sh.dynamic_tuner, without.dynamic_tuner
        va, vb = ta.debug_values(), tb.debug_values()
        self.assertEqual(va, vb)
        self.assertEqual((ta.brake_pid.i, ta.brake_gain_converged), (tb.brake_pid.i, tb.brake_gain_converged))
        self.assertGreater(sum(totals(with_sh.dynamic_tuner.shadow)), 0, "the scenario must exercise the shadow")
    self.assertTrue(any(x.startswith("hondashadow ") for x in lines))

  def test_a_raising_shadow_is_switched_off_and_the_car_carries_on(self):
    cc_obj = _build(True)
    tu = cc_obj.dynamic_tuner
    tu.shadow.update = mock.Mock(side_effect=ZeroDivisionError("boom"))
    cs = _CS()
    cs.out.vEgo = 10.0
    with mock.patch.object(dt.carlog, "exception") as logged:
      for i in range(10):
        cc_obj.update(_make_cc(-1.0, LongCtrlState.pid, True, 0.0), structs.CarControlSP(), cs, i * int(1e7))
    self.assertIsNone(tu.shadow)
    logged.assert_called_once()

  def test_garbage_in_never_raises(self):
    sh = HondaShadowLearners()
    nan, inf = float("nan"), float("inf")
    for v, a, ref, frac, gas, pitch in [(nan, 0, 0, 0, 0, 0), (10, inf, -1, .3, 0, 0), (10, -1, nan, nan, nan, nan),
                                        (inf, -1, -1, .3, 0, 0), (2, 1, 1, 0, inf, 0), (-5, -1, -1, -1, -1, 0)]:
      feed(sh, sl.WINDOW + 5, v=v, a=a, ref=ref, counts=frac * sl.NIDEC_BRAKE_MAX if math.isfinite(frac) else frac,
           gas=gas, pitch=pitch)
    for row in sh.brake:
      for c in row:
        self.assertTrue(c.n == 0 or math.isfinite(c.mean()))
    self.assertTrue(isinstance(sh.line(), str))


if __name__ == "__main__":
  unittest.main()
