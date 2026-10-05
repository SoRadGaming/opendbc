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
class Hud:
  leadVisible: bool = False


@dataclass
class FakeCC:
  longActive: bool = True
  actuators: Act = field(default_factory=Act)
  hudControl: Hud = field(default_factory=Hud)


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
  pcm_pedal_gas: float = 0.0     # CarStateExt's 0x17C PEDAL_GAS


def feed(sh, n, v=10.0, a=-1.0, ref=-1.0, counts=80.0, gas=0.0, pitch=0.0, pose_fresh=True,
         mode_ok=True, long_active=True, state=LongCtrlState.pid, pcm=None, lead=False, gain=1.0, **out):
  """n 50 Hz frames of one state. pcm: the PCM's pedal reading; by default it sees the interceptor's command."""
  cc = FakeCC(long_active, Act(ref, state), Hud(lead))
  cs = FakeCS(Out(vEgo=v, aEgo=a, **out), (40.0 if gas > 0 else 0.0) if pcm is None else pcm)
  for _ in range(n):
    sh.update(cc, cs, pitch=pitch, pose_fresh=pose_fresh, mode_ok=mode_ok, cmd_ref=ref,
              brake_frac=counts / sl.NIDEC_BRAKE_MAX, gas_cmd=gas, brake_gain=gain)


def totals(sh):
  return (sum(c.n for row in sh.brake for c in row), sum(c.n for c in sh.coast_acc), sum(r.n for r in sh.launch))


def parse(line):
  out = {}
  for tok in line.split()[1:]:
    k, v = tok.split("=", 1)
    if k in sl.BUILD_KEYS:
      out[k] = v           # tags stay text: a commit can be all digits
    else:
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

  def test_applicable_only_on_the_elesys_accord_with_the_interceptor(self):
    def sp(interceptor):
      CP = CarInterface.get_non_essential_params(PLATFORM)
      CP_SP = CarInterface.get_non_essential_params_sp(CP, PLATFORM)
      CP_SP.enableGasInterceptor = interceptor
      return CP_SP
    self.assertTrue(sl.shadow_applicable(CarInterface.get_non_essential_params(PLATFORM), sp(True)))
    self.assertFalse(sl.shadow_applicable(CarInterface.get_non_essential_params(PLATFORM), sp(False)),
                     "no interceptor: the pedal command reads 0 on every frame")
    self.assertFalse(sl.shadow_applicable(CarInterface.get_non_essential_params(CAR.HONDA_CIVIC), sp(True)))
    self.assertFalse(sl.shadow_applicable(None, None))


class TestGates(unittest.TestCase):
  def test_steady_brake_lands_in_its_cell_with_gravity_removed(self):
    sh = HondaShadowLearners()
    # 2% downhill (nose-down, pitch < 0): the car shows -1.0 but the brakes delivered -1.0 + 0.196
    pitch = -0.02
    feed(sh, sl.CLEAN_HOLD + 49, v=12.0, a=-1.0, ref=-1.0, counts=80.0, pitch=pitch)
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

  def test_nothing_is_admitted_until_a_clean_second_after_an_override(self):
    # route 10f t 2425.8: the driver lifts off, the law's commands read zero over the WINDOW (the interceptor sent 0
    # during the override), and aEgo still carries the driver's throttle: +1.17 'coast'. A brake and a launch sample
    # on the frames after a release, or after an engagement, are the same hole.
    cases = {"coast": dict(counts=0.0, v=11.5, a=1.2, ref=-0.1), "brake": dict(counts=80.0, v=12.0, a=0.5, ref=-1.0),
             "launch": dict(counts=0.0, v=2.0, a=2.0, ref=1.0, gas=0.15)}
    overrides = {"gas pressed": dict(gasPressed=True), "disengaged": dict(long_active=False),
                 "gas pressed and disengaged": dict(gasPressed=True, long_active=False),
                 "brake pressed": dict(brakePressed=True), "stock AEB": dict(stockAeb=True),
                 "stopping": dict(state=LongCtrlState.stopping), "pose stale": dict(pose_fresh=False)}
    for name, kw in cases.items():
      for oname, okw in overrides.items():
        if name == "launch" and oname == "stopping":
          continue   # not an override for a launch any more: its window runs through the stop (next test)
        with self.subTest(f"{name} after {oname}"):
          sh = HondaShadowLearners()
          feed(sh, 200, **(kw | okw))
          self.assertEqual(totals(sh), (0, 0, 0))
          feed(sh, sl.CLEAN_HOLD - 1, **kw)
          self.assertEqual(totals(sh), (0, 0, 0), "still inside the clean hold")
          feed(sh, 1, **kw)
          self.assertEqual(sum(totals(sh)), 1, "a clean second later")

  def test_a_launch_from_an_openpilot_held_stop_is_sampled_from_first_wheel_motion(self):
    # batch 3 (learnaudit G6): 115 t 511, 10f t 303 / 2408 -- control left the stopping state with the car already at
    # 0.61-1.07 m/s, and a second of PID after that put the first launch sample above 1 m/s. The launch's clean run now
    # counts through the held stop, so the 0.5-1 m/s slice is sampled while control is still stopping.
    stopping = LongCtrlState.stopping
    sh = HondaShadowLearners()
    feed(sh, 150, v=0.0, a=0.0, ref=-0.5, counts=180.0, state=stopping)      # held at the stop, engaged, no pedals
    self.assertEqual(totals(sh), (0, 0, 0))
    feed(sh, sl.WINDOW, v=0.3, a=1.2, ref=1.0, counts=0.0, gas=0.15, state=stopping)   # first motion: creeping off
    feed(sh, 30, v=0.7, a=1.3, ref=1.0, counts=0.0, gas=0.15, state=stopping)
    self.assertEqual(sh.launch[0].n, 30, "sampled at 0.7 m/s, still in the stopping state")
    self.assertEqual(sh.launch_episodes, 1)
    self.assertEqual(totals(sh)[:2], (0, 0), "the brake and coast tables still need a second of PID")
    # a driver's pedal still costs the launch a clean second, in any state
    sh = HondaShadowLearners()
    feed(sh, 150, v=0.0, a=0.0, ref=-0.5, counts=180.0, state=stopping, gasPressed=True)
    feed(sh, sl.CLEAN_HOLD - 1, v=0.7, a=1.3, ref=1.0, counts=0.0, gas=0.15, state=stopping)
    self.assertEqual(totals(sh), (0, 0, 0))
    feed(sh, 1, v=0.7, a=1.3, ref=1.0, counts=0.0, gas=0.15, state=stopping)
    self.assertEqual(totals(sh), (0, 0, 1))

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
    feed(sh, sl.CLEAN_HOLD, counts=50.0)     # admitted, <= 60 counts
    before = totals(sh)
    feed(sh, sl.WINDOW - 1, counts=65.0)     # the window still holds a <=60 command, so 65 crosses bands: not admitted
    self.assertEqual(totals(sh), before)
    feed(sh, 1, counts=65.0)                 # the whole window at 65 now
    self.assertEqual(sh.brake[speed_band(10.0)][count_band(65.0)].n, 1)

  def test_coast(self):
    sh = HondaShadowLearners()
    feed(sh, sl.CLEAN_HOLD + 99, v=20.0, a=-0.35, ref=-0.2, counts=0.0)
    b = speed_band(20.0)
    self.assertEqual(sh.coast_acc[b].n, 100)
    self.assertAlmostEqual(sh.coast_acc[b].mean(), -0.35)
    self.assertAlmostEqual(sh.coast_err[b].mean(), -0.15)
    sh = HondaShadowLearners()
    feed(sh, 200, v=2.5, a=-0.3, ref=-0.2, counts=0.0)   # below COAST_MIN_SPEED
    self.assertEqual(totals(sh), (0, 0, 0))

  def test_launch(self):
    sh = HondaShadowLearners()
    feed(sh, sl.CLEAN_HOLD + 149, v=2.0, a=1.4, ref=1.0, counts=0.0, gas=0.15)
    self.assertEqual(sh.launch[0].n, 150)
    self.assertAlmostEqual(sh.launch[0].ratio(), 1.4)
    self.assertAlmostEqual(launch_multiplier(*sh.launch_pooled()), 1 / 1.4)
    self.assertEqual(sh.launch_episodes, 1)
    for name, kw in {"weak demand": dict(ref=0.2), "brake in the window": dict(counts=5.0),
                     "pedal not pressing": dict(gas=0.0), "above 6 m/s": dict(v=6.5), "creep": dict(v=0.3),
                     "steeper than 4 deg": dict(pitch=0.075), "the PCM does not see the pedal": dict(pcm=0.0),
                     "the PCM's pedal not recorded": dict(pcm=float("nan"))}.items():
      with self.subTest(name):
        sh = HondaShadowLearners()
        args = dict(v=2.0, a=1.4, ref=1.0, counts=0.0, gas=0.15) | kw
        feed(sh, 200, **args)
        self.assertEqual(totals(sh)[2], 0, name)

  def test_a_launch_behind_a_lead_is_kept_apart(self):
    # A_synth L2b learns from no-lead launches: behind a lead the samples are logged on their own, never in lmult
    sh = HondaShadowLearners()
    feed(sh, sl.CLEAN_HOLD + 149, v=2.0, a=1.4, ref=1.0, counts=0.0, gas=0.15, lead=True)
    self.assertEqual((sh.launch[0].n, sh.launch_lead[0].n), (0, 150))
    self.assertEqual(launch_multiplier(*sh.launch_pooled()), 1.0)
    self.assertEqual((sh.launch_episodes, sh.launch_episodes_lead), (0, 1))
    d = parse(sh.line())
    self.assertEqual(d["lnl"], [150.0, 0.0])
    self.assertAlmostEqual(d["lratiol"][0], 1.4)
    self.assertEqual(d["ln"], [0.0, 0.0])
    self.assertEqual((d["lep"], d["lepl"]), (0.0, 1.0))

  def test_launch_on_a_grade_is_measured_with_gravity_removed(self):
    sh = HondaShadowLearners()     # route 115's launch: -2.7 deg, the car shows 1.88 for a 1.04 command
    pitch = math.radians(-2.7)
    feed(sh, sl.CLEAN_HOLD + 149, v=2.0, a=1.88, ref=1.04, counts=0.0, gas=0.15, pitch=pitch)
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
    feed(sh, sl.CLEAN_HOLD + sl.MIN_CELL_SAMPLES, v=4.0, a=-0.7, ref=-1.0, counts=50.0)
    self.assertAlmostEqual(sh.brake_correction(4.0, 50.0), -0.3, places=6)
    self.assertAlmostEqual(sh.brake_correction(1.5, 50.0), -0.15, places=6)
    self.assertEqual(sh.brake_correction(0.5, 50.0), 0.0)
    self.assertEqual(sh.brake_correction(4.0, 2.0), 0.0)


class TestLog(unittest.TestCase):
  def test_line_cadence_and_shape(self):
    lines = []
    with mock.patch.object(sl.carlog, "info", lambda msg, *a, **k: lines.append(msg)):
      sh = HondaShadowLearners()
      feed(sh, 2 * sl.LOG_INTERVAL, v=12.0, a=-0.8, ref=-1.0, counts=80.0, gain=1.07)
      self.assertEqual(len(lines), 2)                        # once a minute while learning...
      feed(sh, 2 * sl.LOG_INTERVAL, long_active=False)
      self.assertEqual(len(lines), 3)                        # ...once at the disengage, then silent
      feed(sh, 5, v=12.0, a=-0.8, ref=-1.0, counts=80.0, gasPressed=True)   # engaged, nothing admitted
      feed(sh, 5, long_active=False)
      self.assertEqual(len(lines), 3)                        # nothing admitted in the window: no line
    d = parse(lines[-1])
    self.assertTrue(lines[-1].startswith("hondashadow v=2 commit=- gaslaw=- cap=- pump=- blaw=- tuner=- "), lines[-1])
    self.assertLess(max(len(x) for x in lines), 1200)
    n_cells = len(sl.SPEED_BP) * (len(sl.BRAKE_COUNT_BP) + 1)
    for k in ("bn", "be", "bsd", "bcorr", "bcb", "bacc"):
      self.assertEqual(len(d[k]), n_cells, k)
    i = speed_band(12.0) * (len(sl.BRAKE_COUNT_BP) + 1) + count_band(80.0)
    self.assertEqual(d["bn"][i], 2 * sl.LOG_INTERVAL - sl.CLEAN_HOLD + 1)
    self.assertAlmostEqual(d["be"][i], 0.2, places=3)
    self.assertAlmostEqual(d["bcorr"][i], -0.2, places=3)
    self.assertTrue(math.isnan(d["be"][0]))
    self.assertEqual(d["lmult"], 1.0)
    self.assertEqual(d["bgain"], [1.07], "the live brake gain the table was measured at")
    for k in ("lnl", "lral", "lrrl", "lratiol"):
      self.assertEqual(len(d[k]), len(sl.LAUNCH_SPEED_BP) - 1, k)


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
    self.pcm_pedal_gas = 40.0


def _build(shadow: bool, tuning: bool = True, flags: int = 0, store: dict | None = None):
  params = _Params({"HondaDynamicTuningEnabled": tuning, eg.GAS_LAW_PARAM: True, **(store or {})})
  with mock.patch.object(dt, "_open_params", lambda: params), mock.patch.object(eg, "_open_params", lambda: params):
    CP = CarInterface.get_non_essential_params(PLATFORM)
    CP_SP = CarInterface.get_non_essential_params_sp(CP, PLATFORM)
    CP.openpilotLongitudinalControl = True
    CP_SP.enableGasInterceptor = True
    CP_SP.flags |= flags
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

  def test_a_shadow_that_cannot_be_built_leaves_the_controller_working(self):
    # an import or construction failure in __init__ (the trap dynamic_tuning.py's import comment describes)
    with mock.patch.object(sl, "HondaShadowLearners", side_effect=RuntimeError("boom")), \
         mock.patch.object(dt.carlog, "exception") as logged:
      cc_obj = _build(True)
    self.assertIsNone(cc_obj.dynamic_tuner.shadow)
    logged.assert_called_once()
    self.assertTrue(cc_obj.dynamic_tuner.enabled)
    cs = _CS()
    cs.out.vEgo = 10.0
    for i in range(10):
      cc_obj.update(_make_cc(-1.0, LongCtrlState.pid, True, 0.0), structs.CarControlSP(), cs, i * int(1e7))

  def test_with_the_tuner_off_the_shadow_runs_and_nothing_actuated_changes(self):
    # batch 3: the shadow learners and the per-mode counter run with Dynamic Tuning's live parts off (logging mode).
    # Against a controller whose tuner has no logging mode at all, the CAN and the actuator outputs are identical.
    lines = []
    with mock.patch.object(sl.carlog, "info", lambda msg, *a, **k: lines.append(msg)), \
         mock.patch.object(dt.carlog, "info", lambda msg, *a, **k: lines.append(msg)):
      for seed in (4, 5):
        logging_on = _build(True, tuning=False)
        with mock.patch.object(sl, "shadow_applicable", lambda *a, **k: False):
          logging_off = _build(True, tuning=False)
        ta, tb = logging_on.dynamic_tuner, logging_off.dynamic_tuner
        self.assertEqual((ta.enabled, ta.logging, ta.shadow is not None), (False, True, True))
        self.assertEqual((tb.enabled, tb.logging, tb.shadow is None), (False, False, True))
        self.assertIsNone(logging_on.soft_stop, "the soft stop stays with the toggle")
        cs_a, cs_b = _CS(), _CS()
        for i, f in enumerate(_scenario(seed)):
          for cs in (cs_a, cs_b):
            cs.out.vEgo, cs.out.aEgo, cs.out.standstill = f["v"], f["a"], f["v"] < 0.01
            cs.out.gasPressed, cs.out.brakePressed = f["gas"], f["brake"]
          cc = _make_cc(f["accel"], f["state"], f["active"], f["pitch"])
          act_a, sends_a = logging_on.update(cc, structs.CarControlSP(), cs_a, i * int(1e7))
          act_b, sends_b = logging_off.update(cc, structs.CarControlSP(), cs_b, i * int(1e7))
          self.assertEqual([(s[0], bytes(s[1]), s[2]) for s in sends_a], [(s[0], bytes(s[1]), s[2]) for s in sends_b], i)
          self.assertEqual((act_a.gas, act_a.brake, act_a.accel, act_a.torque), (act_b.gas, act_b.brake, act_b.accel, act_b.torque))
          self.assertIsNone(ta.filtered_pitch())
        self.assertEqual(ta.brake_gain(cc, cs_a, 0.3), 1.0)
        self.assertGreater(sum(totals(ta.shadow)), 0, "the scenario must exercise the shadow")
        self.assertGreater(ta.mode_moving["D"], 10.0)
        self.assertEqual(ta.mode_seconds["D"], 0.0, "engaged seconds (persisted, the UI's) stay the tuner's")
        self.assertIsNone(ta._writer, "nothing is persisted with the toggle off")
    shadow_lines = [x for x in lines if x.startswith("hondashadow ")]
    dyn_lines = [x for x in lines if x.startswith("hondadyn ")]
    self.assertTrue(shadow_lines and all(" tuner=0 " in x and " gaslaw=v2 cap=1 " in x for x in shadow_lines), shadow_lines[:1])
    self.assertTrue(dyn_lines and all(" tuner=0 " in x for x in dyn_lines))

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


class TestBuildTagsAndPersistence(unittest.TestCase):
  """batch 3: every hondashadow line says what it was measured on, and a drive's totals reach disk at a disengage and
  at card's exit, not only once a minute."""

  def test_tags_follow_the_flags_the_commit_and_the_gas_law(self):
    @dataclass
    class SP:
      flags: int = 0
    self.assertEqual((sl.pump_rule_tag(SP(0)), sl.brake_law_tag(SP(0))), ("v5", "v1"))
    self.assertEqual((sl.pump_rule_tag(SP(16)), sl.brake_law_tag(SP(16))), ("v6", "v1"))
    self.assertEqual((sl.pump_rule_tag(SP(32 | 8)), sl.brake_law_tag(SP(32 | 8))), ("v5", "v2"), "8 is stock ACC")
    self.assertEqual((sl.pump_rule_tag(None), sl.brake_law_tag(None)), ("-", "-"))
    self.assertEqual(sl.git_commit_tag(_Params({"GitCommit": "d995bc95a1b2c3"})), "d995bc95a")
    self.assertEqual(sl.git_commit_tag(_Params({"GitCommit": b"862540c00aa"})), "862540c00")
    self.assertEqual([sl.git_commit_tag(p) for p in (_Params({}), None, _Params({"GitCommit": "a b=c"}))], ["-", "-", "-"])
    self.assertEqual([sl.launch_cap_tag(x) for x in ("v2", "v1", "nidec", "-", "")], ["1", "0", "0", "-", "-"])
    from opendbc.sunnypilot.car.honda.values_ext import HondaFlagsSP
    for name, value in (("ELESYS_PUMP_V6", sl.PUMP_V6_FLAG), ("ELESYS_BRAKE_LAW_V2", sl.BRAKE_LAW_V2_FLAG)):
      if hasattr(HondaFlagsSP, name):
        self.assertEqual(int(getattr(HondaFlagsSP, name)), value, f"the fixed contract: {name}")
    self.assertEqual((sl.PUMP_V6_FLAG, sl.BRAKE_LAW_V2_FLAG), (16, 32))

  def test_the_controller_tags_its_lines(self):
    lines = []
    with mock.patch.object(sl.carlog, "info", lambda msg, *a, **k: lines.append(msg)):
      for flags, pump, blaw in ((0, "v5", "v1"), (sl.PUMP_V6_FLAG | sl.BRAKE_LAW_V2_FLAG, "v6", "v2")):
        cc_obj = _build(True, flags=flags, store={"GitCommit": "862540c0123456"})
        tu = cc_obj.dynamic_tuner
        self.assertEqual(tu.build, {"commit": "862540c01", "gaslaw": "-", "cap": "-", "pump": pump, "blaw": blaw, "tuner": "1"})
        cs = _CS()
        cs.out.vEgo = 10.0
        for i in range(10):
          cc_obj.update(_make_cc(-1.0, LongCtrlState.pid, True, 0.0), structs.CarControlSP(), cs, i * int(1e7))
        tu.shadow._dirty = True
        tu.shadow.emit()
        d = parse(lines[-1])
        self.assertEqual({k: d[k] for k in sl.BUILD_KEYS},
                         {"commit": "862540c01", "gaslaw": "v2", "cap": "1", "pump": pump, "blaw": blaw, "tuner": "1"})

  def test_mode_counter_counts_manual_moving_time(self):
    tu = _build(True).dynamic_tuner
    cc = _make_cc(0.0, LongCtrlState.off, False, 0.0)
    cs = _CS()
    cs.out.vEgo = 12.0
    for _ in range(1000):          # 10 s driven by hand
      tu.update_state(cc, cs)
    cs.out.vEgo = 0.5
    for _ in range(500):           # 5 s crawling below MODE_MOVING_SPEED
      tu.update_state(cc, cs)
    self.assertAlmostEqual(tu.mode_moving["D"], 10.0, places=6)
    self.assertEqual(tu.mode_seconds["D"], 0.0)
    assert "modemov=[10.0,0.0,0.0]" in self._dyn_line(tu)

  @staticmethod
  def _dyn_line(tu):
    out = []
    with mock.patch.object(dt.carlog, "info", lambda msg, *a, **k: out.append(msg)):
      tu.log_state(0)
    return out[-1]

  def test_a_disengage_persists_the_drive(self):
    tu = _build(True).dynamic_tuner
    tu._writer.put_many = mock.Mock()
    cs = _CS()
    cs.out.vEgo = 10.0
    on, off = _make_cc(-0.5, LongCtrlState.pid, True, 0.0), _make_cc(0.0, LongCtrlState.off, False, 0.0)
    for frame in range(1, 300):
      tu.update_state(on, cs)
      tu.persist(frame)
    tu._writer.put_many.assert_not_called()
    tu.update_state(off, cs)
    tu.persist(300)
    tu._writer.put_many.assert_called_once()
    self.assertAlmostEqual(tu._writer.put_many.call_args[0][0]["HondaDynModeSecD"], 2.99, places=6)
    for frame in range(301, 400):
      tu.update_state(off, cs)
      tu.persist(frame)
    self.assertEqual(tu._writer.put_many.call_count, 1, "once per disengage, not every frame after it")

  def test_exit_flush_writes_synchronously_and_ends_on_the_totals(self):
    store = {}
    cc_obj = _build(True, store=store)
    tu = cc_obj.dynamic_tuner
    params = tu._params
    tu.mode_seconds["D"] = 12.5
    tu.shadow._dirty = True
    lines = []
    with mock.patch.object(sl.carlog, "info", lambda msg, *a, **k: lines.append(msg)), \
         mock.patch.object(dt.carlog, "info", lambda msg, *a, **k: lines.append(msg)):
      tu.flush_at_exit()
      tu.flush_at_exit()
    self.assertEqual(params.store["HondaDynModeSecD"], 12.5)
    assert "HondaDynBrakeGain" in params.store
    self.assertEqual([x.split()[0] for x in lines], ["hondashadow", "hondadyn"], "once, the shadow's total then the tuner's")

  def test_write_now_is_never_overtaken_by_an_older_batch(self):
    import threading
    import time
    started, release = threading.Event(), threading.Event()

    class Slow(_Params):
      def put(self, key, val, block=False):
        if val == 1.0:
          started.set()
          release.wait(2.0)
        super().put(key, val)
    p = Slow({})
    w = dt._ParamWriter(p)
    w.put_many({"k": 1.0})
    started.wait(2.0)
    threading.Timer(0.2, release.set).start()
    self.assertTrue(w.write_now({"k": 3.0}))    # waits for the batch in flight, then writes last
    w.put_many({"k": 2.0})                       # anything after the final write is dropped
    time.sleep(0.2)
    self.assertEqual(p.store["k"], 3.0)

  def test_exit_flush_is_hooked_only_on_the_device(self):
    for device, calls in ((True, 1), (False, 0)):
      with self.subTest(device=device), mock.patch.object(dt, "_device_params", lambda p, d=device: d), \
           mock.patch("atexit.register") as reg, mock.patch("multiprocessing.util.Finalize") as fin:
        tu = _build(True).dynamic_tuner
        self.assertEqual((reg.call_count, fin.call_count), (calls, calls))
        if calls:
          self.assertEqual(fin.call_args.kwargs.get("exitpriority"), 10)
          with mock.patch.object(tu, "flush_at_exit") as flush:
            reg.call_args[0][0]()          # what the process exit runs
          flush.assert_called_once()


if __name__ == "__main__":
  unittest.main()
