"""FORK(HONDA_ACCORD_9G_AU): brake320 - this car's brake ceiling is 320 counts (owner decision 2026-10-24).

opendbc/sunnypilot/car/honda/brake_ceiling.py has the why. What this file pins, through the real CarInterface,
CarController and (for the panda's side) the real safety code:

  * the safety param: 4|32|128|256 = 420 on this car with openpilot longitudinal (164 before the 0x17C BRAKE_SWITCH
    mirror added bit 256 on 2026-10-29), 68 in stock ACC mode, and no 128 on any other Honda;
  * the car side never sends what panda would refuse, on ANY Honda platform and config: every 0x1FA the real
    CarController sends under the hardest demand (accel -10, a learned gain of 1.6, NaN/inf inputs, random traces) goes
    through the real honda.h tx hook with the same safety param and is allowed, and brake_ceiling.panda_max_brake()
    equals the panda's own limit, found by scanning it;
  * the law: the scale stays 2.6 m/s^2 per 256 counts, the fraction may reach 1.25 on this car only, full brake is 317
    counts at a gain of 1.0 (320 from a gain of 1.0081), and COMPUTER_BRAKE 320 is encoded and decoded as 320;
  * a request inside the old range (the plan plus the pitch feedforward no harder than -2.45 m/s^2) sends push12's
    frame, byte for byte, with the tuner on or off, at any speed and grade; beyond it, only ever more brake;
  * the learned gain (up to 1.6x) can take an ordinary request no further than 255;
  * (review 2026-10-24) only the pid state earns: a stop entered from a hard brake holds at most 255 at a standstill;
    the brake learner sees at most push12's 0.99, so after a beyond episode its gain and the counts are push12's;
    the radar's FCW/CHIME ride on openpilot's frame while openpilot's brake is the stronger;
  * the release limiter (32 counts a send) and both pump rules behave up to 320;
  * the VSA fallback: each signal latches 255 for the rest of the drive, accFaulted/espDisabled only after the VSA's
    start-up window; on the real frames of the 10-01 incident it latches, on a clean start and comma route 69 it does
    not; latched mid-brake it walks down to 255 through the release limiter; latched from the start every frame is the
    255 controller's;
  * the telemetry (customReserved14's numbers and the hondabrake line);
  * every other Honda: no brake320, and compute_gas_brake() unchanged.
"""
import logging
import math
import unittest
from collections import defaultdict
from unittest import mock

import numpy as np

from opendbc.car import gen_empty_fingerprint, structs
from opendbc.car.carlog import carlog
from opendbc.car.honda import hondacan
from opendbc.car.honda.carcontroller import CarController, compute_gas_brake, compute_gb_honda_elesys, \
  brake_pump_c1b_elesys, brake_pump_hysteresis_elesys
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR, DBC, HONDA_ELESYS, HondaSafetyFlags
from opendbc.car.honda.tests.test_elesys import _FakeCS, _as_tuple, _elesys_long_controller, structs_CC_SP
from opendbc.sunnypilot.car.honda import brake_ceiling as bc
from opendbc.sunnypilot.car.honda.brake_ceiling import Brake320Fallback, ElesysBrakeCeiling, brake320_status, \
  earned_ceiling, earned_fraction, panda_max_brake, request_fraction
from opendbc.sunnypilot.car.interfaces import setup_interfaces

ELESYS_CAR = CAR.HONDA_ACCORD_9G_AU
LCS = structs.CarControl.Actuators.LongControlState
F = HondaSafetyFlags


def _quiet():
  return mock.patch.object(carlog, "level", logging.CRITICAL)


def _cb(dat: bytes) -> int:
  return (dat[0] << 2) | (dat[1] >> 6)


class _CS(_FakeCS):
  """_FakeCS with every HUD dict answering 0 (the Bosch radarless controllers read keys the Nidec double lacks), and an
  optional brake320 fallback."""

  def __init__(self, fallback_reason=""):
    super().__init__()
    self.lkas_hud = defaultdict(int)
    self.acc_hud = defaultdict(int, self.acc_hud)
    self.stock_brake = defaultdict(int, self.stock_brake)
    self.brake320_fallback = Brake320Fallback()
    if fallback_reason:
      self.brake320_fallback.reason = fallback_reason


def _cc(accel, pitch=0.0, long_active=True, state=LCS.pid):
  cc = structs.CarControl.new_message()
  cc.enabled = cc.longActive = long_active
  cc.orientationNED = [0.0, pitch, 0.0]
  cc.actuators.accel = float(accel)
  cc.actuators.longControlState = state
  cc.hudControl.speedVisible = True
  cc.hudControl.setSpeed = 30.0
  return cc.as_reader()


def _run(cc_obj, v_kph, accels, pitch=0.0, cs=None, gain=None, latch_at=None, states=None, a_ego=None):
  """Drive cc_obj at a steady speed, one request per 100 Hz frame. gain pins the learned brake gain; latch_at latches
  the CarState's VSA fallback on that frame; a_ego(request) is the car's answer (default: exactly the request). Returns
  (frame, brake_last, COMPUTER_BRAKE, the 0x1FA payload) per send."""
  cs = cs or _CS()
  cs.out.vEgo = cs.out.vEgoRaw = v_kph / 3.6
  if gain is not None:
    cc_obj.dynamic_tuner.brake_gain = lambda *a, **k: gain
  out = []
  for i, a in enumerate(accels):
    if latch_at is not None and i == latch_at:
      cs.brake320_fallback.reason = "vsaFault"
    cs.out.aEgo = (float(a) if a_ego is None else float(a_ego(a))) if math.isfinite(a) else 0.0
    st = states[i] if states is not None else LCS.pid
    _, sends = cc_obj.update(_cc(a, pitch, state=st), structs_CC_SP(), cs, i * int(1e7))
    for addr, dat, _ in map(_as_tuple, sends):
      if addr == 0x1FA:
        out.append((i, cc_obj.brake_last, _cb(dat), dat))
  return out


def _params(car, alpha_long=False, pedal=True, stock_acc=False):
  fp = gen_empty_fingerprint()
  if pedal:
    fp[0][0x201] = 6
  fp[0][0x188] = 8
  CP = CarInterface.get_params(car, fp, [], alpha_long, False, False)
  CP_SP = CarInterface.get_params_sp(CP, car, fp, [], alpha_long, False, False)
  setup_interfaces(CarInterface, CP, CP_SP, [{"HondaElesysStockAcc": stock_acc}] if stock_acc else [])
  return CP, CP_SP


class TestBrake320Params(unittest.TestCase):
  def test_this_car_with_openpilot_long_is_420(self):
    for pedal in (True, False):
      CP, _ = _params(ELESYS_CAR, pedal=pedal)
      self.assertEqual(CP.safetyConfigs[-1].safetyParam,
                       F.NIDEC_ALT | F.ELESYS_SCM_STANDDOWN | F.ELESYS_BRAKE_320 | F.ELESYS_BRAKE_SWITCH_MIRROR)
      self.assertEqual(CP.safetyConfigs[-1].safetyParam, 420)
      self.assertEqual(panda_max_brake(CP), 320)
      self.assertIsNotNone(CarController(DBC[ELESYS_CAR], *_params(ELESYS_CAR, pedal=pedal)).brake320)

  def test_stock_acc_mode_is_68_as_before(self):
    CP, CP_SP = _params(ELESYS_CAR, stock_acc=True)
    self.assertEqual(CP.safetyConfigs[-1].safetyParam, 68)
    self.assertEqual(panda_max_brake(CP), 255)
    self.assertIsNone(CarController(DBC[ELESYS_CAR], CP, CP_SP).brake320)
    with _quiet():
      self.assertIsNone(CarInterface(CP, CP_SP).CS.brake320_fallback)

  def test_no_other_honda_carries_it(self):
    for car in CAR:
      if car in HONDA_ELESYS:
        continue
      for alpha in (False, True):
        for pedal in (False, True):
          CP, CP_SP = _params(car, alpha, pedal)
          for cfg in CP.safetyConfigs:
            self.assertFalse(cfg.safetyParam & F.ELESYS_BRAKE_320, (car, alpha, pedal))
          self.assertEqual(panda_max_brake(CP), 255, car)
          self.assertIsNone(CarController(DBC[car], CP, CP_SP).brake320, car)

  def test_the_mirror_rule(self):
    cfg = structs.CarParams.SafetyConfig.new_message()
    cp = structs.CarParams.new_message()
    for model, want in ((structs.CarParams.SafetyModel.hondaNidec, 320), (structs.CarParams.SafetyModel.hondaBosch, 255)):
      for param in range(256):
        cfg.safetyModel = model
        cfg.safetyParam = param
        cp.safetyConfigs = [cfg]
        on = bool(param & 128) and bool(param & 32) and not (param & 64)
        self.assertEqual(panda_max_brake(cp), want if on else 255, (model, param))
    # two configs: the lower
    a = structs.CarParams.SafetyConfig.new_message(safetyModel=structs.CarParams.SafetyModel.hondaNidec, safetyParam=164)
    b = structs.CarParams.SafetyConfig.new_message(safetyModel=structs.CarParams.SafetyModel.noOutput)
    cp.safetyConfigs = [b, a]
    self.assertEqual(panda_max_brake(cp), 255)
    cp.safetyConfigs = [a]
    self.assertEqual(panda_max_brake(cp), 320)
    for odd in (None, object(), structs.CarParams.new_message()):
      self.assertEqual(panda_max_brake(odd), 255)


class TestCarSideNeverExceedsPanda(unittest.TestCase):
  """The failure this guards against: a 0x1FA above panda's limit is refused, 0x1FA stops, and the VSA latches
  BRAKE_ERROR about a second later - mid-brake. Every 0x1FA the real controller sends, on every Honda platform and
  config, goes through the real panda safety with that config's params."""

  @classmethod
  def setUpClass(cls):
    from opendbc.safety.tests.libsafety import libsafety_py
    cls.libsafety_py = libsafety_py
    cls.safety = libsafety_py.libsafety

  def _hook(self, CP, CP_SP):
    cfg = CP.safetyConfigs[-1]
    self.safety.set_current_safety_param_sp(int(CP_SP.safetyParam))
    self.safety.set_safety_hooks(cfg.safetyModel.raw, cfg.safetyParam)
    self.safety.init_tests()
    self.safety.set_controls_allowed(True)

  def _panda_limit(self, CP, CP_SP, bus):
    """The largest COMPUTER_BRAKE the panda lets through on this bus, with controls allowed (-1: no 0x1FA at all)."""
    self._hook(CP, CP_SP)
    ok = [b for b in range(1024) if self.safety.safety_tx_hook(self.libsafety_py.make_CANPacket(
      0x1FA, bus, bytes([(b >> 2) & 0xFF, (b & 3) << 6, 0, 0, 0, 0, 0, 0])))]
    return max(ok) if ok else -1

  def _drive(self, cc_obj, rng):
    """The hardest demand: accel -10 at four speeds, a learned gain of 1.6, then random traces with NaN/inf."""
    frames = []
    for kph in (20, 60, 100, 130):
      frames += _run(cc_obj, kph, [-10.0] * 300, gain=1.6)
      frames += _run(cc_obj, kph, [0.5] * 50)
    frames += _run(cc_obj, 50, [float(x) for x in rng.uniform(-6.0, 1.0, 600)], pitch=-0.08, gain=1.6)
    if cc_obj.CP.carFingerprint in HONDA_ELESYS and cc_obj.CP_SP.enableGasInterceptor:
      frames += _run(cc_obj, math.nan, [-10.0] * 300, gain=1.6)    # a non-finite vEgo: the brake block's NaN guard
    return frames

  def test_every_platform_and_config(self):
    rng = np.random.default_rng(20261024)
    seen_320 = 0
    configs = 0
    with _quiet():
      for car in CAR:
        for alpha in (False, True):
          for pedal in (False, True):
            for stock_acc in ((False, True) if car in HONDA_ELESYS else (False,)):
              CP, CP_SP = _params(car, alpha, pedal, stock_acc)
              cc_obj = CarController(DBC[car], CP, CP_SP)
              frames = self._drive(cc_obj, rng)
              if not frames:
                continue
              configs += 1
              bus = cc_obj.CAN.pt
              limit = self._panda_limit(CP, CP_SP, bus)
              msg = f"{car} alpha={alpha} pedal={pedal} stock_acc={stock_acc}"
              if car not in HONDA_ELESYS and not (CP.flags & 4):   # every Nidec but this car: upstream's 255
                self.assertEqual(limit, 255, msg)
              if not (CP.flags & 4):                                 # Nidec: the mirror is exact
                self.assertEqual(panda_max_brake(CP), limit, msg)
              self.assertLessEqual(max(f[2] for f in frames), limit, msg)
              self._hook(CP, CP_SP)
              for i, _, cb, dat in frames:
                self.assertTrue(self.safety.safety_tx_hook(self.libsafety_py.make_CANPacket(0x1FA, bus, dat)),
                                f"{msg}: frame {i} COMPUTER_BRAKE {cb} refused")
              seen_320 += sum(f[2] > 255 for f in frames)
    self.assertGreater(configs, 40)
    self.assertGreater(seen_320, 100, "this car must actually brake above 255 in the drive")

  def test_a_fallback_mid_brake_steps_down_through_the_limiter(self):
    # review 2026-10-24: a VSA fallback latched at 320 used to cut the command to 255 in one send (-65 counts, about
    # -0.5 m/s^2 in 20 ms mid-brake). With the tuner, whose release limiter takes every other release down 32 counts a
    # send, the fallback walks down the same way (320 -> 288 -> 256 -> 255) and stays at 255; without the tuner, where
    # every release is immediate, it is 255 on the latch frame. Never above panda's 320, and never back above 255.
    for tuner in (False, True):
      cc_obj = _elesys_long_controller(tuner)
      frames = _run(cc_obj, 80, [-4.0] * 400, gain=1.6, latch_at=250)
      before = [f[2] for f in frames if f[0] < 250]
      after = [f[2] for f in frames if f[0] >= 250]
      self.assertEqual(max(before), 320)
      self.assertEqual(before[-1], 320)
      self.assertEqual(cc_obj.brake320.counts, 255)
      steps = [a - b for a, b in zip([before[-1]] + after, after, strict=False)]
      if tuner:
        self.assertEqual(after[:3], [288, 256, 255], "the limiter's walk")
        self.assertLessEqual(max(steps), 32)
      else:
        self.assertEqual(after[0], 255, "no limiter: at once, as every release without the tuner")
      self.assertTrue(all(c == 255 for c in after[3:]), f"tuner {tuner}: back above the old ceiling")


class TestBrake320Law(unittest.TestCase):
  def test_the_scale_is_unchanged(self):
    from opendbc.car.honda.values import CarControllerParams
    self.assertEqual(CarControllerParams.NIDEC_BRAKE_MAX, 256)
    self.assertEqual(bc.BRAKE_SCALE, 256)
    self.assertEqual(bc.BRAKE_FRAC_MAX_320, 1.25)
    # at 5 m/s and above (no creep offset) a request up to -2.6 is the same fraction with either clip
    for v in (5.0, 10.0, 30.0):
      for a in (-0.5, -1.3, -2.0, -2.5, -2.6):
        self.assertEqual(compute_gb_honda_elesys(a, v), compute_gb_honda_elesys(a, v, 1.25), (a, v))
    # beyond -2.6 net the fraction now asks for more, up to 1.25 at -3.25
    self.assertEqual(compute_gb_honda_elesys(-2.6, 10.0, 1.25), (0.0, 1.0))
    self.assertAlmostEqual(compute_gb_honda_elesys(-2.9, 10.0, 1.25)[1], 2.9 / 2.6)
    self.assertEqual(compute_gb_honda_elesys(-3.25, 10.0, 1.25)[1], 1.25)
    self.assertEqual(compute_gb_honda_elesys(-6.0, 10.0, 1.25)[1], 1.25)
    self.assertEqual(compute_gb_honda_elesys(-6.0, 10.0)[1], 1.0)

  def test_other_hondas_ignore_the_clip(self):
    from opendbc.car.honda.interface import CarInterface as CI
    for car in (CAR.HONDA_CRV, CAR.ACURA_ILX, CAR.HONDA_CIVIC):
      CP = CI.get_non_essential_params(car)
      for a in (-1.0, -4.0, -8.0):
        self.assertEqual(compute_gas_brake(a, 10.0, CP), compute_gas_brake(a, 10.0, CP, 1.25), car)
        self.assertLessEqual(compute_gas_brake(a, 10.0, CP, 1.25)[1], 1.0)

  def test_full_brake_is_317_at_any_speed_and_320_with_a_learned_gain(self):
    for kph in (15, 30, 60, 100, 126):
      frames = _run(_elesys_long_controller(tuner=False), kph, [-4.0] * 300)
      self.assertEqual(frames[-1][2], 317, kph)
      self.assertAlmostEqual(frames[-1][1], 1.24, places=9)
    for gain, want in ((1.0, 317), (1.008, 319), (1.0081, 320), (1.0158, 320), (1.6, 320), (0.85, 269)):
      self.assertEqual(_run(_elesys_long_controller(tuner=True), 60, [-4.0] * 300, gain=gain)[-1][2], want, gain)

  def test_320_is_encoded_and_decoded_as_320(self):
    from opendbc.can import CANPacker, CANParser
    from opendbc.car import Bus
    packer = CANPacker(DBC[ELESYS_CAR][Bus.pt])
    CP = CarInterface.get_non_essential_params(ELESYS_CAR)
    CP_SP = CarInterface.get_non_essential_params_sp(CP, ELESYS_CAR)
    can = hondacan.CanBus(CP)
    sb = {"CHIME": 0}
    ref = None
    for counts in (1, 253, 255, 256, 300, 317, 319, 320):
      addr, dat, bus = hondacan.create_brake_command(packer, can, counts, True, True, False, False, sb, CP_SP,
                                                     is_metric=True, elesys=True)
      self.assertEqual((addr, bus), (0x1FA, 0))
      self.assertEqual(_cb(dat), counts)
      cp = CANParser(DBC[ELESYS_CAR][Bus.pt], [("BRAKE_COMMAND", math.nan)], 0)
      cp.update([(int(1e9), [(addr, dat, 0)])])
      self.assertEqual(cp.vl["BRAKE_COMMAND"]["COMPUTER_BRAKE"], counts)
      self.assertEqual(cp.vl["BRAKE_COMMAND"]["BRAKE_PUMP_REQUEST"], 1)
      self.assertEqual(cp.vl["BRAKE_COMMAND"]["CRUISE_OVERRIDE"], 1)
      # nothing but COMPUTER_BRAKE moves with the count (byte 7 is the counter and the checksum)
      body = bytes([dat[1] & 0x3F]) + dat[2:7]
      ref = ref or body
      self.assertEqual(body, ref, counts)

  def test_inside_the_old_range_every_frame_is_push12s(self):
    # the request (accel plus the pitch feedforward) never harder than -2.45: brake320 sends exactly what the
    # controller without it sends, tuner on (learner, release limiter, soft stop, pitch) or off, at any speed and grade
    rng = np.random.default_rng(2445)
    for tuner in (False, True):
      for kph in (2, 8, 30, 60, 110):
        for pitch in (0.0, -0.05, 0.04):
          # the pitch feedforward (tuner on) adds g*sin(pitch) to the request: keep accel + it no harder than -2.44
          lower = -2.44 - (9.81 * math.sin(pitch) if (tuner and pitch < 0) else 0.0)
          accels = [float(a) for a in np.clip(np.cumsum(rng.normal(0, 0.08, 1500)) * 0.3 - 1.0, lower, 0.8)]
          new = _run(_elesys_long_controller(tuner=tuner), kph, accels, pitch=pitch)
          old = _run(_elesys_long_controller(tuner=tuner, brake320=False), kph, accels, pitch=pitch)
          self.assertEqual([f[3] for f in new], [f[3] for f in old], f"tuner {tuner} {kph} km/h pitch {pitch}")
          self.assertTrue(all(f[2] <= 255 for f in new))

  def test_beyond_it_only_more_brake(self):
    # light, firm, beyond, release, beyond again: before the first frame beyond -2.45 the frames are push12's; beyond,
    # every send brakes at least as hard; after it the old frames come back within the release tail
    phases = [(-0.8, 200), (-2.0, 200), (-3.4, 250), (-1.5, 150), (0.4, 120), (-2.9, 200), (-1.2, 200), (0.3, 100)]
    for tuner in (False, True):
      for kph in (30, 70, 110):
        accels = [a for a, n in phases for _ in range(n)]
        new = _run(_elesys_long_controller(tuner=tuner), kph, accels)
        old = _run(_elesys_long_controller(tuner=tuner, brake320=False), kph, accels)
        msg = f"tuner {tuner} {kph} km/h"
        first = next(k for k, f in enumerate(new) if accels[f[0]] < -2.45)
        self.assertEqual([f[3] for f in new[:first]], [f[3] for f in old[:first]], msg)
        beyond = [k for k, f in enumerate(new) if accels[f[0]] < -2.45]
        # every send beyond brakes at least as hard. (Before the review of 2026-10-24 the tuner's second episode could
        # start from a learned gain a hair apart, the learner fed 1.0 where push12 fed 0.99: 2 counts of slack.)
        self.assertTrue(all(new[k][2] >= old[k][2] for k in beyond), msg)
        self.assertEqual(max(new[k][2] for k in beyond), 317 if not tuner else max(new[k][2] for k in beyond))
        self.assertGreaterEqual(max(new[k][2] for k in beyond), 310, msg)
        self.assertLessEqual(max(old[k][2] for k in beyond), 255, msg)
        # the tail: within 10 sends of the request coming back inside the old range the counts are push12's again
        # (tuner on, the release limiter walks down from the higher command first), and stay so
        for end in (k for k in range(1, len(new)) if accels[new[k][0]] >= -2.45 > accels[new[k - 1][0]]):
          back = next(j for j in range(end, len(new)) if new[j][2] == old[j][2])
          self.assertLessEqual(back - end, 10, f"{msg}: send {end}")
          nxt = next((j for j in range(end, len(new)) if accels[new[j][0]] < -2.45), len(new))
          self.assertEqual([f[2] for f in new[back:nxt]], [f[2] for f in old[back:nxt]], f"{msg}: send {end}")

  def test_a_runaway_gain_stays_at_255_on_an_ordinary_request(self):
    # the learned gain (up to 1.6x) used to be bounded by the 255 clip; it still is, until the request itself passes
    # the old ceiling's -2.45, and then only as far as the request earns
    for kph in (30, 80, 120):
      for a in (-1.0, -1.8, -2.2, -2.44):
        new = _run(_elesys_long_controller(tuner=True), kph, [a] * 400, gain=1.6)
        old = _run(_elesys_long_controller(tuner=True, brake320=False), kph, [a] * 400, gain=1.6)
        self.assertEqual([f[3] for f in new], [f[3] for f in old], (kph, a))
        self.assertLessEqual(max(f[2] for f in new), 255)
      cb = {a: _run(_elesys_long_controller(tuner=True), kph, [a] * 400, gain=1.6)[-1][2] for a in (-2.5, -2.6, -2.9, -3.25)}
      for a, c in cb.items():
        self.assertEqual(c, earned_ceiling(request_fraction(a), 320), (kph, a))
      self.assertEqual(cb[-2.6], 267)
      self.assertEqual(cb[-3.25], 320)

  def test_the_earned_clips(self):
    self.assertAlmostEqual(bc.EARN_START_FRAC, 2.45 / 2.6)
    for f, frac, counts in ((0.0, 1.0, 255), (0.5, 1.0, 255), (2.45 / 2.6, 1.0, 255), (1.0, None, 267), (1.24, None, 320),
                            (1.25, 1.25, 320), (2.0, 1.25, 320)):
      if frac is not None:
        self.assertAlmostEqual(earned_fraction(f, 320), frac, places=12, msg=f)
      self.assertEqual(earned_ceiling(f, 320), counts, f)
      self.assertEqual(earned_fraction(f, 255), 1.0)
      self.assertEqual(earned_ceiling(f, 255), 255)
    for odd in (math.nan, math.inf, None, "x"):
      self.assertEqual(earned_fraction(odd, 320), 1.0)
      self.assertEqual(earned_ceiling(odd, 320), 255)
      self.assertEqual(request_fraction(odd), 0.0)
    self.assertEqual(request_fraction(1.0), 0.0)
    self.assertAlmostEqual(request_fraction(-2.6), 1.0)
    # monotone in the request
    fs = np.linspace(0, 1.5, 400)
    self.assertTrue(all(np.diff([earned_fraction(f, 320) for f in fs]) >= 0))
    self.assertTrue(all(np.diff([earned_ceiling(f, 320) for f in fs]) >= 0))

  def test_creep_alone_never_earns_it(self):
    # route test_dynamic_tuning_integration's [2]: a -1.5 request in the PID state at a crawl, where the creep offset
    # (+1.15 at a standstill) and a slight downhill take the net past -2.6. push12 clipped it at 1.0 (253); so does this
    for kph in (0.0, 0.5, 2.0, 5.0):
      for a in (-1.0, -1.5, -2.0):
        new = _run(_elesys_long_controller(tuner=True), kph, [a] * 300, pitch=-0.03)
        old = _run(_elesys_long_controller(tuner=True, brake320=False), kph, [a] * 300, pitch=-0.03)
        self.assertEqual([f[3] for f in new], [f[3] for f in old], (kph, a))

  @staticmethod
  def _stop(cc_obj, held):
    """A hard stop as LongControl drives it: the PID state braking at `held` from 8 m/s down to the stopping speed
    (0.8 m/s, stopping_tune.py), then the stopping state, which keeps LongControl's last output when that is already
    below stopAccel (longcontrol.py), to a standstill and 20 s of it. Returns (state, vEgo, COMPUTER_BRAKE) per send."""
    cs = _CS()
    out = []
    v, st = 8.0, LCS.pid
    for i in range(2500):
      if v <= 0.8:
        st = LCS.stopping
      cs.out.vEgo = cs.out.vEgoRaw = v
      cs.out.aEgo = -2.4 if v > 0.0 else 0.0
      cs.out.standstill = v == 0.0
      _, sends = cc_obj.update(_cc(held, state=st), structs_CC_SP(), cs, i * int(1e7))
      for addr, dat, _ in map(_as_tuple, sends):
        if addr == 0x1FA:
          out.append((st, v, _cb(dat)))
      v = max(v - 0.024, 0.0)
    return out

  def test_a_hard_stop_holds_no_more_than_255(self):
    # review 2026-10-24: the stopping state holds LongControl's last output, and the pitch term is 0 there, so a stop
    # entered from a hard brake went on "earning" beyond 255 at a standstill: 267-317 counts held for the whole stop (the
    # logged stops: 8 of 373, 13-34 s each) with nothing to gain, on the VSA that logged the 32-11 fault. Only the pid
    # state earns now. The approach still brakes past 255; in the stopping state the clips are the old 1.0 and 255 and
    # the hysteresis state the old ceiling's (<= 0.99; without that it kept 1.01 of the higher command and held 255), so
    # the standstill hold is push12's, frame for frame.
    for tuner in (False, True):
      for held in (-2.6, -3.0, -3.25):
        new = self._stop(_elesys_long_controller(tuner=tuner), held)
        old = self._stop(_elesys_long_controller(tuner=tuner, brake320=False), held)
        msg = f"tuner {tuner} held {held}"
        self.assertGreater(max(cb for st, _, cb in new if st == LCS.pid), 255, f"{msg}: the approach earns")
        still_new = [cb for _, v, cb in new if v == 0.0]
        still_old = [cb for _, v, cb in old if v == 0.0]
        self.assertGreater(len(still_new), 1000, msg)
        self.assertLessEqual(max(still_new), 255, msg)
        self.assertEqual(still_new[-1000:], still_old[-1000:], f"{msg}: the hold is push12's")
        # in the stopping state nothing is above 255 once the release limiter has walked down from the approach
        stopping = [cb for st, _, cb in new if st == LCS.stopping]
        self.assertLessEqual(max(stopping[3:]), 255, msg)

  def test_after_a_beyond_episode_the_learned_gain_is_push12s(self):
    # review 2026-10-24: the brake learner weights its error by the brake fraction. Fed 1.0 beyond the old ceiling where
    # push12's never passed 0.99 (actuator_hysteresis()'s ceiling), its gain came out a hair apart, and later ordinary
    # braking differed by a count either way for minutes (replay: 283 sends a count LESS, up to 14 min later). Capped at
    # 0.99 it learns what push12's learned. Here the car answers 0.05 m/s^2 short of the request, or 0.05 past it, so
    # the learner moves on every steady stretch - before, during and after a request beyond -2.45 - without reaching
    # its rails: once the release tail is over every count is push12's again, and both learned gains are the same
    # floats. What the longer release tail itself leaves (it pumped 2 sends longer above 200 counts) is C1b's 6 s
    # top-up burst 2 sends later in the same application; from the next application every frame is push12's.
    from opendbc.can import CANPacker
    from opendbc.car import Bus
    packer = CANPacker(DBC[ELESYS_CAR][Bus.pt])
    off_ = packer.make_can_msg("BRAKE_COMMAND", 0, {"BRAKE_PUMP_REQUEST": 0})[1]
    on_ = packer.make_can_msg("BRAKE_COMMAND", 0, {"BRAKE_PUMP_REQUEST": 1})[1]
    pump = bytes(a ^ b for a, b in zip(off_[:7], on_[:7], strict=True))
    phases = [(-1.6, 500), (-3.4, 500), (-1.6, 700), (0.3, 100), (-2.0, 500)]
    accels = [a for a, n in phases for _ in range(n)]
    for off in (0.05, -0.05):
      for kph in (40, 90):
        new_cc = _elesys_long_controller(tuner=True)
        old_cc = _elesys_long_controller(tuner=True, brake320=False)
        new = _run(new_cc, kph, accels, a_ego=lambda a, off=off: a + off)
        old = _run(old_cc, kph, accels, a_ego=lambda a, off=off: a + off)
        msg = f"{kph} km/h, the car {off:+} off the request"
        self.assertGreaterEqual(max(f[2] for f in new), 300, msg)
        self.assertLessEqual(max(f[2] for f in old), 255, msg)
        t_new, t_old = new_cc.dynamic_tuner, old_cc.dynamic_tuner
        self.assertNotEqual(t_old.brake_gain_converged, 0.0, f"{msg}: the learner moved")
        self.assertEqual((t_new._shadow_brake_gain, t_new.brake_gain_converged), (t_old._shadow_brake_gain, t_old.brake_gain_converged),
                         msg)
        end = next(k for k, f in enumerate(new) if f[0] >= 1000)   # the request back inside the old range
        nxt = next(k for k, f in enumerate(new) if f[0] >= 1800)   # the next application
        self.assertEqual([f[2] for f in new[end + 25:]], [f[2] for f in old[end + 25:]], msg)
        for a, b in zip(new[end + 25:nxt], old[end + 25:nxt], strict=True):
          self.assertTrue(all((x ^ y) & ~m == 0 for x, y, m in zip(a[3][:7], b[3][:7], pump, strict=True)), f"{msg}: frame {a[0]}")
        self.assertEqual([f[3] for f in new[nxt:]], [f[3] for f in old[nxt:]], msg)

  def test_the_radars_warning_rides_on_openpilots_frame(self):
    # review 2026-10-24: while openpilot brakes harder than a CMBS brake of 256-319, panda keeps sending openpilot's 0x1FA
    # (the stronger brake wins, honda.h), so the radar's own frame with its FCW and CHIME reached the car only once
    # openpilot braked less - and today's stockAeb rule does not fire on these events (the 27 logged CMBS brakes at 256+
    # carry FCW=2 and none of its four bits), so nothing else warned. With openpilot capped at 255 a CMBS brake of 256+
    # always won and its warning reached the car. Now, above 255, openpilot's own 0x1FA carries the radar's FCW (2) and
    # CHIME while the radar warns; up to 255, and without a radar warning, it is push12's frame; the 255 controller and
    # every other car are untouched. The radar's frame is parsed by the real CANParser, as CarState hands it over.
    from opendbc.can import CANPacker, CANParser
    from opendbc.car import Bus
    dbc = DBC[ELESYS_CAR][Bus.pt]
    packer = CANPacker(dbc)

    def radar(fcw, chime, cb=280):
      cp = CANParser(dbc, [("BRAKE_COMMAND", math.nan)], 2)
      _, dat, _ = _as_tuple(packer.make_can_msg("BRAKE_COMMAND", 2, {"COMPUTER_BRAKE": cb, "FCW": fcw, "CHIME": chime,
                                                                      "BRAKE_PUMP_REQUEST": 1, "COMPUTER_BRAKE_REQUEST": 1}))
      cp.update([(int(1e9), [(0x1FA, dat, 2)])])
      return cp.vl["BRAKE_COMMAND"]

    def drive(cc_obj, stock_brake, visual=None, accel=-4.0):
      cs = _CS()
      cs.stock_brake = stock_brake
      cs.out.vEgo = cs.out.vEgoRaw = 60 / 3.6
      out = []
      for i in range(300):
        cs.out.aEgo = -2.4
        cc = structs.CarControl.new_message()
        cc.enabled = cc.longActive = True
        cc.actuators.accel = accel
        cc.actuators.longControlState = LCS.pid
        cc.hudControl.speedVisible = True
        cc.hudControl.setSpeed = 30.0
        if visual is not None:
          cc.hudControl.visualAlert = visual
        _, sends = cc_obj.update(cc.as_reader(), structs_CC_SP(), cs, i * int(1e7))
        for addr, dat, _ in map(_as_tuple, sends):
          if addr == 0x1FA:
            p = CANParser(dbc, [("BRAKE_COMMAND", math.nan)], 0)
            p.update([(int(1e9), [(addr, dat, 0)])])
            v = p.vl["BRAKE_COMMAND"]
            out.append((int(v["COMPUTER_BRAKE"]), int(v["FCW"]), int(v["CHIME"]), dat))
      return out

    warn = radar(2, 2)
    self.assertEqual((warn["FCW"], warn["CHIME"]), (2, 2))
    for tuner in (False, True):
      new = drive(_elesys_long_controller(tuner=tuner), warn)
      self.assertGreater(max(f[0] for f in new), 300)
      self.assertTrue(all((f[1], f[2]) == ((2, 2) if f[0] > 255 else (0, 0)) for f in new), f"tuner {tuner}")
      # the 255 controller never carries it; and a request inside the old range sends push12's frames, warning or not
      self.assertTrue(all(f[1:3] == (0, 0) for f in drive(_elesys_long_controller(tuner=tuner, brake320=False), warn)))
      inside = drive(_elesys_long_controller(tuner=tuner), warn, accel=-2.3)
      self.assertEqual([f[3] for f in inside], [f[3] for f in drive(_elesys_long_controller(tuner=tuner, brake320=False), warn, accel=-2.3)])
      self.assertTrue(all(f[1:3] == (0, 0) for f in inside) and max(f[0] for f in inside) > 200)
      # no radar warning (FCW 0, or the bare test double without the key): openpilot's own flag only
      for quiet in (radar(0, 0), radar(1, 0), {"CHIME": 0}):
        self.assertTrue(all(f[1:3] == (0, 0) for f in drive(_elesys_long_controller(tuner=tuner), quiet)), quiet)
      # openpilot's own FCW alert: FCW and the radar's CHIME, as upstream, at any count
      own = drive(_elesys_long_controller(tuner=tuner), radar(0, 3), visual=structs.CarControl.HUDControl.VisualAlert.fcw)
      self.assertTrue(all(f[1:3] == (2, 3) for f in own))
    self.assertFalse(ElesysBrakeCeiling.fcw_on_the_wire(False, 300, object()), "unreadable: openpilot's own flag")
    self.assertTrue(ElesysBrakeCeiling.fcw_on_the_wire(True, 10, None))

  def test_release_limiter_from_317(self):
    accels = [-4.0] * 300 + [0.3] * 200
    cbs = [f[2] for f in _run(_elesys_long_controller(tuner=True), 90, accels)]
    self.assertGreaterEqual(max(cbs), 316)
    drops = [a - b for a, b in zip(cbs, cbs[1:], strict=False)]
    self.assertLessEqual(max(drops), 32)
    self.assertGreaterEqual(drops.count(32), 9, "317 -> 0 is ten 32-count sends")
    self.assertEqual(cbs[-1], 0)
    # without the tuner (no limiter) it is let go at once, as before
    cbs = [f[2] for f in _run(_elesys_long_controller(tuner=False), 90, accels)]
    self.assertEqual(cbs[149], 317)
    self.assertEqual(cbs[150], 0)

  def test_pump_up_to_320(self):
    # both pump rules: moving above 200 counts the pump runs continuously, 320 included; at a standstill the rules are
    # unchanged above 255 (a firm hold is never topped up; a rise of more than 15 counts is delivered)
    for cb in (201, 255, 256, 300, 317, 320):
      pump, _, _, _ = brake_pump_c1b_elesys(cb, 20.0, 0, 0, 0.0, 100.0)
      self.assertTrue(pump, cb)
      pump, _, _ = brake_pump_hysteresis_elesys(cb, 20.0, 0, 0.0, 100.0)
      self.assertTrue(pump, cb)
    lvl, trig, last = 0, 0, -10.0
    on = []
    for k in range(400):
      p, lvl, trig, last = brake_pump_c1b_elesys(300, 0.0, lvl, trig, last, k * 0.02)
      on.append(p)
    self.assertTrue(on[0] and not any(on[50:]), "one build burst, then quiet")
    # and on the wire: every send above 200 counts while moving carries the pump bit
    from opendbc.can import CANPacker
    from opendbc.car import Bus
    packer = CANPacker(DBC[ELESYS_CAR][Bus.pt])
    off = packer.make_can_msg("BRAKE_COMMAND", 0, {"BRAKE_PUMP_REQUEST": 0})[1]
    on = packer.make_can_msg("BRAKE_COMMAND", 0, {"BRAKE_PUMP_REQUEST": 1})[1]
    mask = bytes(a ^ b for a, b in zip(off[:7], on[:7], strict=True))
    for tuner in (False, True):
      frames = _run(_elesys_long_controller(tuner=tuner), 60, [-4.0] * 300)
      hi = [f for f in frames if f[2] > 200]
      self.assertTrue(hi and max(f[2] for f in hi) >= 316)
      self.assertTrue(all(any(x & m for x, m in zip(f[3][:7], mask, strict=True)) for f in hi), f"tuner {tuner}")


class TestBrake320Fallback(unittest.TestCase):
  def test_each_signal_latches_for_the_drive(self):
    names = ("acc_faulted", "esp_disabled", "vsa_fault", "vsa_stored_fault", "monitor_error")
    want = {"acc_faulted": "accFaulted", "esp_disabled": "espDisabled", "vsa_fault": "vsaFault",
            "vsa_stored_fault": "vsaStoredFault", "monitor_error": "vsaMonitorError"}
    for name in names:
      fb = Brake320Fallback()
      kw = dict.fromkeys(names, False)
      with _quiet():
        for _ in range(10):
          self.assertEqual(fb.update(settled=True, **kw), "")
        kw[name] = True
        self.assertEqual(fb.update(settled=True, **kw), want[name])
        kw[name] = False
        for _ in range(1000):
          self.assertEqual(fb.update(settled=True, **kw), want[name], "latched: it never clears")
      self.assertEqual(fb.latched_frame, 11)

  def test_acc_faulted_and_esp_disabled_wait_for_the_start_up_window(self):
    fb = Brake320Fallback()
    for _ in range(600):
      self.assertEqual(fb.update(True, True, False, False, settled=False), "")
    with _quiet():
      self.assertEqual(fb.update(True, True, False, False, settled=True), "accFaulted+espDisabled")
    # the VSA's own two count at once
    fb = Brake320Fallback()
    with _quiet():
      self.assertEqual(fb.update(False, False, True, False, settled=False), "vsaFault")

  def test_one_error_line(self):
    fb = Brake320Fallback()
    with mock.patch.object(bc.carlog, "error") as err:
      for _ in range(300):
        fb.update(True, False, True, False, settled=True)
    err.assert_called_once()
    self.assertIn("hondabrake fallback=vsaFault+accFaulted", err.call_args[0][0])
    self.assertIn("320 -> 255", err.call_args[0][0])

  def test_never_raises(self):
    class Bad:
      def __bool__(self):
        raise RuntimeError("boom")
    fb = Brake320Fallback()
    with _quiet():
      self.assertEqual(fb.update(Bad(), False, False, False, True), "error", "unknown is the safe side: 255")
    fb = Brake320Fallback()
    with mock.patch.object(bc.carlog, "error", side_effect=RuntimeError("log")):
      self.assertEqual(fb.update(False, False, True, False, True), "vsaFault")

  def _replay(self, name, batches=None):
    from opendbc.sunnypilot.car.honda import test_vsa_fault as V
    batches = batches or V.SCENARIOS[name]["batches"]
    ci = V.make_ci(ELESYS_CAR)
    self.assertIsNotNone(ci.CS.brake320_fallback)
    out = []
    t = batches[0][0]
    i = 0
    with V.quiet_carlog():
      while t <= batches[-1][0] + V.DT_MS:
        frames = []
        while i < len(batches) and batches[i][0] < t + V.DT_MS:
          frames += [(addr, bytes.fromhex(h), 0) for addr, h in batches[i][1]]
          i += 1
        ret, _ = ci.update([(int(t * 1e6), frames)])
        out.append((t / 1000.0, ci.CS.brake320_fallback.reason, ret.espDisabled, ret.accFaulted))
        t += V.DT_MS
    return out

  def test_the_incident_latches_it_and_a_clean_start_does_not(self):
    # real frames (test_vsa_fault.py's fixture): the 10-01 live onsets and stored starts latch; 10f's clean start, whose
    # bulb check lights ESP_DISABLED, and comma route 69's other lamp state do not
    for name, first in (("110_episode_A", "vsaFault"), ("112_episode_B", "vsaFault"), ("111_stored_start", "vsaStoredFault"),
                        ("113_stored_start_and_clear", "vsaStoredFault")):
      rows = self._replay(name)
      reasons = {r[1] for r in rows}
      latched = next(r for r in rows if r[1])
      self.assertTrue(latched[1].startswith(first), (name, latched))
      self.assertEqual(rows[-1][1], latched[1], f"{name}: still latched at the end")
      self.assertLessEqual(len(reasons), 2, name)
    for name in ("10f_clean_start", "comma69_lamp_b3_5_b4_1"):
      rows = self._replay(name)
      self.assertFalse(any(r[1] for r in rows), name)
    self.assertTrue(any(r[2] for r in self._replay("10f_clean_start")), "the fixture's bulb check lights ESP_DISABLED")

  def test_a_bulb_check_stretched_past_the_longest_seen_does_not(self):
    from opendbc.sunnypilot.car.honda import test_vsa_fault as V
    batches = V.SCENARIOS["10f_clean_start"]["batches"]
    t0 = next(t for t, fr in batches for a, _ in fr if a == 0x1A4)
    stretched = []
    for t, fr in batches:
      out = []
      for a, h in fr:
        if a == 0x1A4 and t - t0 <= 4080.0:
          d = bytearray.fromhex(h)
          d[3] = 0xF6
          d[-1] &= 0xF0
          d[-1] |= V.honda_checksum(a, bytes(d))
          h = d.hex()
        out.append([a, h])
      stretched.append([t, out])
    rows = self._replay("10f_clean_start", batches=stretched)
    self.assertFalse(any(r[1] for r in rows))

  def test_esp_disabled_after_the_window_latches(self):
    # the VSA OFF lamp (driver's switch, or the VSA's own fault) after start-up: 10f's frames with ESP_DISABLED set
    # from 5.5 s after the VSA's first frame
    from opendbc.sunnypilot.car.honda import test_vsa_fault as V
    from opendbc.can import CANPacker
    from opendbc.car import Bus
    batches = V.SCENARIOS["10f_clean_start"]["batches"]
    t0 = next(t for t, fr in batches for a, _ in fr if a == 0x1A4) + 5500.0
    self.assertGreater(batches[-1][0], t0 + 500.0)
    packer = CANPacker(DBC[ELESYS_CAR][Bus.pt])
    probe = packer.make_can_msg("VSA_STATUS", 0, {"ESP_DISABLED": 1})[1]
    mask = bytearray(a ^ b for a, b in zip(probe, packer.make_can_msg("VSA_STATUS", 0, {})[1], strict=True))
    mask[-1] = 0
    self.assertEqual(sum(bin(m).count("1") for m in mask), 1, "one bit")
    edited = []
    for t, fr in batches:
      out = []
      for a, h in fr:
        if a == 0x1A4 and t >= t0:
          d = bytearray.fromhex(h)
          for k in range(len(d) - 1):
            d[k] |= mask[k]
          d[-1] = (d[-1] & 0xF0)
          d[-1] |= V.honda_checksum(a, bytes(d))
          h = d.hex()
        out.append([a, h])
      edited.append([t, out])
    rows = self._replay("10f_clean_start", batches=edited)
    latched = [r for r in rows if r[1]]
    self.assertTrue(latched and latched[0][1] == "espDisabled", latched[:1])
    self.assertAlmostEqual(latched[0][0], t0 / 1000.0, delta=0.05)

  def test_once_latched_every_frame_is_the_255_controllers(self):
    rng = np.random.default_rng(320255)
    phases = [(-0.8, 150), (-2.2, 150), (-4.0, 300), (-1.0, 150), (0.4, 100), (-3.0, 300), (0.3, 100)]
    for tuner in (False, True):
      for kph in (20, 60, 110):
        accels = [a + float(rng.normal(0, 0.05)) for a, n in phases for _ in range(n)]
        new = _run(_elesys_long_controller(tuner=tuner), kph, accels, cs=_CS("vsaStoredFault"))
        old = _run(_elesys_long_controller(tuner=tuner, brake320=False), kph, accels)
        self.assertEqual([f[3] for f in new], [f[3] for f in old], f"tuner {tuner} {kph} km/h")


class TestBrake320Telemetry(unittest.TestCase):
  def test_the_window_and_the_drive(self):
    cc_obj = _elesys_long_controller(tuner=False)
    with mock.patch.object(bc.carlog, "info") as info:
      frames = _run(cc_obj, 60, [-4.0] * 400 + [-1.0] * 200)
    b = cc_obj.brake320
    self.assertEqual(b.drive_max, 317)
    self.assertEqual(b.drive_above, sum(f[2] > 255 for f in frames))
    self.assertEqual(b.last_max, max(f[2] for f in frames if f[0] >= 500))
    self.assertEqual(b.last_above, 0)
    lines = [c[0][0] for c in info.call_args_list if c[0][0].startswith("hondabrake")]
    self.assertEqual(len(lines), sum(1 for w in range(6) if max(f[2] for f in frames if w * 100 <= f[0] < (w + 1) * 100) > 255))
    self.assertTrue(lines[-1].startswith("hondabrake max=317 above255=50 ceil=320 allowed=320 fallback=-"), lines[-1])

  def test_status(self):
    with _quiet():
      ci = CarInterface(*_params(ELESYS_CAR))
    st = brake320_status(ci)
    self.assertEqual(st, {"allowedCounts": 320, "ceilingCounts": 320, "vsaFallback": False, "vsaFallbackReason": "",
                          "vsaFallbackFrame": 0, "maxCountsLastSecond": 0, "sendsAbove255LastSecond": 0,
                          "maxCountsDrive": 0, "sendsAbove255Drive": 0})
    ci.CS.brake320_fallback.reason, ci.CS.brake320_fallback.latched_frame = "espDisabled", 812
    st = brake320_status(ci)
    self.assertEqual((st["ceilingCounts"], st["vsaFallback"], st["vsaFallbackReason"], st["vsaFallbackFrame"]),
                     (255, True, "espDisabled", 812))
    # every other car, and stock ACC mode: nothing to report
    for CPs in (_params(CAR.HONDA_CRV), _params(ELESYS_CAR, stock_acc=True)):
      with _quiet():
        self.assertIsNone(brake320_status(CarInterface(*CPs)))
    self.assertIsNone(brake320_status(object()))

  def test_never_raises(self):
    b = ElesysBrakeCeiling(_params(ELESYS_CAR)[0])
    b.sent("x")
    b.sent(None)
    b.update(object())
    self.assertEqual(b.counts, 320)

    class BadCS:
      @property
      def brake320_fallback(self):
        raise RuntimeError("boom")
    b.update(BadCS())
    self.assertEqual((b.counts, b.fallback_reason), (255, "error"))


class TestOtherHondasUnchanged(unittest.TestCase):
  def test_byte_identical_whatever_the_request(self):
    # every other Nidec runs the old brake lines (no brake320 object); a hard request is clipped at 255 as ever
    for car in (CAR.HONDA_CRV, CAR.ACURA_ILX, CAR.HONDA_ODYSSEY, CAR.HONDA_PILOT):
      CP, CP_SP = _params(car)
      cc_obj = CarController(DBC[car], CP, CP_SP)
      self.assertIsNone(cc_obj.brake320)
      frames = _run(cc_obj, 60, [-8.0] * 300)
      self.assertLessEqual(max(f[2] for f in frames), 255)
      self.assertEqual(frames[-1][2], int((0.99 - float(np.interp(60 / 3.6, [0.0, 2.3, 35.0], [0.001, 0.002, 0.15]))) * 256))


if __name__ == "__main__":
  unittest.main()
