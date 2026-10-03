"""FORK(HONDA_ACCORD_9G_AU): stock ACC mode (HondaElesysStockAcc).

The car's own ACC (the Elesys radar on panda bus 2) does gas and brake, openpilot steers only, and every frame passes
between bus 0 and bus 2. One hook, _initialize_honda (opendbc/sunnypilot/car/interfaces.py), turns it on from the param;
with the param off nothing may change by a single byte - CarParams, CarParamsSP and every CAN send.
"""
import copy
import unittest

from opendbc.can import CANPacker
from opendbc.car import Bus, gen_empty_fingerprint, structs
from opendbc.car.can_definitions import CanData
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR, DBC, HONDA_ELESYS, HondaSafetyFlags
from opendbc.sunnypilot.car.honda.values_ext import HondaFlagsSP, HondaSafetyFlagsSP
from opendbc.sunnypilot.car.interfaces import setup_interfaces

ELESYS_CAR = CAR.HONDA_ACCORD_9G_AU
PARAM = "HondaElesysStockAcc"
STOCK_SAFETY_PARAM = HondaSafetyFlags.NIDEC_ALT | HondaSafetyFlags.ELESYS_STOCK_ACC   # 68, what pandaStates must echo


def _fingerprint(pedal: bool = True):
  fp = gen_empty_fingerprint()
  fp[0][0x188] = 8          # GEARBOX_AUTO, as on the car
  if pedal:
    fp[0][0x201] = 6        # GAS_SENSOR: the comma pedal
  return fp


def _params(car=ELESYS_CAR, pedal: bool = True, params_list=None, hook: bool = True):
  """car_helpers.get_car's order: get_params, get_params_sp, then the sunnypilot hooks."""
  fp = _fingerprint(pedal)
  CP = CarInterface.get_params(car, fp, [], False, False, False)
  CP_SP = CarInterface.get_params_sp(CP, car, fp, [], False, False, False)
  if hook:
    setup_interfaces(CarInterface, CP, CP_SP, params_list)
  return CP, CP_SP


def _sp_tuple(CP_SP):
  """CarParamsSP is a dataclass here (card converts it to capnp); compare it whole."""
  return repr(CP_SP)


class TestStockAccHookOff(unittest.TestCase):
  """The toggle OFF is today's car, byte for byte."""

  def test_off_is_byte_identical_to_no_hook(self):
    for pedal in (True, False):
      base_cp, base_sp = _params(pedal=pedal, hook=False)
      base_bytes = base_cp.to_bytes()
      for params_list in (None, [], [{PARAM: False}], [{PARAM: "0"}], [{PARAM: 0}]):
        CP, CP_SP = _params(pedal=pedal, params_list=params_list)
        self.assertEqual(CP.to_bytes(), base_bytes, msg=f"pedal={pedal} {params_list}")
        self.assertEqual(_sp_tuple(CP_SP), _sp_tuple(base_sp), msg=f"pedal={pedal} {params_list}")

  def test_off_values_are_todays(self):
    # the car's running configuration (route 113): hondaNidec 36, SP 2, openpilot long, interceptor, pcmCruise False
    CP, CP_SP = _params(params_list=[{PARAM: False}])
    self.assertEqual(CP.safetyConfigs[-1].safetyModel, structs.CarParams.SafetyModel.hondaNidec)
    self.assertEqual(CP.safetyConfigs[-1].safetyParam, 36)
    self.assertEqual(CP_SP.safetyParam, HondaSafetyFlagsSP.GAS_INTERCEPTOR)
    self.assertTrue(CP.openpilotLongitudinalControl)
    self.assertFalse(CP.pcmCruise)
    self.assertTrue(CP_SP.enableGasInterceptor)
    self.assertTrue(CP.autoResumeSng)
    self.assertEqual(CP_SP.flags & HondaFlagsSP.ELESYS_STOCK_ACC, 0)


class TestStockAccHookOn(unittest.TestCase):
  def test_on_with_the_pedal(self):
    for value in (True, "1", 1):
      CP, CP_SP = _params(params_list=[{PARAM: value}])
      self.assertEqual(CP.safetyConfigs[-1].safetyModel, structs.CarParams.SafetyModel.hondaNidec)
      self.assertEqual(CP.safetyConfigs[-1].safetyParam, STOCK_SAFETY_PARAM)
      self.assertEqual(STOCK_SAFETY_PARAM, 68)
      self.assertFalse(CP.safetyConfigs[-1].safetyParam & HondaSafetyFlags.ELESYS_SCM_STANDDOWN)
      self.assertEqual(CP_SP.safetyParam, 0)
      self.assertFalse(CP.openpilotLongitudinalControl)
      self.assertTrue(CP.pcmCruise)
      self.assertFalse(CP_SP.enableGasInterceptor)
      self.assertFalse(CP.autoResumeSng)
      self.assertTrue(CP_SP.flags & HondaFlagsSP.ELESYS_STOCK_ACC)
      self.assertEqual(HondaFlagsSP.ELESYS_STOCK_ACC, 8)
      self.assertTrue(CP_SP.pcmCruiseSpeed)
      self.assertFalse(CP.alphaLongitudinalAvailable)

  def test_on_keeps_min_enable_speed(self):
    off, _ = _params(params_list=[{PARAM: False}])
    on, _ = _params(params_list=[{PARAM: True}])
    self.assertAlmostEqual(on.minEnableSpeed, off.minEnableSpeed)
    self.assertAlmostEqual(on.minEnableSpeed, 19 * 0.44704, places=4)

  def test_on_without_the_pedal_is_the_same_mode(self):
    with_pedal, sp_with = _params(pedal=True, params_list=[{PARAM: True}])
    without, sp_without = _params(pedal=False, params_list=[{PARAM: True}])
    self.assertEqual(with_pedal.to_bytes(), without.to_bytes())
    self.assertEqual(_sp_tuple(sp_with), _sp_tuple(sp_without))

  def test_only_the_named_fields_change(self):
    # the hook's whole footprint: everything else in CarParams is exactly the toggle-off value
    off, off_sp = _params(params_list=[{PARAM: False}])
    on, on_sp = _params(params_list=[{PARAM: True}])
    off_d, on_d = off.to_dict(), on.to_dict()
    changed = {k for k in set(off_d) | set(on_d) if off_d.get(k) != on_d.get(k)}
    self.assertEqual(changed, {"openpilotLongitudinalControl", "pcmCruise", "autoResumeSng", "safetyConfigs"})
    expect_sp = copy.deepcopy(off_sp)
    expect_sp.flags |= HondaFlagsSP.ELESYS_STOCK_ACC.value
    expect_sp.enableGasInterceptor = False
    expect_sp.safetyParam = 0
    self.assertEqual(_sp_tuple(on_sp), _sp_tuple(expect_sp))

  def test_never_the_stand_down_and_the_stock_bit_together(self):
    for pedal in (True, False):
      for value in (False, True):
        CP, _ = _params(pedal=pedal, params_list=[{PARAM: value}])
        p = CP.safetyConfigs[-1].safetyParam
        self.assertNotEqual(p & (HondaSafetyFlags.ELESYS_SCM_STANDDOWN | HondaSafetyFlags.ELESYS_STOCK_ACC),
                            HondaSafetyFlags.ELESYS_SCM_STANDDOWN | HondaSafetyFlags.ELESYS_STOCK_ACC)

  def test_other_hondas_ignore_the_param(self):
    for car in CAR:
      if car in HONDA_ELESYS:
        continue
      fp = gen_empty_fingerprint()
      base = CarInterface.get_params(car, fp, [], False, False, False)
      base_sp = CarInterface.get_params_sp(base, car, fp, [], False, False, False)
      setup_interfaces(CarInterface, base, base_sp, None)
      CP = CarInterface.get_params(car, fp, [], False, False, False)
      CP_SP = CarInterface.get_params_sp(CP, car, fp, [], False, False, False)
      setup_interfaces(CarInterface, CP, CP_SP, [{PARAM: "1"}])
      self.assertEqual(CP.to_bytes(), base.to_bytes(), msg=str(car))
      self.assertEqual(_sp_tuple(CP_SP), _sp_tuple(base_sp), msg=str(car))


def _cc(cancel=False, resume=False):
  cc = structs.CarControl.new_message()
  cc.enabled = True
  cc.latActive = True
  cc.longActive = False
  cc.actuators.torque = 0.3
  cc.actuators.accel = -1.0
  cc.cruiseControl.cancel = cancel
  cc.cruiseControl.resume = resume
  cc.hudControl.speedVisible = True
  cc.hudControl.setSpeed = 25.0
  return cc.as_reader()


class _FakeCS:
  """What CarController.update() reads from CS."""

  def __init__(self):
    self.out = structs.CarState.new_message()
    self.out.vEgo = 25.0
    self.out.cruiseState.enabled = True
    self.out.cruiseState.available = True
    self.out.cruiseState.speed = 25.0
    self.out.cruiseState.standstill = False
    self.v_cruise_factor = 1.0
    self.stock_brake = {"CHIME": 0, "AEB_REQ_1": 0, "AEB_REQ_2": 0, "AEB_STATUS": 0}
    self.acc_hud = {"FCM_OFF": 0, "FCM_OFF_2": 0, "FCM_PROBLEM": 0, "ICONS": 0}
    self.lkas_hud = {}
    self.scm_buttons = {"CRUISE_BUTTONS": 0, "CRUISE_SETTING": 0, "MAIN_ON": 1}
    self.is_metric = True
    self.econ_on = False
    self.out_sp = structs.CarStateSP()


def _sends(CP, CP_SP, frames, cc_for):
  CI = CarInterface(CP, CP_SP)
  cs = _FakeCS()
  out = []
  for i in range(frames):
    _, sends = CI.CC.update(cc_for(i), structs.CarControlSP(), cs, i * int(1e7))  # ty: ignore[invalid-argument-type]
    out.append([(m[0], bytes(m[1]), m[2]) if isinstance(m, tuple) else (m.address, bytes(m.dat), m.src) for m in sends])
  return out


class TestStockAccCarController(unittest.TestCase):
  FRAMES = 1200

  def test_only_steering_and_the_hud_side_channel(self):
    # cancel and resume both asked for (controlsd asks for cancel whenever openpilot is not engaged): neither is sent
    for pedal in (True, False):
      CP, CP_SP = _params(pedal=pedal, params_list=[{PARAM: True}])
      for cc_for in (lambda i: _cc(cancel=True), lambda i: _cc(resume=True), lambda i: _cc(cancel=i % 2 == 0, resume=i % 2 == 1)):
        frames = _sends(CP, CP_SP, self.FRAMES, cc_for)
        counts: dict[tuple[int, int], int] = {}
        for sends in frames:
          for addr, dat, bus in sends:
            self.assertGreater(len(dat), 0, msg=f"empty frame {hex(addr)} on bus {bus}")
            counts[(addr, bus)] = counts.get((addr, bus), 0) + 1
        self.assertEqual(set(counts), {(0xE4, 0), (0x500, 0)})
        self.assertEqual(counts[(0xE4, 0)], self.FRAMES)            # 100 Hz
        self.assertEqual(counts[(0x500, 0)], self.FRAMES // 10)     # 10 Hz
        for i, sends in enumerate(frames):
          self.assertEqual([a for a, _, _ in sends], [0xE4, 0x500] if i % 10 == 0 else [0xE4], msg=f"frame {i}")

  def test_toggle_off_still_sends_the_long_frames_and_the_stand_down(self):
    # the other side of the same gate: the interceptor build keeps every frame it has today
    CP, CP_SP = _params(pedal=True, params_list=[{PARAM: False}])
    frames = _sends(CP, CP_SP, 200, lambda i: _cc(cancel=True))
    seen = {(a, b) for sends in frames for a, _, b in sends}
    self.assertEqual(seen, {(0xE4, 0), (0x1FA, 0), (0x200, 0), (0x1A6, 2), (0x30C, 0), (0x500, 0)})

  def test_toggle_off_sends_are_unchanged_by_the_param_plumbing(self):
    # the same controller with and without the hook having run with the param off: every byte equal
    for pedal in (True, False):
      cp_a, sp_a = _params(pedal=pedal, hook=False)
      cp_b, sp_b = _params(pedal=pedal, params_list=[{PARAM: False}])
      a = _sends(cp_a, sp_a, 300, lambda i: _cc(cancel=i % 3 == 0))
      b = _sends(cp_b, sp_b, 300, lambda i: _cc(cancel=i % 3 == 0))
      self.assertEqual(a, b)


class TestStockAccCarState(unittest.TestCase):
  """accFaulted (BRAKE_ERROR) is what disengages on the VSA's live fault; stock ACC mode keeps it."""

  def _acc_faulted(self, CP, CP_SP, brake_error: bool) -> bool:
    CI = CarInterface(CP, CP_SP)
    packer = CANPacker(DBC[ELESYS_CAR][Bus.pt])
    ret = None
    for i in range(10):
      msg = packer.make_can_msg("STANDSTILL", 0, {"BRAKE_ERROR_1": int(brake_error)})
      ret, _ = CI.update([((i + 1) * int(1e7), [CanData(msg[0], bytes(msg[1]), msg[2])])])
    assert ret is not None
    return ret.accFaulted

  def test_stock_mode_reports_brake_error(self):
    CP, CP_SP = _params(params_list=[{PARAM: True}])
    self.assertTrue(self._acc_faulted(CP, CP_SP, True))
    self.assertFalse(self._acc_faulted(CP, CP_SP, False))

  def test_toggle_off_unchanged(self):
    CP, CP_SP = _params(params_list=[{PARAM: False}])
    self.assertTrue(self._acc_faulted(CP, CP_SP, True))
    self.assertFalse(self._acc_faulted(CP, CP_SP, False))

  def test_stock_long_without_the_flag_still_ignores_it(self):
    # upstream's rule for every other stock-long Honda: no BRAKE_ERROR -> accFaulted without openpilot long
    CP, CP_SP = _params(params_list=[{PARAM: True}])
    CP_SP.flags &= ~HondaFlagsSP.ELESYS_STOCK_ACC.value
    self.assertFalse(self._acc_faulted(CP, CP_SP, True))


if __name__ == "__main__":
  unittest.main()
