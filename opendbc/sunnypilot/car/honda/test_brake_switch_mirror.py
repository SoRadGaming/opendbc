"""FORK(BRAKE-SWITCH-MIRROR): the 0x17C mirror to the Elesys radar (brake_switch_mirror.py has the why).

The gate, the frame, the safety param and the CarController's sends. The panda's half - the TX allowance and the
forward block - is in safety/tests/test_honda.py (TestHondaElesysBrakeSwitchMirror*).
"""
import math
import unittest

from opendbc.can import CANPacker, CANParser
from opendbc.car import Bus, gen_empty_fingerprint, structs
from opendbc.car.honda import hondacan
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR, DBC, HONDA_ELESYS, HondaSafetyFlags as F
from opendbc.sunnypilot.car.honda.brake_switch_mirror import NO_LIGHT_MASK, brake_lamps_lit, brake_switch_mirror_on
from opendbc.sunnypilot.car.interfaces import setup_interfaces

ELESYS = CAR.HONDA_ACCORD_9G_AU
CAR_DBC = DBC[ELESYS][Bus.pt]
RADAR_BUS = 2  # bus 2 is the Elesys radar branch on this harness (hondacan.CanBus.camera)
SafetyModel = structs.CarParams.SafetyModel

# a 0x17C with something in every signal, so "every other bit is copied" means something; the foot is off the pedal
CAR_FRAME = {"PEDAL_GAS": 42, "ENGINE_RPM": 2100, "GAS_PRESSED": 1, "ACC_STATUS": 1, "BOH_17C": 5,
             "BRAKE_SWITCH": 0, "BOH2_17C": 300, "BRAKE_PRESSED": 0, "BOH3_17C": 9}


def _params(car=ELESYS, pedal=True, params_list=None):
  fp = gen_empty_fingerprint()
  fp[0][0x188] = 8          # GEARBOX_AUTO, as on the car
  if pedal:
    fp[0][0x201] = 6        # GAS_SENSOR: the comma pedal
  CP = CarInterface.get_params(car, fp, [], False, False, False)
  CP_SP = CarInterface.get_params_sp(CP, car, fp, [], False, False, False)
  setup_interfaces(CarInterface, CP, CP_SP, params_list)
  return CP, CP_SP


def _with_board(**blr):
  sp = structs.CarStateSP()
  sp.brakeLampRelay = structs.CarStateSP.BrakeLampRelay(**blr)
  return type("CS", (), {"out_sp": sp})()


def _decode(msg):
  addr, dat, bus = msg
  assert addr == 0x17C and len(dat) == 8, (hex(addr), len(dat))
  cp = CANParser(CAR_DBC, [("POWERTRAIN_DATA", math.nan)], bus)
  cp.update([(10**9, [msg])])
  return dict(cp.vl["POWERTRAIN_DATA"])


class TestBrakeLampsLit(unittest.TestCase):
  def test_only_a_fresh_driven_lit_board(self):
    self.assertTrue(brake_lamps_lit(_with_board(valid=True, outDrive=True)))
    for name, blr in (("no board", {}), ("stale", {"outDrive": True}), ("not driving", {"valid": True}),
                      ("asked for, not delivered", {"valid": True, "request": True, "wouldOn": True}),
                      ("dark", {"valid": True, "outDrive": True, "faults": NO_LIGHT_MASK})):
      self.assertFalse(brake_lamps_lit(_with_board(**blr)), name)

  def test_each_dark_diagnosis_alone_is_enough_and_no_other_fault_is(self):
    for bit in range(8):
      lit = brake_lamps_lit(_with_board(valid=True, outDrive=True, faults=1 << bit))
      self.assertEqual(lit, not ((NO_LIGHT_MASK >> bit) & 1), bit)

  def test_it_never_raises(self):
    class Hostile:
      def __getattr__(self, name):
        raise RuntimeError(name)

    for cs in (None, object(), Hostile(), type("CS", (), {"out_sp": None})(),
               _with_board(valid=True, outDrive=True, faults="not a number")):
      self.assertFalse(brake_lamps_lit(cs), repr(cs))


class TestTheFrame(unittest.TestCase):
  """Copy every bit, change the one."""

  def setUp(self):
    self.packer = CANPacker(CAR_DBC)

  def _mirror(self, frame, lit):
    msg = hondacan.create_brake_switch_on(self.packer, RADAR_BUS, dict(frame), lit)
    self.assertEqual(msg[2], RADAR_BUS)
    return _decode(msg)

  def test_only_brake_switch_ever_differs(self):
    # the foot off the pedal, then the driver's own brake: BRAKE_PRESSED is never forced
    for frame in (CAR_FRAME, dict(CAR_FRAME, BRAKE_SWITCH=1, BRAKE_PRESSED=1)):
      for lit in (False, True):
        got = self._mirror(frame, lit)
        for sig, want in (dict(frame, BRAKE_SWITCH=1) if lit else frame).items():
          self.assertAlmostEqual(got[sig], want, msg=f"{sig} {lit=} {frame['BRAKE_SWITCH']=}")

  def test_counter_and_checksum_are_the_packers(self):
    # dropped from the copy, so a stale COUNTER from the car's frame can never go out
    stale = dict(CAR_FRAME, COUNTER=3, CHECKSUM=15)
    self.assertEqual({self._mirror(stale, True)["COUNTER"] for _ in range(8)}, {0, 1, 2, 3})


class TestOnForThisVariant(unittest.TestCase):
  """No toggle: the port sets the panda's bit, and brake_switch_mirror_on() reads it back."""

  def test_this_car_with_openpilot_long(self):
    for pedal in (True, False):
      CP, _ = _params(pedal=pedal)
      self.assertEqual(CP.safetyConfigs[-1].safetyParam,  # 420
                       F.NIDEC_ALT | F.ELESYS_SCM_STANDDOWN | F.ELESYS_BRAKE_320 | F.ELESYS_BRAKE_SWITCH_MIRROR)
      self.assertTrue(brake_switch_mirror_on(CP))

  def test_not_in_stock_acc_mode(self):
    CP, _ = _params(params_list=[{"HondaElesysStockAcc": "1"}])
    self.assertEqual(CP.safetyConfigs[-1].safetyParam, 68)
    self.assertFalse(brake_switch_mirror_on(CP))

  def test_no_other_honda(self):
    for car in CAR:
      if car not in HONDA_ELESYS:
        CP, _ = _params(car=car, pedal=False)
        self.assertFalse(any(c.safetyParam & F.ELESYS_BRAKE_SWITCH_MIRROR for c in CP.safetyConfigs), car)
        self.assertFalse(brake_switch_mirror_on(CP), car)

  def test_the_reader_is_hondas_rule(self):
    # honda_nidec_init()'s rule to the bit, or the CarController and the panda disagree about who owns 0x17C
    cfg = structs.CarParams.SafetyConfig.new_message()
    cp = structs.CarParams.new_message()
    for model in (SafetyModel.hondaNidec, SafetyModel.hondaBosch):
      for param in range(512):
        cfg.safetyModel, cfg.safetyParam = model, param
        cp.safetyConfigs = [cfg]
        want = model == SafetyModel.hondaNidec and bool(param & 256) and bool(param & 32) and not (param & 64)
        self.assertEqual(brake_switch_mirror_on(cp), want, (model, param))
    on = structs.CarParams.SafetyConfig.new_message(safetyModel=SafetyModel.hondaNidec, safetyParam=420)
    cp.safetyConfigs = [on]
    self.assertTrue(brake_switch_mirror_on(cp))
    cp.safetyConfigs = [structs.CarParams.SafetyConfig.new_message(safetyModel=SafetyModel.noOutput), on]
    self.assertFalse(brake_switch_mirror_on(cp), "every config, or off")
    for odd in (None, object(), structs.CarParams.new_message()):
      self.assertFalse(brake_switch_mirror_on(odd), repr(odd))


class _CS:
  """What CarController.update() reads from CS (test_elesys's _NowCS), plus the car's 0x17C and the board."""

  def __init__(self, lamps_lit=False, seen=True):
    self.out = structs.CarState.new_message()
    self.out.vEgo = 16.0
    self.out.cruiseState.available = True
    self.out.cruiseState.speed = 16.0
    self.v_cruise_factor = 1.0
    self.stock_brake = {"CHIME": 0, "AEB_REQ_1": 0, "AEB_REQ_2": 0, "AEB_STATUS": 0}
    self.acc_hud = {"FCM_OFF": 0, "FCM_OFF_2": 0, "FCM_PROBLEM": 0, "ICONS": 0}
    self.lkas_hud = {}
    self.scm_buttons = {"CRUISE_BUTTONS": 0, "CRUISE_SETTING": 0, "MAIN_ON": 1}
    self.powertrain_data = dict(CAR_FRAME) if seen else None
    self.is_metric = True
    self.econ_on = False
    self.out_sp = structs.CarStateSP()
    self.out_sp.brakeLampRelay = structs.CarStateSP.BrakeLampRelay(valid=lamps_lit, outDrive=lamps_lit)


def _sends(mirror=True, frames=6, **cs):
  """Every frame's sends with the mirror on (this car) or off (its bit cleared, as every other config has it)."""
  CP, CP_SP = _params()
  if not mirror:
    CP.safetyConfigs[-1].safetyParam &= ~F.ELESYS_BRAKE_SWITCH_MIRROR.value
  CI = CarInterface(CP, CP_SP)
  state = _CS(**cs)
  out = []
  for i in range(frames):
    cc = structs.CarControl.new_message()
    cc.enabled = cc.latActive = cc.longActive = True
    cc.actuators.accel = -1.0
    cc.actuators.torque = 0.1
    _, sends = CI.CC.update(cc.as_reader(), structs.CarControlSP(), state, i * int(1e7))
    out.append(list(sends))
  return out


def _mirrors(sends):
  return [m for frame in sends for m in frame if m[0] == 0x17C]


def _others(sends):
  return [[m for m in frame if m[0] != 0x17C] for frame in sends]


class TestTheController(unittest.TestCase):
  def test_one_per_frame_on_the_radars_bus(self):
    sends = _sends()
    self.assertEqual([[m[2] for m in frame if m[0] == 0x17C] for frame in sends], [[RADAR_BUS]] * len(sends))

  def test_the_bit_follows_the_board(self):
    for lit in (False, True):
      for msg in _mirrors(_sends(lamps_lit=lit)):
        self.assertEqual(_decode(msg)["BRAKE_SWITCH"], int(lit))

  def test_nothing_without_the_bit_or_before_the_cars_frame(self):
    self.assertEqual(_mirrors(_sends(mirror=False)), [])
    self.assertEqual(_mirrors(_sends(seen=False)), [])

  def test_an_addition_only(self):
    # every other frame openpilot sends is the same with the mirror on and off
    self.assertEqual(_others(_sends(mirror=False)), _others(_sends()))


if __name__ == "__main__":
  unittest.main()
