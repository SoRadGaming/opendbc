"""FORK(HONDA_ACCORD_9G_AU): the brake pump rule (HondaElesysPumpC1b) and the two retired flags.

_initialize_honda (opendbc/sunnypilot/car/interfaces.py) reads the pump rule once, at ignition, into CarParamsSP.flags -
ELESYS_PUMP_C1B = 64 - for HONDA_ELESYS with openpilot longitudinal only, never in stock ACC mode. Nothing else in
CarParams or CarParamsSP may move, and with it off they are byte-identical to no hook.
Flag 16 belonged to the retired rule C1 (HondaElesysPumpV6) and flag 32 to the retired brake law v2
(HondaElesysBrakeLawV2, removed 2026-10-09): both reserved, never set, and their old keys are not read.
"""
import copy
import unittest

from opendbc.car import gen_empty_fingerprint
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR, HONDA_ELESYS
from opendbc.sunnypilot.car.honda.values_ext import HondaFlagsSP
from opendbc.sunnypilot.car.interfaces import setup_interfaces

ELESYS_CAR = CAR.HONDA_ACCORD_9G_AU
PUMP = "HondaElesysPumpC1b"
RETIRED_PUMP = "HondaElesysPumpV6"     # the retired rule C1's key
RETIRED_LAW = "HondaElesysBrakeLawV2"  # the retired brake law v2's key
STOCK_ACC = "HondaElesysStockAcc"
RETIRED = HondaFlagsSP.ELESYS_PUMP_V6 | HondaFlagsSP.ELESYS_BRAKE_LAW_V2
ALL = HondaFlagsSP.ELESYS_PUMP_C1B | RETIRED


def _params(car=ELESYS_CAR, pedal=True, params_list=None, hook=True):
  """car_helpers.get_car's order: get_params, get_params_sp, then the sunnypilot hooks."""
  fp = gen_empty_fingerprint()
  fp[0][0x188] = 8          # GEARBOX_AUTO, as on the car
  if pedal:
    fp[0][0x201] = 6        # GAS_SENSOR: the comma pedal
  CP = CarInterface.get_params(car, fp, [], False, False, False)
  CP_SP = CarInterface.get_params_sp(CP, car, fp, [], False, False, False)
  if hook:
    setup_interfaces(CarInterface, CP, CP_SP, params_list)
  return CP, CP_SP


class TestPumpBrakeLawFlags(unittest.TestCase):
  def test_values_are_the_contract(self):
    self.assertEqual(HondaFlagsSP.ELESYS_PUMP_C1B, 64)
    self.assertEqual(HondaFlagsSP.ELESYS_BRAKE_LAW_V2, 32)   # reserved: the retired brake law v2, never set again
    self.assertEqual(HondaFlagsSP.ELESYS_PUMP_V6, 16)        # reserved: the retired rule C1, never set again
    self.assertEqual(HondaFlagsSP.ELESYS_STOCK_ACC, 8)
    # one bit each, none shared
    flags = [f.value for f in HondaFlagsSP]
    self.assertEqual(len(flags), len(set(flags)))
    self.assertTrue(all(f & (f - 1) == 0 for f in flags))

  def test_off_is_byte_identical_to_no_hook(self):
    # a missing key (opendbc without openpilot) is off too: the rule before this setting
    for pedal in (True, False):
      base_cp, base_sp = _params(pedal=pedal, hook=False)
      base_bytes = base_cp.to_bytes()
      for params_list in (None, [], [{PUMP: False}], [{PUMP: "0"}], [{PUMP: 0}], [{PUMP: None}], [{PUMP: b"0"}]):
        CP, CP_SP = _params(pedal=pedal, params_list=params_list)
        self.assertEqual(CP.to_bytes(), base_bytes, msg=f"pedal={pedal} {params_list}")
        self.assertEqual(repr(CP_SP), repr(base_sp), msg=f"pedal={pedal} {params_list}")

  def test_the_setting_sets_only_its_bit(self):
    base_cp, base_sp = _params(params_list=[{PUMP: False}])
    base_bytes = base_cp.to_bytes()
    for value in (True, "1", 1, b"1"):
      CP, CP_SP = _params(params_list=[{PUMP: value}])
      self.assertEqual(CP.to_bytes(), base_bytes, msg=repr(value))
      self.assertEqual(CP_SP.flags & ALL, HondaFlagsSP.ELESYS_PUMP_C1B, msg=repr(value))
      expect_sp = copy.deepcopy(base_sp)
      expect_sp.flags |= HondaFlagsSP.ELESYS_PUMP_C1B.value
      self.assertEqual(repr(CP_SP), repr(expect_sp), msg=repr(value))

  def test_the_retired_keys_and_flags_are_never_set(self):
    # the owner's old "Quieter brake pump" and "Measured brake law (testing)" = 1 (or 0) must not reach the flags:
    # C1b has its own key, and flags 16 and 32 are dead
    for pump in (False, True):
      base_cp, base_sp = _params(params_list=[{PUMP: pump}])
      base_bytes = base_cp.to_bytes()
      for retired in (True, "1", 1, b"1", False):
        CP, CP_SP = _params(params_list=[{RETIRED_PUMP: retired, RETIRED_LAW: retired, PUMP: pump}])
        self.assertEqual(CP.to_bytes(), base_bytes, msg=f"{pump} {retired!r}")
        self.assertEqual(repr(CP_SP), repr(base_sp), msg=f"{pump} {retired!r}")
        self.assertEqual(CP_SP.flags & RETIRED, 0)

  def test_never_in_stock_acc_mode(self):
    _, CP_SP = _params(params_list=[{STOCK_ACC: "1", PUMP: True, RETIRED_LAW: True}])
    self.assertTrue(CP_SP.flags & HondaFlagsSP.ELESYS_STOCK_ACC)
    self.assertEqual(CP_SP.flags & ALL, 0)

  def test_only_with_openpilot_longitudinal(self):
    CP, CP_SP = _params(hook=False)
    CP.openpilotLongitudinalControl = False
    setup_interfaces(CarInterface, CP, CP_SP, [{PUMP: True, RETIRED_LAW: True}])
    self.assertEqual(CP_SP.flags & ALL, 0)

  def test_other_hondas_ignore_it(self):
    for car in CAR:
      if car in HONDA_ELESYS:
        continue
      _, base_sp = _params(car=car, pedal=False, params_list=None)
      CP, CP_SP = _params(car=car, pedal=False, params_list=[{PUMP: True, RETIRED_LAW: True}])
      self.assertEqual(CP_SP.flags & ALL, 0, msg=str(car))
      self.assertEqual(repr(CP_SP), repr(base_sp), msg=str(car))


if __name__ == "__main__":
  unittest.main()
