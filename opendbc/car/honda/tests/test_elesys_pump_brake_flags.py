"""FORK(HONDA_ACCORD_9G_AU): the brake pump rule (HondaElesysPumpV6) and the brake law (HondaElesysBrakeLawV2).

_initialize_honda (opendbc/sunnypilot/car/interfaces.py) reads both once, at ignition, into CarParamsSP.flags -
ELESYS_PUMP_V6 = 16 and ELESYS_BRAKE_LAW_V2 = 32 - for HONDA_ELESYS with openpilot longitudinal only, never in stock
ACC mode. Nothing else in CarParams or CarParamsSP may move, and with both off they are byte-identical to no hook.
"""
import copy
import unittest

from opendbc.car import gen_empty_fingerprint
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR, HONDA_ELESYS
from opendbc.sunnypilot.car.honda.values_ext import HondaFlagsSP
from opendbc.sunnypilot.car.interfaces import setup_interfaces

ELESYS_CAR = CAR.HONDA_ACCORD_9G_AU
PUMP = "HondaElesysPumpV6"
LAW = "HondaElesysBrakeLawV2"
STOCK_ACC = "HondaElesysStockAcc"
BOTH = HondaFlagsSP.ELESYS_PUMP_V6 | HondaFlagsSP.ELESYS_BRAKE_LAW_V2


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
    self.assertEqual(HondaFlagsSP.ELESYS_PUMP_V6, 16)
    self.assertEqual(HondaFlagsSP.ELESYS_BRAKE_LAW_V2, 32)
    self.assertEqual(HondaFlagsSP.ELESYS_STOCK_ACC, 8)
    # one bit each, none shared
    flags = [f.value for f in HondaFlagsSP]
    self.assertEqual(len(flags), len(set(flags)))
    self.assertTrue(all(f & (f - 1) == 0 for f in flags))

  def test_both_off_is_byte_identical_to_no_hook(self):
    # missing keys (opendbc without openpilot) are off too: the rule and law before these settings
    for pedal in (True, False):
      base_cp, base_sp = _params(pedal=pedal, hook=False)
      base_bytes = base_cp.to_bytes()
      for params_list in (None, [], [{PUMP: False, LAW: False}], [{PUMP: "0", LAW: "0"}], [{PUMP: 0}, {LAW: 0}],
                          [{PUMP: None, LAW: None}], [{PUMP: b"0", LAW: b"0"}]):
        CP, CP_SP = _params(pedal=pedal, params_list=params_list)
        self.assertEqual(CP.to_bytes(), base_bytes, msg=f"pedal={pedal} {params_list}")
        self.assertEqual(repr(CP_SP), repr(base_sp), msg=f"pedal={pedal} {params_list}")

  def test_each_setting_sets_only_its_bit(self):
    base_cp, base_sp = _params(params_list=[{PUMP: False, LAW: False}])
    base_bytes = base_cp.to_bytes()
    for values, expect in (((True, False), HondaFlagsSP.ELESYS_PUMP_V6), ((False, True), HondaFlagsSP.ELESYS_BRAKE_LAW_V2),
                           (("1", "1"), BOTH), ((1, 1), BOTH), ((b"1", b"1"), BOTH)):
      CP, CP_SP = _params(params_list=[{PUMP: values[0], LAW: values[1]}])
      self.assertEqual(CP.to_bytes(), base_bytes, msg=str(values))
      self.assertEqual(CP_SP.flags & BOTH, expect, msg=str(values))
      expect_sp = copy.deepcopy(base_sp)
      expect_sp.flags |= expect.value
      self.assertEqual(repr(CP_SP), repr(expect_sp), msg=str(values))

  def test_never_in_stock_acc_mode(self):
    _, CP_SP = _params(params_list=[{STOCK_ACC: "1", PUMP: True, LAW: True}])
    self.assertTrue(CP_SP.flags & HondaFlagsSP.ELESYS_STOCK_ACC)
    self.assertEqual(CP_SP.flags & BOTH, 0)

  def test_only_with_openpilot_longitudinal(self):
    CP, CP_SP = _params(hook=False)
    CP.openpilotLongitudinalControl = False
    setup_interfaces(CarInterface, CP, CP_SP, [{PUMP: True, LAW: True}])
    self.assertEqual(CP_SP.flags & BOTH, 0)

  def test_other_hondas_ignore_both(self):
    for car in CAR:
      if car in HONDA_ELESYS:
        continue
      _, base_sp = _params(car=car, pedal=False, params_list=None)
      CP, CP_SP = _params(car=car, pedal=False, params_list=[{PUMP: True, LAW: True}])
      self.assertEqual(CP_SP.flags & BOTH, 0, msg=str(car))
      self.assertEqual(repr(CP_SP), repr(base_sp), msg=str(car))


if __name__ == "__main__":
  unittest.main()
