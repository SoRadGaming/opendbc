"""FORK(HONDA_ELESYS): the gas law (elesys_gas.py) and its drive-mode plumbing, in isolation.

test_dynamic_tuning_integration.py sections [16] and [17] drive the same code through the real
CarController; this file pins the law itself: golden values, the shape properties the skeptic
review made conditions of shipping it, the crossfade bound, the slot rule and the param read.
"""
import math
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from opendbc.car import Bus, structs
from opendbc.car.honda.values import CAR, DBC
from opendbc.can.packer import CANPacker
from opendbc.sunnypilot.car.honda import elesys_gas as eg
from opendbc.sunnypilot.car.honda.gas_interceptor import GasInterceptorCarController

GearShifter = structs.CarState.GearShifter

# Duplicated on purpose, so a change to the deployed law fails here until it is re-derived.
GOLD_V1_BP = [0., 3., 6., 10., 15., 20.]
GOLD_V1_V = [0.55, 0.85, 1.20, 1.55, 1.95, 2.75]
GOLD_FF_BP = [0., 3., 6., 10., 15., 20., 25., 30.]
GOLD_FF_K = [8.73, 5.65, 6.8, 4.7, 3.4, 2.9, 2.0, 2.0]      # 0 and 3 m/s: v1's, rounded
GOLD_FF_G0 = [0.045, 0.045, 0.080, 0.102, 0.110, 0.122, 0.150, 0.211]
GOLD_NETS = [-0.5, -0.1, 0.0, 0.5, 1.0, 2.0]                 # m/s^2, after creep
GOLD_PEDAL = {                                               # v (m/s) -> pedal per GOLD_NETS
  0.: [0.0, 0.0, 0.000413, 0.057704, 0.114996, 0.229579],
  3.: [0.0, 0.0, 0.003295, 0.091836, 0.163333, 0.246667],    # 1.0 and 2.0: the launch cap, 0.08 + net/12
  6.: [0.0, 0.0, 0.016872, 0.090401, 0.163930, 0.310989],
  10.: [0.0, 0.0, 0.042838, 0.149221, 0.255604, 0.468370],
  15.: [0.0, 0.011990, 0.086990, 0.234049, 0.381107, 0.675225],
  20.: [0.0, 0.045805, 0.122000, 0.294414, 0.466828, 0.811655],
  25.: [0.0, 0.076558, 0.150000, 0.400000, 0.650000, 1.0],
  30.: [0.0, 0.126047, 0.211000, 0.461000, 0.711000, 1.0],
}
GOLD_LAUNCH_CAP_BP = [0., 3., 6.]
GOLD_LAUNCH_CAP_K = [13.0, 12.0, 6.8]
GOLD_LAUNCH_CAP_P0 = 0.08
SPEEDS = list(np.round(np.arange(0.0, 36.0, 0.25), 2))


def wb(v):
  """carcontroller.py's wind_brake, at the frozen aero scale of 1.0."""
  return float(np.interp(v, [0.0, 2.3, 35.0], [0.001, 0.002, 0.15]))


def split(net):
  """compute_gb_honda_elesys()'s gas and brake fractions for a (post-creep) net accel."""
  return max(net, 0.0) / 4.8, max(-net, 0.0) / 2.6


def v2(v, net, k_mult=1.0):
  g, b = split(net)
  return eg.elesys_pedal_v2(v, g, b, wb(v), k_mult)


def v2_uncapped(v, net, k_mult=1.0):
  """v2 as it was before the launch cap."""
  with mock.patch.object(eg, "LAUNCH_CAP_V_END", -1.0):
    return v2(v, net, k_mult)


def launch_cap(v, net, k_mult=1.0):
  """The cap's line, P0 + net / K(v), in the same net >= 0 terms as `gas`."""
  return GOLD_LAUNCH_CAP_P0 + max(net, 0.0) / (float(np.interp(v, GOLD_LAUNCH_CAP_BP, GOLD_LAUNCH_CAP_K)) * k_mult)


def v1(v, net):
  g, b = split(net)
  return eg.elesys_pedal_v1(v, g, b, wb(v))


class FakeParams:
  def __init__(self, value=None, raises=False):
    self.value, self.raises = value, raises

  def get(self, key, return_default=False):
    if self.raises:
      raise KeyError(f"UnknownKeyName: {key}")
    return self.value


class TestTables(unittest.TestCase):
  def test_tables_match_the_golden_copy(self):
    self.assertEqual(eg.ELESYS_FF_BP, GOLD_FF_BP)
    self.assertEqual(eg.ELESYS_FF_G0, GOLD_FF_G0)
    for k, gold in zip(eg.ELESYS_FF_K, GOLD_FF_K, strict=True):
      self.assertAlmostEqual(k, gold, delta=0.005)
    self.assertEqual(eg.ELESYS_GAS_BP, GOLD_V1_BP)
    self.assertEqual(eg.ELESYS_GAS_V, GOLD_V1_V)

  def test_0_and_3_ms_are_v1_exactly(self):
    # the "8.73, 5.65" of the decision are 4.8/0.55 and 4.8/0.85 rounded; the law holds the exact
    # v1 multipliers, in the same quantity v1 interpolates, so the segment is v1's bit for bit
    self.assertEqual(eg.ELESYS_FF_GM[:2], GOLD_V1_V[:2])
    for v in np.linspace(0.0, 3.0, 61):
      self.assertEqual(eg.elesys_ff_gm(float(v)), eg.elesys_gas_multiplier(float(v)), msg=f"v={v}")

  def test_measured_k_is_what_the_law_uses_above_3_ms(self):
    for v, k in zip(GOLD_FF_BP[2:], GOLD_FF_K[2:], strict=True):
      self.assertAlmostEqual(4.8 / eg.elesys_ff_gm(v), k, places=9, msg=f"v={v}")

  def test_mode_k_is_neutral_today(self):
    self.assertEqual(set(eg.MODE_K), set(eg.DRIVE_MODE_SLOTS))
    self.assertTrue(all(m == 1.0 for m in eg.MODE_K.values()))


class TestGasLawV2(unittest.TestCase):
  def test_golden_values(self):
    for v, row in GOLD_PEDAL.items():
      for net, gold in zip(GOLD_NETS, row, strict=True):
        self.assertAlmostEqual(v2(v, net), gold, delta=1e-6, msg=f"v={v} net={net}")

  def test_v1_golden_is_the_shipped_formula(self):
    for v in SPEEDS:
      for net in (-1.0, -0.05, 0.0, 0.3, 1.2):
        g, b = split(net)
        ship = float(np.clip(float(np.interp(v, GOLD_V1_BP, GOLD_V1_V)) * (g - b + wb(v) * 3 / 4), 0., 1.))
        self.assertEqual(eg.elesys_pedal_v1(v, g, b, wb(v)), ship, msg=f"v={v} net={net}")

  def test_at_or_below_3_ms_v2_is_v1_up_to_the_launch_cap(self):
    for v in np.linspace(0.0, 3.0, 61):
      for net in np.linspace(-1.0, 2.0, 61):
        v, net = float(v), float(net)
        self.assertAlmostEqual(v2_uncapped(v, net), v1(v, net), delta=1e-12, msg=f"v={v} net={net}")
        expect = float(np.clip(min(v1(v, net), launch_cap(v, net)), 0.0, 1.0))
        self.assertAlmostEqual(v2(v, net), expect, delta=1e-12, msg=f"v={v} net={net}")

  def test_continuous_at_zero(self):
    for v in SPEEDS:
      self.assertAlmostEqual(v2(v, 1e-9), v2(v, -1e-9), delta=1e-8, msg=f"v={v}")
      self.assertAlmostEqual(v2(v, 0.0), eg.elesys_ff_offset(v, wb(v)), delta=1e-15)

  def test_zero_at_the_pedal_zero_point_and_continuous_through_it(self):
    for v in SPEEDS:
      nz = -1.95 * wb(v)
      self.assertLess(v2(v, nz), 1e-12, msg=f"v={v}")
      self.assertLess(v2(v, nz + 1e-9), 1e-6, msg=f"v={v}")
      self.assertGreater(v2(v, nz * 0.5), 0.0, msg=f"v={v}")

  def test_no_pedal_at_or_below_the_brake_on_point(self):
    for v in SPEEDS:
      brake_on = -2.6 * wb(v)
      for net in (brake_on, brake_on - 0.01, brake_on - 1.0, -4.0):
        self.assertEqual(v2(v, net), 0.0, msg=f"v={v} net={net}")
        self.assertEqual(v1(v, net), 0.0, msg=f"v={v} net={net}")

  def test_monotonic_in_net(self):
    nets = np.linspace(-3.0, 3.0, 1201)
    for v in SPEEDS:
      p = [v2(v, float(n)) for n in nets]
      self.assertTrue(all(b >= a for a, b in zip(p, p[1:], strict=False)), msg=f"v={v}")

  def test_offset_never_above_v1_and_equal_to_it_below_16_8_ms(self):
    for v in SPEEDS:
      today = eg.elesys_gas_multiplier(v) * (wb(v) * 3 / 4)     # v1's offset, as v1 groups it
      off = eg.elesys_ff_offset(v, wb(v))
      self.assertLessEqual(off, today + 1e-15, msg=f"v={v}")
      if v <= 16.8:
        self.assertEqual(off, today, msg=f"v={v}")

  def test_negative_branch_window_unchanged_below_16_8_and_never_steeper(self):
    # the skeptic's condition: no high-gain window. Slope of the pedal against net across the
    # window (nz, 0), per m/s^2, compared with v1's gm1/2.6.
    for v in SPEEDS:
      nz = -1.95 * wb(v)
      lo, hi = nz * 0.75, nz * 0.25
      s2 = (v2(v, hi) - v2(v, lo)) / (hi - lo)
      s1 = (v1(v, hi) - v1(v, lo)) / (hi - lo)
      self.assertLessEqual(s2, s1 + 1e-9, msg=f"v={v}: v2 {s2:.4f} v1 {s1:.4f}")
      if v <= 16.8:
        self.assertAlmostEqual(s2, s1, delta=1e-9, msg=f"v={v}")
    # and gentler where the measured G0 takes over (0.76 against 1.06 per m/s^2 at 20 m/s)
    nz = -1.95 * wb(20.0)
    self.assertAlmostEqual((v2(20.0, 0.0) - v2(20.0, nz)) / -nz, 0.762, delta=0.002)
    self.assertAlmostEqual((v1(20.0, 0.0) - v1(20.0, nz)) / -nz, 1.058, delta=0.002)

  def test_v2_asks_less_than_v1_where_the_car_over_delivered(self):
    # pedal for a 1.0 m/s^2 request at 20 m/s: 0.742 -> 0.467 (audit P1)
    self.assertAlmostEqual(v1(20.0, 1.0), 0.742, delta=0.001)
    self.assertAlmostEqual(v2(20.0, 1.0), 0.467, delta=0.001)
    for v in (6.0, 10.0, 15.0, 20.0):
      self.assertLess(v2(v, 1.0), v1(v, 1.0), msg=f"v={v}")

  def test_k_mult_scales_the_slope_only(self):
    for v in (6.0, 15.0, 25.0):
      off = v2(v, 0.0)
      self.assertEqual(v2(v, 0.0, 2.0), off)
      self.assertAlmostEqual(v2(v, 0.5, 2.0) - off, (v2(v, 0.5) - off) / 2.0, delta=1e-12)
      for bad in (0.0, -1.0, float("nan"), None, "x"):
        self.assertEqual(v2(v, 0.5, bad), v2(v, 0.5), msg=f"k_mult={bad!r}")

  def test_never_raises_and_always_in_range(self):
    odd = (float("nan"), float("inf"), -float("inf"), None, "x", -1.0, 1e9)
    for v in odd + (10.0,):
      for x in odd + (0.2,):
        for w in odd + (0.04,):
          p = eg.elesys_pedal_v2(v, x, x, w)
          self.assertTrue(isinstance(p, float) and math.isfinite(p) and 0.0 <= p <= 1.0,
                          msg=f"v={v!r} x={x!r} wb={w!r} -> {p!r}")

  def test_no_aero_term_means_no_offset_and_no_pedal_while_braking(self):
    for w in (0.0, -0.1, float("nan")):
      self.assertEqual(eg.elesys_pedal_v2(20.0, 0.0, 0.1, w), 0.0)
      self.assertEqual(eg.elesys_pedal_v2(20.0, 0.0, 0.0, w), 0.0)
      self.assertGreater(eg.elesys_pedal_v2(20.0, 0.1, 0.0, w), 0.0)


class TestLaunchCap(unittest.TestCase):
  """The launch cap: below 6 m/s v2 never sends more than 0.08 + net / K(v). Route 115 t 511 asked 1.6-2.0 m/s^2 and
  got 2.4-2.7 from a stop; the car's launch response is a hinge (nothing below ~0.08 pedal, ~13 m/s^2 per unit
  above it at 1.5-3 m/s), which v1's line through zero over-asks above ~1 m/s^2 and under-asks below it."""

  NETS = [float(x) for x in np.linspace(-3.0, 3.0, 241)]

  def test_constants_match_the_golden_copy_and_meet_the_table(self):
    self.assertEqual(eg.LAUNCH_CAP_BP, GOLD_LAUNCH_CAP_BP)
    self.assertEqual(eg.LAUNCH_CAP_K, GOLD_LAUNCH_CAP_K)
    self.assertEqual(eg.LAUNCH_CAP_P0, GOLD_LAUNCH_CAP_P0)
    self.assertEqual(eg.LAUNCH_CAP_V_END, 6.0)
    # it ends on the v2 table's own measured k at 6 m/s, so the cap meets the table there
    self.assertEqual(eg.LAUNCH_CAP_BP[-1], eg.ELESYS_FF_BP[2])
    self.assertEqual(eg.LAUNCH_CAP_K[-1], eg.ELESYS_FF_K[2])

  def test_never_more_pedal_than_before_and_identical_from_6_ms(self):
    for v in SPEEDS:
      for net in self.NETS:
        capped, before = v2(v, net), v2_uncapped(v, net)
        self.assertLessEqual(capped, before, msg=f"v={v} net={net}")
        if v >= 6.0:
          self.assertEqual(capped, before, msg=f"v={v} net={net}")

  def test_stop_and_go_demand_and_the_braking_side_are_untouched(self):
    # binds only above the crossing with v2's line (~0.82 at 3 m/s, higher elsewhere): every demand up to 0.8,
    # and the whole pedal-zero window and brake-on point, get exactly the pedal they got before
    for v in np.arange(0.0, 6.0, 0.05):
      for net in np.linspace(-3.0, 0.8, 77):
        self.assertEqual(v2(float(v), float(net)), v2_uncapped(float(v), float(net)), msg=f"v={v} net={net}")

  def test_binds_on_a_launch(self):
    for v, net in ((1.8, 1.4), (2.2, 1.5), (3.0, 2.0), (3.8, 1.5), (4.5, 2.0)):
      self.assertAlmostEqual(v2(v, net), launch_cap(v, net), delta=1e-12, msg=f"v={v} net={net}")
      self.assertLess(v2(v, net), v2_uncapped(v, net) - 0.01, msg=f"v={v} net={net}")
    # and it stops binding before 6 m/s for anything the planner can ask (accel limit 2.0)
    for v in np.arange(5.4, 6.0, 0.01):
      for net in np.linspace(0.0, 2.0, 41):
        self.assertEqual(v2(float(v), float(net)), v2_uncapped(float(v), float(net)), msg=f"v={v} net={net}")

  def test_continuous_in_speed_and_in_net(self):
    for net in (0.5, 1.0, 1.5, 2.0, 3.0):
      for edge in (3.0, 6.0):
        self.assertAlmostEqual(v2(edge - 1e-9, net), v2(edge, net), delta=1e-6, msg=f"net={net} edge={edge}")
      p = [v2(float(v), net) for v in np.arange(0.0, 8.0, 0.005)]
      self.assertLess(max(abs(b - a) for a, b in zip(p, p[1:], strict=False)), 0.002, msg=f"net={net}")
    for v in (0.5, 2.2, 3.0, 4.0, 5.5):
      p = [v2(v, n) for n in self.NETS]
      self.assertTrue(all(b >= a for a, b in zip(p, p[1:], strict=False)), msg=f"v={v}")
      # no step where the cap takes over: on the gas side the pedal per 0.025 m/s^2 stays under v2's own slope
      gas_side = [v2(v, n) for n in self.NETS if n >= 0.0]
      self.assertLess(max(b - a for a, b in zip(gas_side, gas_side[1:], strict=False)), 0.025 * 0.85 / 4.8 + 1e-9, msg=f"v={v}")

  def test_k_mult_scales_the_cap_slope_too(self):
    for m in (0.85, 1.0, 1.3):
      self.assertAlmostEqual(v2(2.2, 1.5, m), min(v2_uncapped(2.2, 1.5, m), launch_cap(2.2, 1.5, m)), delta=1e-12)

  def test_route_115_launch(self):
    # route 115 (b29973dca) t 511.5-513.1, 0.2 s apart: vEgo, the commanded accel (actuators.accel + the
    # tuner's pitch term), the interceptor command it sent, aEgo and the planner's aTarget
    from opendbc.car.honda.carcontroller import compute_gb_honda_elesys
    frames = [
      (0.9995, 1.1331, 0.1549, 1.6244, 1.2516),
      (1.3546, 1.2743, 0.1831, 1.7620, 1.4692),
      (1.7821, 1.3786, 0.2109, 2.2127, 1.6454),
      (2.2611, 1.3780, 0.2262, 2.3595, 1.8047),
      (2.7714, 1.3349, 0.2334, 2.5945, 1.9405),
      (3.3209, 1.1699, 0.2114, 2.7584, 1.9818),
      (3.8365, 0.9927, 0.1761, 2.5654, 1.9964),
      (4.2542, 0.8556, 0.1524, 2.0641, 2.0000),
    ]
    for v, accel, sent, a_ego, a_target in frames:
      g, b = compute_gb_honda_elesys(accel, v)
      before = eg.elesys_pedal_v2(v, g, b, wb(v)) if v >= 6 else v2_uncapped(v, g * 4.8 - b * 2.6)
      now = eg.elesys_pedal_v2(v, g, b, wb(v))
      self.assertAlmostEqual(before, sent, delta=0.005, msg=f"v={v}: the replay reproduces what was sent")
      self.assertLessEqual(now, before)
      if a_ego > a_target + 0.4 and accel > 1.1:
        # where the car over-delivered by 0.4-0.8 m/s^2 on a demand above the crossing, the cap takes off 0.02-0.04
        # of pedal (~12-13 m/s^2 per unit); at 3.8 m/s the demand had already fallen to 0.99 and nothing changes
        self.assertLess(now, sent - 0.015, msg=f"v={v}")
        self.assertAlmostEqual(now, launch_cap(v, accel), delta=1e-9, msg=f"v={v}")


class TestModeSlot(unittest.TestCase):
  @staticmethod
  def cs(gear=None, econ=None):
    return SimpleNamespace(out=SimpleNamespace(gearShifter=gear), econ_on=econ)

  def test_slot_rule(self):
    self.assertEqual(eg.mode_slot(self.cs(GearShifter.sport, True)), "S")
    self.assertEqual(eg.mode_slot(self.cs(GearShifter.sport, False)), "S")
    self.assertEqual(eg.mode_slot(self.cs(GearShifter.drive, True)), "ECON")
    self.assertEqual(eg.mode_slot(self.cs(GearShifter.drive, False)), "D")
    self.assertEqual(eg.mode_slot(self.cs(GearShifter.drive, None)), "D")

  def test_unknown_and_odd_gears_are_d(self):
    for gear in (None, GearShifter.unknown, GearShifter.park, GearShifter.reverse, GearShifter.neutral, True, 999, object()):
      self.assertEqual(eg.mode_slot(self.cs(gear, False)), "D", msg=f"{gear!r}")
    self.assertEqual(eg.mode_slot(SimpleNamespace()), "D")
    self.assertEqual(eg.mode_slot(None), "D")

  def test_every_shape_of_sport_is_s(self):
    sport = structs.CarState.new_message(gearShifter=GearShifter.sport).as_reader().gearShifter
    for gear in (GearShifter.sport, sport, "sport", "GearShifter.sport"):
      self.assertEqual(eg.mode_slot(self.cs(gear, None)), "S", msg=f"{gear!r}")


class TestCrossfade(unittest.TestCase):
  def setUp(self):
    self._mode_k = dict(eg.MODE_K)

  def tearDown(self):
    eg.MODE_K.clear()
    eg.MODE_K.update(self._mode_k)

  @staticmethod
  def run_law(slots, v=15.0, net=0.8):
    law = eg.ElesysGasLaw(FakeParams(True))
    cc = SimpleNamespace(longActive=True)
    gas, brake = split(net)
    gear = {"D": GearShifter.drive, "ECON": GearShifter.drive, "S": GearShifter.sport}
    out = []
    for slot in slots:
      cs = SimpleNamespace(out=SimpleNamespace(vEgo=v, gearShifter=gear[slot]), econ_on=slot == "ECON")
      out.append(law.update(cc, cs, gas, brake, wb(v)))
    return out

  def test_a_slot_change_is_a_bounded_linear_fade(self):
    eg.MODE_K.update({"D": 1.0, "ECON": 0.7, "S": 1.3})
    p_d, p_e = v2(15.0, 0.8, 1.0), v2(15.0, 0.8, 0.7)
    out = self.run_law(["D"] * 10 + ["ECON"] * 150)
    self.assertEqual(out[9], p_d)
    steps = [abs(b - a) for a, b in zip(out[9:], out[10:], strict=False)]
    self.assertLessEqual(max(steps), abs(p_e - p_d) / eg.CROSSFADE_FRAMES + 1e-12)
    self.assertAlmostEqual(out[9 + eg.CROSSFADE_FRAMES], p_e, delta=1e-12)
    self.assertEqual(out[-1], p_e)

  def test_a_change_mid_fade_starts_from_where_the_blend_is(self):
    eg.MODE_K.update({"D": 1.0, "ECON": 0.7, "S": 1.3})
    p = {s: v2(15.0, 0.8, eg.MODE_K[s]) for s in eg.DRIVE_MODE_SLOTS}
    out = self.run_law(["D"] * 10 + ["ECON"] * 40 + ["S"] * 200)
    bound = max(abs(a - b) for a in p.values() for b in p.values()) / eg.CROSSFADE_FRAMES + 1e-12
    steps = [abs(b - a) for a, b in zip(out, out[1:], strict=False)]
    self.assertLessEqual(max(steps), bound)
    self.assertEqual(out[-1], p["S"])

  def test_equal_laws_are_bit_identical_through_any_switching(self):
    out = self.run_law(["D", "ECON", "S", "D", "S", "ECON"] * 30)
    self.assertTrue(all(x == v2(15.0, 0.8) for x in out))

  def test_the_first_frame_does_not_fade_from_nothing(self):
    eg.MODE_K.update({"D": 1.0, "ECON": 0.7, "S": 1.3})
    self.assertEqual(self.run_law(["S"])[0], v2(15.0, 0.8, 1.3))


class TestGasLawParam(unittest.TestCase):
  def test_read(self):
    for value, want in ((True, True), (False, False), ("1", True), ("0", False), (b"1", True), (b"0", False),
                        ("true", True), ("false", False), (1, True), (0, False)):
      self.assertIs(eg.read_gas_law_v2(FakeParams(value)), want, msg=f"{value!r}")

  def test_anything_unreadable_is_the_registered_default(self):
    self.assertIs(eg.GAS_LAW_DEFAULT, True)
    for p in (FakeParams(None), FakeParams("maybe"), FakeParams(raises=True), FakeParams([1])):
      self.assertIs(eg.read_gas_law_v2(p), True)

  def test_read_once(self):
    p = FakeParams(False)
    law = eg.ElesysGasLaw(p)
    p.value = True
    self.assertEqual(law.law, "v1")
    cc, cs = SimpleNamespace(longActive=True), SimpleNamespace(out=SimpleNamespace(vEgo=20.0))
    g, b = split(1.0)
    self.assertEqual(law.update(cc, cs, g, b, wb(20.0)), eg.elesys_pedal_v1(20.0, g, b, wb(20.0)))

  def test_not_long_active_is_zero(self):
    for p in (FakeParams(True), FakeParams(False)):
      law = eg.ElesysGasLaw(p)
      cs = SimpleNamespace(out=SimpleNamespace(vEgo=20.0))
      self.assertEqual(law.update(SimpleNamespace(longActive=False), cs, 0.3, 0.0, 0.08), 0.0)

  def test_update_never_raises(self):
    law = eg.ElesysGasLaw(FakeParams(True))
    for cc, cs in ((None, None), (SimpleNamespace(longActive=True), SimpleNamespace()),
                   (SimpleNamespace(longActive=True), SimpleNamespace(out=SimpleNamespace(vEgo=float("nan")))),
                   (SimpleNamespace(longActive=True), SimpleNamespace(out=SimpleNamespace(vEgo=None)))):
      p = law.update(cc, cs, 0.2, 0.0, 0.05)
      self.assertTrue(math.isfinite(p) and 0.0 <= p <= 1.0)


class TestGasInterceptorController(unittest.TestCase):
  """Only HONDA_ELESYS takes the new law; every other car is upstream's line, bit for bit,
  with or without a tuner passed in (the retired pedal gain used to reach other Nidecs too)."""

  class Tuner:
    def __init__(self):
      self.seen = []

    def observe_pedal(self, CC, CS, gas, law=""):
      self.seen.append((gas, law))

  def setUp(self):
    patcher = mock.patch.object(eg, "_open_params", lambda: FakeParams(True))
    patcher.start()
    self.addCleanup(patcher.stop)

  @staticmethod
  def controller(car):
    from opendbc.car.honda.interface import CarInterface
    CP = CarInterface.get_non_essential_params(car)
    CP_SP = CarInterface.get_non_essential_params_sp(CP, car)
    CP_SP.enableGasInterceptor = True
    return GasInterceptorCarController(CP, CP_SP), CANPacker(DBC[car][Bus.pt])

  def test_other_nidec_is_upstream_bit_for_bit(self):
    ctrl, packer = self.controller(CAR.HONDA_CIVIC)
    self.assertIsNone(ctrl.elesys_gas)
    tuner = self.Tuner()
    for i, v in enumerate(np.linspace(0.0, 30.0, 61)):
      for net in (-0.6, -0.05, 0.0, 0.4, 1.5):
        g, b = split(net)
        w = wb(float(v))
        for t in (None, tuner):
          ctrl.update(SimpleNamespace(longActive=True), SimpleNamespace(out=SimpleNamespace(vEgo=float(v))), g, b, w, packer, i, t)
          upstream = float(np.clip(np.interp(float(v), [0., 10.], [0.4, 1.0]) * (g - b + w * 3 / 4), 0., 1.))
          self.assertEqual(ctrl.gas, upstream, msg=f"v={v} net={net} tuner={t is not None}")
    self.assertTrue(tuner.seen and all(law == "nidec" for _, law in tuner.seen))

  def test_elesys_takes_its_own_law_and_reports_it(self):
    ctrl, packer = self.controller(CAR.HONDA_ACCORD_9G_AU)
    self.assertEqual(ctrl.elesys_gas.law, "v2")
    tuner = self.Tuner()
    g, b = split(1.0)
    ctrl.update(SimpleNamespace(longActive=True), SimpleNamespace(out=SimpleNamespace(vEgo=20.0)), g, b, wb(20.0), packer, 0, tuner)
    self.assertEqual(ctrl.gas, v2(20.0, 1.0))
    self.assertEqual(tuner.seen[-1], (ctrl.gas, "v2"))


if __name__ == "__main__":
  unittest.main()
