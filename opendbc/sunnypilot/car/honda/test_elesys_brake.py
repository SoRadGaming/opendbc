"""FORK(HONDA_ACCORD_9G_AU): brake law v2 (elesys_brake.py), its pedal window (elesys_gas.elesys_pedal_v2_window), the
CarController wiring (carcontroller.py) and the brake gain held at 1.0 (dynamic_tuning.py)."""
import math
import unittest
from unittest.mock import patch

import numpy as np

from opendbc.car import gen_empty_fingerprint, structs
from opendbc.car.honda.carcontroller import actuator_hysteresis, compute_gb_honda_elesys
from opendbc.car.honda.values import CAR, DBC, CarControllerParams
from opendbc.sunnypilot.car.honda import dynamic_tuning as dt
from opendbc.sunnypilot.car.honda import elesys_brake as eb
from opendbc.sunnypilot.car.honda import elesys_gas as eg
from opendbc.sunnypilot.car.honda.values_ext import HondaFlagsSP

LCS = structs.CarControl.Actuators.LongControlState
ELESYS = CAR.HONDA_ACCORD_9G_AU
V2 = HondaFlagsSP.ELESYS_BRAKE_LAW_V2.value

# Steady-state golden values from the fit's reference implementation (batch3/sim/laws.py BrakeLawV2, fit/golden.json):
# (v m/s, accel m/s^2 after the pitch term, BRAKE_COMMAND counts, interceptor command). No hysteresis, gain 1.0.
GOLDEN = [
  (5.0, 0.5, 0, 0.0961), (5.0, 0.2, 0, 0.0489), (5.0, 0.0, 0, 0.0175), (5.0, -0.2, 12, 0.0),
  (5.0, -0.4, 32, 0.0), (5.0, -0.5, 41, 0.0), (5.0, -0.6, 51, 0.0), (5.0, -1.0, 91, 0.0),
  (5.0, -2.0, 189, 0.0), (8.0, 0.5, 0, 0.134), (8.0, 0.2, 0, 0.08), (8.0, 0.0, 0, 0.044),
  (8.0, -0.2, 0, 0.0), (8.0, -0.4, 0, 0.0), (8.0, -0.5, 19, 0.0), (8.0, -0.6, 34, 0.0),
  (8.0, -1.0, 77, 0.0), (8.0, -2.0, 183, 0.0), (10.0, 0.5, 0, 0.1594), (10.0, 0.2, 0, 0.0956),
  (10.0, 0.0, 0, 0.053), (10.0, -0.2, 0, 0.0), (10.0, -0.4, 0, 0.0), (10.0, -0.5, 0, 0.0),
  (10.0, -0.6, 33, 0.0), (10.0, -1.0, 74, 0.0), (10.0, -2.0, 177, 0.0), (15.0, 0.5, 0, 0.2241),
  (15.0, 0.2, 0, 0.1358), (15.0, 0.0, 0, 0.077), (15.0, -0.2, 0, 0.0), (15.0, -0.4, 0, 0.0),
  (15.0, -0.5, 36, 0.0), (15.0, -0.6, 46, 0.0), (15.0, -1.0, 87, 0.0), (15.0, -2.0, 190, 0.0),
  (20.0, 0.5, 0, 0.2724), (20.0, 0.2, 0, 0.169), (20.0, 0.0, 0, 0.1), (20.0, -0.2, 0, 0.0063),
  (20.0, -0.4, 0, 0.0), (20.0, -0.5, 39, 0.0), (20.0, -0.6, 50, 0.0), (20.0, -1.0, 94, 0.0),
  (20.0, -2.0, 205, 0.0), (25.0, 0.5, 0, 0.37), (25.0, 0.2, 0, 0.22), (25.0, 0.0, 0, 0.12),
  (25.0, -0.2, 0, 0.0183), (25.0, -0.4, 0, 0.0), (25.0, -0.5, 46, 0.0), (25.0, -0.6, 58, 0.0),
  (25.0, -1.0, 105, 0.0), (25.0, -2.0, 224, 0.0), (30.0, 0.5, 0, 0.422), (30.0, 0.2, 0, 0.272),
  (30.0, 0.0, 0, 0.172), (30.0, -0.2, 0, 0.0598), (30.0, -0.4, 0, 0.0), (30.0, -0.5, 41, 0.0),
  (30.0, -0.6, 56, 0.0), (30.0, -1.0, 107, 0.0), (30.0, -2.0, 233, 0.0),
]


def steady(v, accel):
  """(counts, pedal) the law gives for a held accel: the reference implementation's steady state."""
  gas, _ = compute_gb_honda_elesys(accel, v)
  f = eb.BrakeLawV2Frame(accel - eb.creep(accel, v), v)
  cb = int(np.clip(f.counts(f.brake_lin), 0, 255)) if f.brake_lin > 0 else 0
  return cb, eg.elesys_pedal_v2_window(v, gas, *f.window)


class TestBrakeLawValues(unittest.TestCase):
  def test_golden(self):
    for v, a, cb, pedal in GOLDEN:
      got_cb, got_pedal = steady(v, a)
      self.assertEqual(got_cb, cb, msg=f"v={v} a={a}")
      self.assertAlmostEqual(got_pedal, pedal, places=4, msg=f"v={v} a={a}")

  def test_tables_shape(self):
    for bp, val in ((eb.COAST_BP, eb.COAST_V), (eb.C0_BP, eb.C0_V), (eb.K_BP, eb.K_V), (eb.DELTA_BP, eb.DELTA_V),
                    (eb.G0_BP, eb.G0_V)):
      self.assertEqual(len(bp), len(val))
      self.assertEqual(list(bp), sorted(bp))
    self.assertTrue(all(c - eb.TOE >= 8 for c in eb.C0_V))    # the brake-on jump: 8-32 counts
    self.assertTrue(all(0.7 < k < 1.2 for k in eb.K_V))

  def test_today_constants_are_carcontrollers(self):
    self.assertEqual(eb.BRAKE_COUNTS, CarControllerParams.NIDEC_BRAKE_MAX)
    # net = accel - creep is exactly what compute_gb_honda_elesys splits into gas/4.8 and brake/2.6
    for v in (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 4.5, 5.0, 8.0):
      for a in (-2.0, -1.0, -0.3, 0.0, 0.2, 0.5, 0.79, 1.5):
        gas, brake = compute_gb_honda_elesys(a, v)
        if gas >= 1.0 or brake >= 1.0:
          continue
        self.assertAlmostEqual(gas * 4.8 - brake * eb.FULL_BRAKE_ACCEL, a - eb.creep(a, v), places=12)
    # carcontroller.py's inline wind_brake
    for v in (0.0, 1.0, 2.3, 10.0, 35.0):
      self.assertEqual(eb.wind_brake(v), float(np.interp(v, [0.0, 2.3, 35.0], [0.001, 0.002, 0.15])))

  def test_steady_counts_today_vs_v2_where_it_matters(self):
    # -0.5 m/s^2 at 10 m/s: today brakes 39 counts against a car that coasts at -0.51; v2 sends nothing
    self.assertEqual(steady(10.0, -0.5), (0, 0.0))
    # above 25 m/s today under-brakes (bias +0.355); v2 asks for more
    self.assertGreater(steady(30.0, -1.0)[0], 100)


class TestDeadZoneAndCoastBand(unittest.TestCase):
  def test_coast_band_sends_nothing(self):
    for v in np.arange(eb.V_HI, 35.0, 0.5):
      zb, _, _, _, _, zp = eb.law_point(v)
      self.assertLess(zb, zp)
      for net in np.linspace(zb + 1e-6, zp - 1e-6, 7):
        f = eb.BrakeLawV2Frame(net, v)
        self.assertEqual(f.brake_lin, 0.0)
        self.assertEqual(f.brake_frac(f.brake_lin), 0.0)
        self.assertEqual(eg.elesys_pedal_v2_window(v, 0.0, *f.window), 0.0)

  def test_hysteresis_keeps_coasting_targets_off_the_brake(self):
    # a target a little below coast(v) stays inside actuator_hysteresis's 0.02 on-threshold: no brake at all
    for v in (8.0, 12.0, 20.0, 28.0):
      zb = eb.law_point(v)[0]
      f = eb.BrakeLawV2Frame(zb - 0.04, v)
      self.assertEqual(actuator_hysteresis(f.brake_lin, False, 0.)[0], 0.)

  def test_brake_on_jump_is_the_low_end_of_the_dead_zone(self):
    for v in (6.0, 7.5, 12.5, 17.5, 22.5, 27.5, 33.0):
      _, c0, k, toe, _, _ = eb.law_point(v)
      f = eb.BrakeLawV2Frame(-3.0, v)
      self.assertAlmostEqual(f.counts(1e-12), c0 - toe, places=3)
      self.assertEqual(f.counts(0.0), 0.0)

  def test_soft_hinge_round_trip_and_smooth(self):
    for c0, toe in ((18., 10.), (42., 10.), (30., 0.)):
      for u in (0.01, 1., 5., 9.99, 10., 10.01, 50., 200.):
        self.assertAlmostEqual(eb.soft_hinge(eb.soft_hinge_inverse(u, c0, toe), c0, toe), u, places=9)
      # value and slope continuous at the end of the toe
      if toe > 0:
        e = 1e-6
        self.assertAlmostEqual(eb.soft_hinge_inverse(toe - e, c0, toe), eb.soft_hinge_inverse(toe + e, c0, toe), places=4)
        lo = (eb.soft_hinge_inverse(toe, c0, toe) - eb.soft_hinge_inverse(toe - 1e-3, c0, toe)) / 1e-3
        hi = (eb.soft_hinge_inverse(toe + 1e-3, c0, toe) - eb.soft_hinge_inverse(toe, c0, toe)) / 1e-3
        self.assertAlmostEqual(lo, hi, places=2)


class TestMonotonicity(unittest.TestCase):
  def test_counts_rise_with_brake_and_with_decel(self):
    for v in (4.5, 6.0, 9.0, 15.0, 22.0, 30.0):
      f = eb.BrakeLawV2Frame(-2.0, v)
      c = [f.counts(b) for b in np.linspace(0., 1., 401)]
      self.assertTrue(all(y >= x for x, y in zip(c, c[1:], strict=False)))
      cbs = [steady(v, a)[0] for a in np.linspace(0.5, -3.5, 161)]
      self.assertTrue(all(y >= x for x, y in zip(cbs, cbs[1:], strict=False)), msg=f"v={v}")

  def test_pedal_rises_with_net(self):
    for v in (4.2, 5.0, 6.0, 9.0, 15.0, 22.0, 30.0):
      ped = [steady(v, a)[1] for a in np.linspace(-1.0, 2.0, 301)]
      self.assertTrue(all(y >= x - 1e-12 for x, y in zip(ped, ped[1:], strict=False)), msg=f"v={v}")

  def test_never_brake_and_pedal_together(self):
    for v in np.arange(4.05, 35.0, 0.5):
      for a in np.linspace(-3.0, 2.0, 101):
        cb, ped = steady(v, a)
        self.assertFalse(cb > 0 and ped > 0, msg=f"v={v} a={a}")


class TestContinuity(unittest.TestCase):
  def test_parameters_are_todays_at_v_lo(self):
    v = eb.V_LO
    wb = eb.wind_brake(v)
    zb, c0, k, toe, off, zp = eb.law_point(v)
    self.assertAlmostEqual(zb, -2.6 * wb, places=12)
    self.assertAlmostEqual(zp, -1.95 * wb, places=12)
    self.assertEqual((c0, k, toe), (0., 1., 0.))
    self.assertEqual(off, eg.elesys_ff_offset(v, wb))

  def test_pedal_at_v_lo_is_gas_law_v2(self):
    v = eb.V_LO
    for a in (1.5, 0.5, 0.1, 0.0, -0.002, -0.005, -0.05, -0.5):
      gas, brake = compute_gb_honda_elesys(a, v)
      f = eb.BrakeLawV2Frame(a - eb.creep(a, v), v)
      self.assertAlmostEqual(eg.elesys_pedal_v2_window(v, gas, *f.window),
                             eg.elesys_pedal_v2(v, gas, brake, eb.wind_brake(v)), places=9)

  def test_counts_at_v_lo_within_the_aero_credit(self):
    # today subtracts wb (0.0026 at 4 m/s) from the brake; v2 at its blend start shifts the brake-on point by it
    v = eb.V_LO + 1e-9
    for a in (-0.3, -0.6, -1.0, -2.0):
      _, brake = compute_gb_honda_elesys(a, v)
      today = max(brake - eb.wind_brake(v), 0.) * 256
      f = eb.BrakeLawV2Frame(a - eb.creep(a, v), v)
      self.assertLessEqual(abs(f.counts(f.brake_lin) - today), 1.5)

  def test_no_steps_across_speed(self):
    vs = np.arange(4.0 + 1e-6, 35.0, 0.01)
    for a in (-0.3, -0.6, -1.0, -1.5, -2.5, 0.0, 0.4, 1.0):
      cbs, lin, eff, peds = [], [], [], []
      for v in vs:
        gas, _ = compute_gb_honda_elesys(a, v)
        f = eb.BrakeLawV2Frame(a - eb.creep(a, v), v)
        cb = f.counts(f.brake_lin)
        cbs.append(cb)
        lin.append(cb >= f.c0 + f.toe)
        eff.append(eb.soft_hinge(cb, f.c0, f.toe) * f.k * 2.6 / 256)   # the decel the law means to add over coast
        peds.append(eg.elesys_pedal_v2_window(v, gas, *f.window))
      cbs, lin = np.array(cbs), np.array(lin)
      # what the law asks the brake to deliver is continuous in speed ...
      self.assertLess(np.max(np.abs(np.diff(eff))), 0.01, msg=f"a={a}")
      # ... and so are the counts outside the toe, whose square root is steep near brake-on by design
      both = lin[:-1] & lin[1:]
      self.assertLess(np.max(np.abs(np.diff(cbs))[both], initial=0.), 1.0, msg=f"a={a}")
      # the brake-on/off edge (0 <-> c0 - toe) happens only where coast(v) crosses this target
      self.assertLessEqual(np.sum((cbs[:-1] > 0) != (cbs[1:] > 0)), 2, msg=f"a={a}")
      self.assertLess(np.max(np.abs(np.diff(peds))), 2e-3, msg=f"a={a}")


class TestLawFrame(unittest.TestCase):
  def test_where_todays_path_runs(self):
    self.assertIsNotNone(eb.law_frame(-1.0, 10.0, True, LCS.pid))
    self.assertIsNone(eb.law_frame(-1.0, 10.0, False, LCS.pid))
    for lcs in (LCS.off, LCS.stopping, LCS.starting):
      self.assertIsNone(eb.law_frame(-1.0, 10.0, True, lcs))
    self.assertIsNone(eb.law_frame(-1.0, eb.V_LO, True, LCS.pid))
    self.assertIsNone(eb.law_frame(-1.0, 2.0, True, LCS.pid))
    for bad in (float('nan'), float('inf'), None, "x"):
      self.assertIsNone(eb.law_frame(bad, 10.0, True, LCS.pid))
      self.assertIsNone(eb.law_frame(-1.0, bad, True, LCS.pid))

  def test_brake_frac_never_raises_and_stays_in_range(self):
    f = eb.law_frame(-2.0, 25.0, True, LCS.pid)
    for b in (float('nan'), float('inf'), -1.0, 0.0, 0.5, 1.0, 3.0, None):
      x = f.brake_frac(b)
      self.assertTrue(math.isfinite(x) and 0.0 <= x <= 1.0)
    self.assertEqual(f.brake_frac(float('nan')), 0.0)
    self.assertEqual(f.brake_frac(5.0), f.brake_frac(1.0))

  def test_brake_frac_is_exact_counts(self):
    f = eb.law_frame(-1.2, 17.0, True, LCS.pid)
    for b in np.linspace(0.01, 1.0, 50):
      c = f.counts(b)
      self.assertEqual(int(np.clip(f.brake_frac(b) * 1.0 * 256, 0, 255)), int(np.clip(c, 0, 255)))


# --- CarController -----------------------------------------------------------------------------------------------
class _Params:
  def __init__(self, store):
    self.store = dict(store)

  def get(self, key, block=False, return_default=False):
    if key in self.store:
      return self.store[key]
    return dt._PARAM_SPEC[key][0] if key in dt._PARAM_SPEC else None

  def get_bool(self, key, block=False):
    return str(self.store.get(key, "0")) in ("1", "True", "true")

  def put(self, key, value, block=False):
    self.store[key] = value


class _CS:
  def __init__(self):
    self.out = structs.CarState.new_message()
    self.out.cruiseState.available = True
    self.v_cruise_factor = 1.0
    self.stock_brake = {"CHIME": 0, "AEB_REQ_1": 0, "AEB_REQ_2": 0, "AEB_STATUS": 0}
    self.acc_hud = {"FCM_OFF": 0, "FCM_OFF_2": 0, "FCM_PROBLEM": 0, "ICONS": 0}
    self.lkas_hud = {}
    self.scm_buttons = {"CRUISE_BUTTONS": 0, "CRUISE_SETTING": 0}
    self.is_metric = True
    self.econ_on = False
    self.out_sp = structs.CarStateSP()


def _controller(flags, gas_law_v2=True, tuner=False, brake_gain=0.0, car=ELESYS):
  from opendbc.car.honda.carcontroller import CarController
  from opendbc.car.honda.interface import CarInterface
  store = _Params({"HondaElesysGasLawV2": "1" if gas_law_v2 else "0", "HondaDynamicTuningEnabled": "1" if tuner else "0",
                   "HondaDynBrakeGain": brake_gain})
  fp = gen_empty_fingerprint()
  fp[0][0x188] = 8   # GEARBOX_AUTO
  fp[0][0x201] = 6   # the comma pedal
  CP = CarInterface.get_params(car, fp, [], False, False, False)
  CP_SP = CarInterface.get_params_sp(CP, car, fp, [], False, False, False)
  CP_SP.flags |= flags
  with patch.object(dt, "_open_params", lambda: store), patch.object(eg, "_open_params", lambda: store):
    return CarController(DBC[car], CP, CP_SP), store


# (seconds, vEgo, accel, longControlState)
DRIVE = [(3.0, 10.0, -1.0, LCS.pid), (3.0, 10.0, -0.4, LCS.pid), (2.0, 10.0, 0.3, LCS.pid), (3.0, 25.0, -2.0, LCS.pid),
         (2.0, 5.0, -0.6, LCS.pid), (2.0, 3.0, -0.8, LCS.pid), (1.0, 0.5, -0.8, LCS.stopping), (2.0, 0.0, -0.8, LCS.stopping)]
LOW = [(2.0, 3.5, -0.5, LCS.pid), (2.0, 2.0, 0.3, LCS.pid), (1.5, 0.5, -0.8, LCS.stopping), (2.0, 0.0, -0.8, LCS.stopping),
       (1.0, 0.0, 0.5, LCS.starting), (1.0, 1.0, 0.8, LCS.pid)]


def _drive(cc_obj, plan):
  out = []
  cs = _CS()
  ccsp = structs.CarControlSP()
  i = 0
  for seconds, v, a, lcs in plan:
    for _ in range(round(seconds * 100)):
      cc = structs.CarControl.new_message()
      cc.enabled = cc.longActive = True
      cc.actuators.accel = a
      cc.actuators.longControlState = lcs
      cs.out.vEgo = v
      cs.out.standstill = v == 0.0
      _, sends = cc_obj.update(cc.as_reader(), ccsp, cs, int(i * 1e7))
      sent = tuple(sorted((s[0], bytes(s[1]), s[2]) if isinstance(s, tuple) else (s.address, bytes(s.dat), s.src)
                          for s in sends))
      out.append((v, a, sent, cc_obj.apply_brake_last, cc_obj.gas))
      i += 1
  return out


def _at(out, v, a):
  """the last frame of the plan step at (v, a): settled"""
  return [o for o in out if o[0] == v and o[1] == a][-1]


class TestCarControllerBrakeLaw(unittest.TestCase):
  def test_flag_clear_never_builds_a_frame(self):
    cc_obj, _ = _controller(0)
    self.assertFalse(cc_obj.elesys_brake_v2)
    with patch("opendbc.car.honda.carcontroller.law_frame", side_effect=AssertionError("called")):
      _drive(cc_obj, DRIVE)
    self.assertIsNone(cc_obj.elesys_gas.window)

  def test_flag_clear_is_todays_law(self):
    # steady on today's path: counts = (brake - hysteresis gap - wb) * 256, pedal = gas law v2
    out = _drive(_controller(0)[0], DRIVE)
    _, brake = compute_gb_honda_elesys(-1.0, 10.0)
    self.assertEqual(_at(out, 10.0, -1.0)[3], int((brake - 0.01 - eb.wind_brake(10.0)) * 256))
    self.assertGreater(_at(out, 10.0, -0.4)[3], 20)          # today brakes where the car would coast

  def test_flag_with_gas_law_v1_is_the_flag_clear_drive(self):
    cc_obj, _ = _controller(V2, gas_law_v2=False)
    self.assertFalse(cc_obj.elesys_brake_v2)
    self.assertEqual(cc_obj.dynamic_tuner.build.get("blaw"), "v1")
    self.assertEqual(_drive(cc_obj, DRIVE), _drive(_controller(0, gas_law_v2=False)[0], DRIVE))

  def test_flag_set_below_v_lo_and_stopping_is_todays_law(self):
    self.assertEqual(_drive(_controller(V2)[0], LOW), _drive(_controller(0)[0], LOW))
    # with the tuner on the only difference is the scalar gain, held at 1.0 for the whole drive (it would otherwise
    # learn here: 66 against 67 counts at 3.5 m/s in this plan); today's law with the gain held gives the same frames
    held, _ = _controller(0, tuner=True)
    held.dynamic_tuner.brake_law_v2 = True
    self.assertEqual(_drive(_controller(V2, tuner=True)[0], LOW), _drive(held, LOW))

  def test_flag_set_runs_v2(self):
    cc_obj, _ = _controller(V2)
    self.assertTrue(cc_obj.elesys_brake_v2)
    out = _drive(cc_obj, DRIVE)
    # actuator_hysteresis settles 0.01 below a target approached from below, 0.01 above one approached from above
    for v, a, gap in ((10.0, -1.0, -0.01), (25.0, -2.0, -0.01), (5.0, -0.6, 0.01)):
      f = eb.law_frame(a, v, True, LCS.pid)
      self.assertEqual(_at(out, v, a)[3], int(np.clip(f.counts(f.brake_lin + gap), 0, 255)), msg=f"v={v} a={a}")
    # the coast band: no brake and no pedal at -0.4 m/s^2, 10 m/s
    self.assertEqual(_at(out, 10.0, -0.4)[3:], (0, 0.0))
    # the pedal is the moved window's
    gas, _ = compute_gb_honda_elesys(0.3, 10.0)
    self.assertAlmostEqual(_at(out, 10.0, 0.3)[4], eg.elesys_pedal_v2_window(10.0, gas, *eb.law_frame(0.3, 10.0, True,
                                                                                                         LCS.pid).window))
    self.assertNotEqual(_at(out, 10.0, 0.3)[4], _at(_drive(_controller(0)[0], DRIVE), 10.0, 0.3)[4])
    # and the stop is today's
    today = _drive(_controller(0)[0], DRIVE)
    self.assertEqual(_at(out, 0.0, -0.8)[3:], _at(today, 0.0, -0.8)[3:])

  def test_brake_gain_held_at_one_and_the_stored_gain_kept(self):
    # tuner on with a learned +0.3: today's law brakes 1.3x, v2 ignores it, and the stored value survives
    on, store = _controller(V2, tuner=True, brake_gain=0.3)
    off, _ = _controller(V2, tuner=False)
    today, _ = _controller(0, tuner=True, brake_gain=0.3)
    plan = [(3.0, 10.0, -1.0, LCS.pid)]
    self.assertEqual(_drive(on, plan)[-1][3], _drive(off, plan)[-1][3])
    self.assertGreater(_drive(today, plan)[-1][3], _drive(_controller(0)[0], plan)[-1][3])
    tu = on.dynamic_tuner
    self.assertTrue(tu.brake_law_v2)
    self.assertEqual(tu.brake_gain_converged, 0.3)
    self.assertEqual(tu.debug_values()["brake_gain"], 1.0)
    self.assertEqual(tu._persist_values()["HondaDynBrakeGain"], 0.3)
    self.assertEqual(tu.build.get("blaw"), "v2")

  def test_enabled_only_where_it_belongs(self):
    self.assertFalse(_controller(0)[0].elesys_brake_v2)
    self.assertFalse(_controller(V2 | HondaFlagsSP.ELESYS_STOCK_ACC.value)[0].elesys_brake_v2)
    self.assertFalse(_controller(V2, car=CAR.HONDA_CIVIC)[0].elesys_brake_v2)
    self.assertTrue(_controller(V2 | HondaFlagsSP.ELESYS_PUMP_C1B.value)[0].elesys_brake_v2)

  def test_non_finite_inputs_never_raise(self):
    # update() must never raise (no 0x1FA -> BRAKE_ERROR): NaN/inf speed or accel, mixed into a v2 drive
    for tuner in (False, True):
      cc_obj, _ = _controller(V2, tuner=tuner)
      nan, inf = float('nan'), float('inf')
      plan = [(0.5, 10.0, -1.0, LCS.pid), (0.2, nan, -1.0, LCS.pid), (0.5, 10.0, -1.0, LCS.pid), (0.2, 10.0, nan, LCS.pid),
              (0.2, inf, -1.0, LCS.pid), (0.2, 10.0, -inf, LCS.pid), (0.5, 10.0, -1.0, LCS.pid)]
      out = _drive(cc_obj, plan)
      for i, o in enumerate(out):
        f = [s for s in o[2] if s[0] == 0x1FA]
        self.assertEqual(len(f), 1 if i % 2 == 0 else 0)
        self.assertTrue(0 <= o[3] <= 255)
        self.assertTrue(math.isfinite(o[4]) and 0.0 <= o[4] <= 1.0)

  def test_set_brake_law_v2_is_a_no_op_when_it_agrees(self):
    cc_obj, _ = _controller(0, tuner=True)
    tu = cc_obj.dynamic_tuner
    before = dict(tu.build)
    tu.set_brake_law_v2(False)
    self.assertEqual(tu.build, before)
    self.assertFalse(tu.brake_law_v2)


class TestPedalWindow(unittest.TestCase):
  def test_mode_multiplier_and_bad_input(self):
    self.assertEqual(eg.elesys_pedal_v2_window(10.0, 0.1, 0.48, 0.05, -0.15, 0.0), eg.elesys_pedal_v2_window(10.0, 0.1,
                                                                                                            0.48, 0.05,
                                                                                                            -0.15, 1.0))
    for bad in (float('nan'), None, "x"):
      self.assertEqual(eg.elesys_pedal_v2_window(10.0, 0.1, bad, 0.05, -0.15), 0.0)
    self.assertEqual(eg.elesys_pedal_v2_window(10.0, 0.0, -0.2, 0.05, -0.15), 0.0)
    self.assertAlmostEqual(eg.elesys_pedal_v2_window(10.0, 0.0, -0.075, 0.05, -0.15), 0.025)

  def test_gas_law_object_window(self):
    law = eg.ElesysGasLaw(params=_Params({"HondaElesysGasLawV2": "1"}))
    cs = _CS()
    cs.out.vEgo = 10.0
    cc = structs.CarControl.new_message()
    cc.longActive = True
    base = law.update(cc.as_reader(), cs, 0.05, 0.0, eb.wind_brake(10.0))
    law.window = (0.24, 0.053, -0.157)
    self.assertAlmostEqual(law.update(cc.as_reader(), cs, 0.05, 0.0, eb.wind_brake(10.0)),
                           eg.elesys_pedal_v2_window(10.0, 0.05, 0.24, 0.053, -0.157))
    law.window = None
    self.assertEqual(law.update(cc.as_reader(), cs, 0.05, 0.0, eb.wind_brake(10.0)), base)


if __name__ == "__main__":
  unittest.main()
