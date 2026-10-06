import unittest

import numpy as np

from types import SimpleNamespace

from opendbc.car import Bus
from opendbc.car.honda import hondacan
from opendbc.car.honda.carcontroller import (brake_pump_hysteresis, brake_pump_hysteresis_elesys,
                                             brake_pump_c1b_elesys,  # FORK(HONDA_ACCORD_9G_AU): pump rule C1b
                                             compute_gas_brake, compute_gb_honda_elesys,
                                             compute_gb_honda_nidec)
from opendbc.sunnypilot.car.honda.gas_interceptor import elesys_gas_multiplier

# golden values measured for the 9G Accord AU (redlight_overshoot_findings.md) -- deliberately
# duplicated here so any change to the deployed mapping fails a test until re-measured
GOLD_BRAKE_SCALE = 2.6                          # m/s^2 at full COMPUTER_BRAKE
GOLD_GAS_SCALE = 4.8
GOLD_CREEP_BP = [0., 0.75, 1.75, 3.0, 5.0]      # grade-corrected coastdown, 76 routes
GOLD_CREEP_V = [1.15, 0.8, 0.45, 0.3, 0.0]
from opendbc.car.honda.values import CAR, DBC, HONDA_BOSCH, HONDA_ELESYS

ELESYS_CAR = CAR.HONDA_ACCORD_9G_AU
NIDEC_CAR = CAR.HONDA_CRV  # plain Nidec, no interceptor assumptions in these unit tests


class TestElesysCategory(unittest.TestCase):
  def test_membership(self):
    self.assertTrue(ELESYS_CAR in HONDA_ELESYS)
    self.assertFalse(ELESYS_CAR in HONDA_BOSCH)
    self.assertFalse(NIDEC_CAR in HONDA_ELESYS)
    # every Elesys car needs a brake scale (or falls back to the default) and must not be Bosch
    for car in HONDA_ELESYS:
      self.assertFalse(car in HONDA_BOSCH)

  def test_dispatch(self):
    # Elesys car routes to the Elesys mapping, Nidec car to upstream mapping.
    # compute_gas_brake() takes a CP upstream; it reads only carFingerprint and flags.
    def cp(car):
      return SimpleNamespace(carFingerprint=car, flags=car.config.flags)
    a, v = -1.5, 7.0
    self.assertEqual(compute_gas_brake(a, v, cp(ELESYS_CAR)), compute_gb_honda_elesys(a, v))
    self.assertEqual(compute_gas_brake(a, v, cp(NIDEC_CAR)), tuple(compute_gb_honda_nidec(a, v)))


class TestComputeGbNidec(unittest.TestCase):
  """The upstream ILX-fitted mapping must be exactly upstream again (no fork leakage)."""

  def test_upstream_formula(self):
    for accel in (-3.0, -1.0, -0.3, 0.0, 0.5, 2.0):
      for speed in (0.0, 1.0, 2.2, 2.3, 10.0):
        creep = max(0.0, (2.3 - speed) / 2.3 * 0.15) if speed < 2.3 else 0.0
        gb = accel / 4.8 - creep
        gas, brake = compute_gb_honda_nidec(accel, speed)
        self.assertAlmostEqual(gas, float(np.clip(gb, 0.0, 1.0)), places=10)
        self.assertAlmostEqual(brake, float(np.clip(-gb, 0.0, 1.0)), places=10)

  def test_signature_is_upstream(self):
    import inspect
    self.assertEqual(list(inspect.signature(compute_gb_honda_nidec).parameters), ['accel', 'speed'])


class TestComputeGbElesys(unittest.TestCase):
  def test_creep_offset_near_stop(self):
    # measured: a small planner decel near standstill must still command a real brake,
    # because the car self-propels at ~1 m/s^2 (torque-converter creep)
    gas, brake = compute_gb_honda_elesys(-0.29, 0.65)
    self.assertEqual(gas, 0.0)
    self.assertGreater(brake * 256, 100)  # above the measured effectiveness threshold

  def test_mid_speed_brake_scale(self):
    # above the creep band the mapping is a pure rescale: accel/2.6
    gas, brake = compute_gb_honda_elesys(-1.5, 7.0)
    self.assertEqual(gas, 0.0)
    self.assertAlmostEqual(brake, 1.5 / GOLD_BRAKE_SCALE, places=6)

  def test_gas_side(self):
    gas, brake = compute_gb_honda_elesys(1.0, 10.0)
    self.assertEqual(brake, 0.0)
    self.assertAlmostEqual(gas, 1.0 / 4.8, places=6)

  def test_mutually_exclusive_and_clipped(self):
    for accel in np.arange(-4.0, 2.5, 0.25):
      for speed in (0.0, 0.5, 1.0, 2.0, 5.0, 15.0):
        gas, brake = compute_gb_honda_elesys(float(accel), speed)
        self.assertGreaterEqual(min(gas, brake), 0.0)
        self.assertLessEqual(max(gas, brake), 1.0)
        self.assertEqual(min(gas, brake), 0.0)  # never both

  def test_stopping_ramp_saturates(self):
    # stopAccel (-2.0) near standstill must be full brake (creep + 2.0 > 2.6)
    _, brake = compute_gb_honda_elesys(-2.0, 0.4)
    self.assertEqual(brake, 1.0)

  def test_light_gas_still_brakes_in_creep_band(self):
    # wanting +0.3 m/s^2 at crawl speed means braking: the car creeps harder than that on its own
    gas, brake = compute_gb_honda_elesys(0.3, 0.5)
    self.assertEqual(gas, 0.0)
    self.assertGreater(brake, 0.0)

  def test_launch_crossover_fades_creep(self):
    # at standstill the brake->gas crossover must sit near 0.5 m/s^2 of demand, not the full
    # creep offset (1.15) -- holding the brakes until a_cmd > creep caused ~1.2 s launch lag
    gas, brake = compute_gb_honda_elesys(0.55, 0.0)
    self.assertEqual(brake, 0.0)
    self.assertGreater(gas, 0.0)
    # below the crossover we still hold some brake (hill/creep hold)
    gas, brake = compute_gb_honda_elesys(0.2, 0.0)
    self.assertEqual(gas, 0.0)
    self.assertGreater(brake, 0.0)

  def test_strong_demand_has_no_creep_subtraction(self):
    # once demand exceeds the fade band, gas must be the plain scale (no double-counting creep)
    gas, brake = compute_gb_honda_elesys(1.6, 1.5)
    self.assertEqual(brake, 0.0)
    self.assertAlmostEqual(gas, 1.6 / GOLD_GAS_SCALE, places=6)

  def test_braking_path_matches_golden_table(self):
    # the fade must not alter any braking behavior (accel <= 0): pure golden-table offset
    for accel in np.arange(-3.0, 0.001, 0.125):
      for speed in np.arange(0.0, 6.0, 0.25):
        creep = float(np.interp(speed, GOLD_CREEP_BP, GOLD_CREEP_V))
        expect = float(np.clip(-(accel - creep) / GOLD_BRAKE_SCALE, 0.0, 1.0))
        _, brake = compute_gb_honda_elesys(float(accel), float(speed))
        self.assertAlmostEqual(brake, expect, places=9)

  def test_net_monotonic_no_overlap(self):
    # net demand must be monotonic in accel (single crossover, gas and brake can't fight)
    for speed in (0.0, 0.5, 1.0, 2.0, 4.0):
      last_net = -1e9
      for accel in np.arange(-2.0, 2.01, 0.05):
        gas, brake = compute_gb_honda_elesys(float(accel), speed)
        net = gas * GOLD_GAS_SCALE - brake * GOLD_BRAKE_SCALE
        self.assertGreaterEqual(net, last_net - 1e-9)
        last_net = net

  def test_deployed_creep_matches_golden_table(self):
    # probe the deployed table through the function: at accel=0 the fade factor is 1, so
    # brake * GOLD_BRAKE_SCALE == creep(v). Pins the inline values against the measurement.
    for v in list(GOLD_CREEP_BP) + [0.375, 1.25, 2.375, 4.0, 6.0, 10.0]:
      expect = float(np.interp(v, GOLD_CREEP_BP, GOLD_CREEP_V))
      _, brake = compute_gb_honda_elesys(0.0, float(v))
      self.assertAlmostEqual(brake * GOLD_BRAKE_SCALE, expect, places=6, msg=f'v={v}')


class TestBrakePumpHysteresis(unittest.TestCase):
  """Upstream pump logic must stay pristine; the Elesys variant trades the 20 s bleed window
  for short runs (0.5 s) at a load-scaled period with a jitter deadband.

  Two independent mechanisms, and they must not be confused:

  1. The CONTINUOUS-RUN branches (v >= 2.5 and cb > 200, or 0.15 <= v < 2.5 and cb > 100) pin
     the pump on at firm braking. A v4 experiment deleted them on the grounds that they were
     rare; that silently reverted commit 2905e73d1 and cost duty exactly where demand is
     highest. Measured v3 -> v4 on the real 13-route stream: at cb >= 200 duty 1.00 -> 0.32 and
     worst pump-off gap 0.16 s -> 5.50 s. RESTORED.

  2. The GRADED DEADBAND ([12, 6, 3] over cb [0, 60, 200]) governs LIGHT braking, where the
     audible whine actually lives -- `rise and in_run` re-primes inside the current run, so a
     slowly-drifting command used to pin the pump on. Cuts light-braking run time 15.75 ->
     12.10 min with no change to the worst gap.

  The two are orthogonal, which is why both are in. Total pump run is still 38.1 min against
  v3's 58.9 on the same stream."""

  def _duty_elesys(self, cb, seconds=20.0, jitter=0, v_ego=5.0):
    anchor, last_pump_ts, on = 0, -99.0, 0
    n = int(seconds * 100)
    for i in range(n):
      c = cb + (jitter if (i // 25) % 2 else -jitter)   # slow +-jitter square wave
      pump, anchor, last_pump_ts = brake_pump_hysteresis_elesys(c, v_ego, anchor, last_pump_ts, i * 0.01)
      on += pump
    return on / n

  def test_steady_duty_backstop_only(self):
    # perfectly steady command: only the periodic backstop runs (0.5 s per 12->6 s period)
    self.assertAlmostEqual(self._duty_elesys(60), 0.5 / 11.25, delta=0.02)
    self.assertAlmostEqual(self._duty_elesys(200), 0.5 / 6.0, delta=0.02)

  def test_steady_duty_is_lower_than_v3(self):
    # the whole point of v4: steady-state duty must be well under the v3 values it replaces
    # (v3 was 1.0 s per 8->5 s == 0.13 light / 0.20 firm)
    self.assertLess(self._duty_elesys(60), 0.09)
    self.assertLess(self._duty_elesys(200), 0.12)

  def test_jitter_does_not_retrigger(self):
    # +-2 count jitter must not add duty beyond the periodic refresh
    self.assertAlmostEqual(self._duty_elesys(100, jitter=2), self._duty_elesys(100, jitter=0), delta=0.03)

  def test_rise_triggers_and_holds(self):
    # a genuine ramp (+3/frame) keeps the pump on continuously
    anchor, last = 0, -99.0
    pumps = []
    for i in range(1, 60):
      p, anchor, last = brake_pump_hysteresis_elesys(3 * i, 5.0, anchor, last, i * 0.01)
      pumps.append(p)
    self.assertTrue(all(pumps[1:]))

  def test_refractory_blocks_small_rises_then_allows(self):
    # small rise right after a run: blocked by the 3.0 s refractory; allowed once idle
    # (run 0.5 + refractory 3.0 == the same 3.5 s idle threshold as v3),
    # and a big rise (+15, real apply ramp) bypasses the refractory entirely.
    # NB the rise used here is +10 at cb 50, because v5 scales the deadband with command
    # level and +6 no longer clears it down there -- see test_deadband_scales_with_command.
    anchor, last = 0, -99.0
    _, anchor, last = brake_pump_hysteresis_elesys(150, 5.0, anchor, last, 0.0)
    _, anchor, last = brake_pump_hysteresis_elesys(50, 5.0, anchor, last, 1.5)    # release re-arms anchor
    p, anchor, last = brake_pump_hysteresis_elesys(60, 5.0, anchor, last, 1.6)    # +10: refractory blocks
    self.assertFalse(p)
    p, anchor, last = brake_pump_hysteresis_elesys(60, 5.0, anchor, last, 4.0)    # idle: now allowed
    self.assertTrue(p)

  def test_deadband_scales_with_command(self):
    # v5. `rise and in_run` re-primes inside the current run, so a command creeping up by the
    # deadband every 0.5 s pins the pump on. With a flat deadband of 3 that made light braking
    # 48.2% of all moving run time (405.9 min engaged, routes 15646e8515eda1a7). Requiring a
    # bigger rise at low command breaks that, and leaves firm braking exactly as it was.
    def rises(anchor_cb, delta):
      """does a rise of `delta` from `anchor_cb` re-prime once idle?"""
      anchor, last = 0, -99.0
      _, anchor, last = brake_pump_hysteresis_elesys(anchor_cb, 5.0, anchor, last, 0.0)
      p, _, _ = brake_pump_hysteresis_elesys(anchor_cb + delta, 5.0, anchor, last, 10.0)
      return p

    # light braking: small drift must NOT re-prime, a real rise still must
    self.assertFalse(rises(30, 5))
    self.assertFalse(rises(30, 8))
    self.assertTrue(rises(30, 14))
    # firm braking: unchanged from v4, +4 is still enough
    self.assertTrue(rises(210, 4))
    self.assertTrue(rises(160, 6))
    # and the deadband is monotonically decreasing in command level
    from opendbc.car.honda.carcontroller import ELESYS_PUMP_DEADBAND_BP, ELESYS_PUMP_DEADBAND_V
    self.assertEqual(ELESYS_PUMP_DEADBAND_V, sorted(ELESYS_PUMP_DEADBAND_V, reverse=True))
    self.assertEqual(len(ELESYS_PUMP_DEADBAND_BP), len(ELESYS_PUMP_DEADBAND_V))

  def test_apply_ramp_into_a_stop_still_pins_the_pump(self):
    # the whole risk of a bigger low-command deadband is delaying pressure build. A genuine
    # apply ramp climbs far faster than the deadband, so it must still run continuously.
    anchor, last, on = 0, -99.0, 0
    for i in range(150):
      p, anchor, last = brake_pump_hysteresis_elesys(40 + i, 2.2 - i * 0.012, anchor, last, i * 0.01)
      on += p
    self.assertGreater(on / 150, 0.95)
    anchor, last = 0, -99.0
    _, anchor, last = brake_pump_hysteresis_elesys(150, 5.0, anchor, last, 0.0)
    p, anchor, last = brake_pump_hysteresis_elesys(170, 5.0, anchor, last, 1.6)   # +20: bypasses refractory
    self.assertTrue(p)

  def test_final_approach_at_firm_command_is_continuous(self):
    # REGRESSION for the v4 experiment that deleted this branch. Firm braking through the stop
    # approach (0.15 <= v < 2.5, cb > 100) must pin the pump on -- one smooth run that fades into
    # the hold. v4 let the ordinary rise/backstop rules govern instead, on the argument that the
    # branch was rare; but the frames it covers are the maximum-demand ones, and once the command
    # is firm the rise-trigger is dead. Measured v3 -> v4 on the real routes: stop-approach duty
    # 0.72 -> 0.26 and worst pump-off gap 7.00 s -> 10.22 s. Restored.
    anchor, last, starts, on, prev = 0, -99.0, 0, 0, False
    for i in range(500):  # 5 s at v ramping 2.4 -> 0.2
      v = 2.4 - i * 0.0044
      p, anchor, last = brake_pump_hysteresis_elesys(150, v, anchor, last, i * 0.01)
      starts += (p and not prev)
      on += p
      prev = p
    self.assertGreater(on / 500, 0.95)
    self.assertEqual(starts, 1)

  def test_light_command_in_the_approach_is_not_continuous(self):
    # the branch is deliberately gated on cb > 100, so a LIGHT command at walking pace -- the
    # audible case that motivated the v4 experiment -- still falls through to the ordinary rules
    anchor, last, on = 0, -99.0, 0
    for i in range(500):
      v = 2.4 - i * 0.0044
      p, anchor, last = brake_pump_hysteresis_elesys(60, v, anchor, last, i * 0.01)
      on += p
    self.assertLess(on / 500, 0.35)

  def test_final_approach_still_primes_on_a_real_ramp(self):
    # the saving must not come out of pressure build: a genuine apply ramp into a stop still
    # keeps the pump running, because each +3 rise re-primes inside the current run
    anchor, last, on = 0, -99.0, 0
    for i in range(150):  # 1.5 s ramping cb 60 -> 210 while slowing through the band
      p, anchor, last = brake_pump_hysteresis_elesys(60 + i, 2.2 - i * 0.012, anchor, last, i * 0.01)
      on += p
    self.assertGreater(on / 150, 0.95)

  def test_standstill_hold_is_quiet(self):
    # stopped with brake held: rare top-ups only (~1 s per 30 s), not a burp every 2.5 s.
    # 487 s of logged holds showed zero effective creep across every pump-off gap.
    duty = self._duty_elesys(250, seconds=60.0, v_ego=0.0)
    self.assertLess(duty, 0.06)

  def test_motion_reprimes_after_long_hold(self):
    # if pressure ever decays enough that the car starts creeping, the standstill period no
    # longer applies and the expired timer must re-prime immediately
    anchor, last = 0, -99.0
    _, anchor, last = brake_pump_hysteresis_elesys(250, 0.0, anchor, last, 0.0)    # prime on entry
    p, anchor, last = brake_pump_hysteresis_elesys(250, 0.0, anchor, last, 10.0)   # deep in the hold: quiet
    self.assertFalse(p)
    p, anchor, last = brake_pump_hysteresis_elesys(250, 0.3, anchor, last, 10.1)   # car creeps -> re-prime
    self.assertTrue(p)

  def test_saturated_moving_braking_pumps_continuously(self):
    # REGRESSION for commit 2905e73d1 "Fixed Pump Blind Spot on Saturated Braking", which the v4
    # experiment silently reverted. At a railed command the rise-trigger has nothing left to rise
    # to, so without this branch only the 6 s backstop runs. Measured v3 -> v4 on the real
    # 13-route command stream at cb >= 200: duty 1.00 -> 0.32, worst pump-off gap 0.16 s -> 5.50 s.
    # That is the exact ~3 s-gap signature that bled 0.5-0.7 m/s^2 and caused the end-of-stop bite.
    duty = self._duty_elesys(253, seconds=30.0, v_ego=15.0)
    self.assertGreater(duty, 0.95)

  def test_saturation_threshold_boundary(self):
    # the branch is cb > 200 exactly: at 200 the ordinary rules still govern, so the deletion
    # cannot be reintroduced by accident via an off-by-one
    self.assertGreater(self._duty_elesys(201, seconds=30.0, v_ego=15.0), 0.95)
    self.assertLess(self._duty_elesys(200, seconds=30.0, v_ego=15.0), 0.5)

  def test_backstop_is_load_scaled_not_flat(self):
    # firm braking must refresh sooner than light braking. This is not cosmetic: the backstop is
    # what re-primes a hold the instant the car creeps (see test_motion_reprimes_after_long_hold),
    # so a flat period would lengthen that recovery for a firm hold.
    self.assertGreater(self._duty_elesys(253, v_ego=15.0), self._duty_elesys(60, v_ego=15.0))

  def test_saturated_standstill_stays_quiet(self):
    # firm command at v=0 must still keep the 30 s top-up period, not the moving backstop
    self.assertLess(self._duty_elesys(253, seconds=60.0, v_ego=0.0), 0.04)

  def test_release_from_firm_braking_stops_pumping(self):
    # leaving firm braking: the current run tails out (<= 0.5 s), then the refractory and
    # backstop govern again -- the pump must not stay latched on
    anchor, last = 0, -99.0
    for i in range(200):  # 2 s firm at speed
      _, anchor, last = brake_pump_hysteresis_elesys(253, 15.0, anchor, last, i * 0.01)
    p, anchor, last = brake_pump_hysteresis_elesys(60, 15.0, anchor, last, 3.2)  # released, 1.2 s later
    self.assertFalse(p)

  def test_run_length_is_half_of_v3(self):
    # pin the headline constant: a single isolated prime runs 0.5 s, not 1.0 s
    anchor, last = 0, -99.0
    _, anchor, last = brake_pump_hysteresis_elesys(150, 15.0, anchor, last, 0.0)
    p, _, _ = brake_pump_hysteresis_elesys(150, 15.0, anchor, last, 0.45)
    self.assertTrue(p)
    p, _, _ = brake_pump_hysteresis_elesys(150, 15.0, anchor, last, 0.55)
    self.assertFalse(p)

  def test_no_pump_without_brake(self):
    p, _, _ = brake_pump_hysteresis_elesys(0, 0.0, 0, -99.0, 5.0)
    self.assertFalse(p)
    p2, _ = brake_pump_hysteresis(0, 0, 0.0, 5.0)
    self.assertFalse(p2)

  def test_upstream_default_unchanged(self):
    # upstream: steady state barely pumps (0.2 s per 20 s), rising always pumps
    last, on = 0.0, 0
    for i in range(2000):
      pump, last = brake_pump_hysteresis(100, 100, last, i * 0.01)
      on += pump
    self.assertLess(on / 2000, 0.06)
    last = 0.0
    for i in range(1, 100):
      pump, last = brake_pump_hysteresis(i + 1, i, last, i * 0.01)
      self.assertTrue(pump)

class _C1b:
  """Pump rule C1b driven the way CarController drives it: one call per 0x1FA frame (every other 100 Hz frame), with
  ts = frame * 0.01 exactly as the controller computes it."""
  DT = 0.02

  def __init__(self):
    self.level, self.trig, self.last, self.frame = 0, 0, -1e9, 0

  def step(self, cb, v=10.0):
    ts = self.frame * 0.01
    p, self.level, self.trig, self.last = brake_pump_c1b_elesys(cb, v, self.level, self.trig, self.last, ts)
    self.frame += 2
    return p

  def hold(self, cb, seconds, v=10.0):
    return [self.step(cb, v) for _ in range(round(seconds / self.DT))]

  @staticmethod
  def starts(on):
    return sum(1 for a, b in zip([False, *on], on, strict=False) if b and not a)


class TestBrakePumpC1b(unittest.TestCase):
  """FORK(HONDA_ACCORD_9G_AU): pump rule C1b (HondaFlagsSP.ELESYS_PUMP_C1B, "Quiet pump at stops"). A burst at the
  first frame of every application, one per rise past the deadband with no minimum gap, the continuous runs at
  v >= 2.5 / cb > 200 and (v5's crawl run) 0.15 <= v < 2.5 / cb > 100, a burst at least every 6 s moving at cb >= 100
  (dry bound + creep guard), and at standstill bursts only to build a hold. No 30 s top-up, no light-braking backstop."""

  def test_onset_burst_at_the_first_frame(self):
    c = _C1b()
    c.hold(0, 1.0)
    self.assertTrue(c.step(1))                # the first frame, at 1 count: C1 waited for 11
    # an application that never passes the deadband still gets its one burst, and only that
    c = _C1b()
    on = c.hold(10, 10.0)
    self.assertEqual(_C1b.starts(on), 1)
    self.assertTrue(on[0])
    self.assertEqual(sum(on), 25)

  def test_every_application_bursts_at_its_first_frame(self):
    c = _C1b()
    on = []
    for cb in (3, 8, 20, 5, 60):              # five applications, 1 s each, 1 s apart
      on += c.hold(0, 1.0)
      app = c.hold(cb, 1.0)
      self.assertTrue(app[0], f"the first frame of the {cb}-count application")
      on += app
    self.assertEqual(_C1b.starts(on), 5)
    # an application that starts while the last one's burst is still running is pumped from its first frame too
    c = _C1b()
    c.hold(40, 0.2)
    self.assertFalse(c.step(0))
    self.assertTrue(c.step(30))

  def test_onset_is_one_half_second_burst_and_then_quiet(self):
    c = _C1b()
    on = c.hold(40, 20.0)
    self.assertEqual(_C1b.starts(on), 1)
    self.assertEqual(sum(on), 25)                        # 0.5 s at 50 Hz
    self.assertTrue(all(on[:25]) and not any(on[25:]))   # no light-braking backstop: 19.5 s dry at cb 40

  def test_rises_versus_the_deadband(self):
    for level, small, big_enough in ((50, 5, 7), (150, 3, 5), (190, 2, 4)):
      c = _C1b()
      c.hold(level, 3.0)
      self.assertFalse(any(c.hold(level + small, 2.0)), msg=f"+{small} at {level} is inside the deadband")
      c = _C1b()
      c.hold(level, 3.0)
      self.assertTrue(any(c.hold(level + big_enough, 0.1)), msg=f"+{big_enough} at {level} is a rise")

  def test_deadband_is_measured_from_the_delivered_level(self):
    # two +5 steps 2 s apart at cb 50: each is inside the deadband of ~6.5, but the second takes the command 10 past
    # what the last burst delivered, so it fires - drift cannot creep away undelivered
    c = _C1b()
    c.hold(50, 3.0)
    self.assertFalse(any(c.hold(55, 2.0)))
    self.assertTrue(any(c.hold(60, 2.0)))

  def test_rises_need_no_gap(self):
    # C1 blocked a rise for 1 s after a burst unless it was over 15 counts; C1b has no minimum gap
    c = _C1b()
    c.hold(60, 0.6)                           # burst 0-0.5 s
    self.assertTrue(c.step(70))               # +10 at 0.6 s: fires at once (C1: blocked until 1.5 s)
    # the deadband alone gates re-triggers: +5 at cb 60 never fires however long after the burst
    c = _C1b()
    c.hold(60, 0.6)
    self.assertFalse(any(c.hold(65, 5.0)))
    # a staircase of +8 every 0.6 s: every step is its own burst, each the moment it arrives
    c = _C1b()
    c.hold(60, 0.6)
    for cb in range(68, 100, 8):
      on = c.hold(cb, 0.6)
      self.assertTrue(on[0], f"step to {cb}")
      self.assertEqual(_C1b.starts(on), 1)

  def test_running_burst_extends_while_the_command_climbs(self):
    # an apply ramp is one smooth run: +3 counts a frame from 20 to 98 (under the crawl and firm bands either way)
    c = _C1b()
    ramp = [c.step(cb) for cb in range(20, 99, 3)]
    on = ramp + c.hold(98, 2.0)
    self.assertEqual(_C1b.starts(on), 1)
    self.assertTrue(all(ramp))
    # a climb slower than EXT (= max(2, deadband / 2)) does not extend: +2 every 0.3 s at cb 30 (EXT 4.5)
    c = _C1b()
    on = c.hold(30, 0.3) + c.hold(32, 0.3) + c.hold(34, 0.3) + c.hold(34, 2.0)
    self.assertEqual(sum(on), 25)

  def test_steady_moving_hold_needs_no_pump_below_100(self):
    c = _C1b()
    self.assertEqual(_C1b.starts(c.hold(90, 60.0)), 1)

  def test_moving_at_100_or_more_bursts_every_six_seconds(self):
    c = _C1b()
    on = c.hold(120, 30.0)
    self.assertEqual(_C1b.starts(on), 5)       # 0, 6, 12, 18, 24 s
    self.assertEqual(sum(on), 5 * 25)

  def test_firm_moving_braking_is_continuous(self):
    c = _C1b()
    self.assertTrue(all(c.hold(201, 30.0, v=15.0)))
    self.assertTrue(all(c.hold(253, 30.0, v=2.5)))
    c = _C1b()
    on = c.hold(200, 30.0, v=15.0)            # cb > 200 exactly: at 200 the burst rules govern
    self.assertLess(sum(on) / len(on), 0.15)

  def test_crawl_run_above_100_counts_below_2_5_m_s(self):
    # v5's crawl continuous run, restored: 0.15 <= v < 2.5 m/s and cb > 100 keeps the pump on, every frame
    c = _C1b()
    on = [c.step(150, v=2.4 - i * 0.0088) for i in range(250)]   # 5 s, 2.4 -> 0.2 m/s: the final approach
    self.assertTrue(all(on))
    for cb, v in ((101, 0.15), (125, 1.0), (253, 2.0), (101, 2.49)):
      c = _C1b()
      self.assertTrue(all(c.hold(cb, 10.0, v=v)), msg=f"cb {cb} at {v} m/s")
    # its edges: 100 counts is not above 100, 0.14 m/s is standstill, 2.5 m/s is the firm band's (cb > 200 there)
    for cb, v in ((100, 1.0), (150, 0.14), (150, 2.5)):
      c = _C1b()
      self.assertLess(sum(c.hold(cb, 30.0, v=v)) / 1500, 0.15, msg=f"cb {cb} at {v} m/s")

  def test_crawl_approach_into_a_firm_hold_then_silence(self):
    # the stop reached in the crawl run at 150 counts: the run tails out (<= 0.5 s) at standstill, and the built hold
    # is never topped up - no 30 s top-up, nothing for two minutes
    c = _C1b()
    c.hold(150, 3.0, v=1.0)
    on = c.hold(150, 120.0, v=0.0)
    self.assertLessEqual(sum(on), 25)
    self.assertFalse(any(on[25:]))
    self.assertEqual(c.level, 150)

  def test_standstill_hold_reached_firm_never_tops_up(self):
    # stop reached at the hold (V5's stops: ~185 counts when the wheels stop, hold 189): the hold is already there
    # (level >= 100) and the last 4 counts are inside BIG_RISE, so nothing pumps while stopped - no 30 s top-up
    c = _C1b()
    c.hold(185, 2.0, v=3.0)
    on = c.hold(189, 120.0, v=0.0)
    self.assertFalse(any(on))
    self.assertEqual(c.level, 185)

  def test_the_soft_stops_rise_to_the_hold_is_delivered_once(self):
    # approach at cb 150 (pumped while moving), the soft stop's cap of 125 while rolling (now inside the crawl run),
    # then its 125 -> 189 rise at standstill (250 counts/s = 5 per 0x1FA frame): delivered in one run - the crawl run's
    # tail extends through the climb - and a steady hold never re-pumps
    c = _C1b()
    c.hold(150, 2.0, v=3.0)
    self.assertTrue(all(c.hold(125, 1.0, v=0.5)))
    on = [c.step(cb, v=0.0) for cb in range(125, 190, 5)] + c.hold(189, 120.0, v=0.0)
    self.assertEqual(_C1b.starts(on), 1)
    self.assertTrue(all(on[:13]), "the whole rise is pumped")
    self.assertGreaterEqual(c.level, 189)
    self.assertFalse(any(on[-5000:]), "no top-up after it")
    # the same rise with the crawl run long over (the wheels stopped 2 s before): one hold-rise burst, as in C1
    c = _C1b()
    c.hold(150, 2.0, v=3.0)
    c.hold(125, 1.0, v=0.5)
    c.hold(125, 2.0, v=0.0)
    self.assertEqual(c.level, 125)
    on = [c.step(cb, v=0.0) for cb in range(125, 190, 5)] + c.hold(189, 120.0, v=0.0)
    self.assertEqual(_C1b.starts(on), 1)
    self.assertGreaterEqual(c.level, 189)

  def test_standstill_hold_reached_light_gets_one_hold_build_burst(self):
    # stop reached at cb 40: the 40 -> 189 rise at standstill builds the hold - one burst, extended through the
    # ramp, then nothing for two minutes
    c = _C1b()
    c.hold(40, 3.0, v=1.0)
    on = c.hold(125, 0.1, v=0.0) + [c.step(cb, v=0.0) for cb in range(127, 190, 2)] + c.hold(189, 120.0, v=0.0)
    self.assertEqual(_C1b.starts(on), 1)
    self.assertGreaterEqual(c.level, 189)
    # and once the delivered level is 100 or more a standstill rise fires only past BIG_RISE (15)
    c = _C1b()
    self.assertTrue(c.step(100, v=0.0))      # an application that starts at standstill: its first-frame burst
    c.hold(100, 1.0, v=0.0)
    self.assertFalse(any(c.hold(115, 5.0, v=0.0)))
    self.assertTrue(any(c.hold(116, 0.1, v=0.0)))

  def test_a_hold_that_rolls_is_pumped_at_once(self):
    # above 100 counts a hold that starts to roll is in the crawl run: pumped from the first moving frame, even
    # within 6 s of the last burst (C1 waited for its 6 s creep guard)
    c = _C1b()
    c.hold(189, 1.0, v=0.0)
    self.assertFalse(any(c.hold(189, 4.0, v=0.0)))
    self.assertTrue(all(c.hold(189, 1.0, v=0.3)))

  def test_creep_guard(self):
    # at exactly 100 counts (below the crawl run) the creep guard is what catches the roll: a burst at once when the
    # last one was 6 s or more ago, none within 6 s
    c = _C1b()
    c.hold(100, 1.0, v=0.0)
    self.assertFalse(any(c.hold(100, 20.0, v=0.0)))
    self.assertTrue(c.step(100, v=0.3))
    c = _C1b()
    c.hold(100, 4.0, v=0.0)
    self.assertFalse(c.step(100, v=0.3))

  def test_release_needs_no_pump_and_rearms(self):
    c = _C1b()
    c.hold(253, 2.0, v=15.0)
    c.hold(60, 0.5, v=15.0)                          # the last firm frame's run tails out (<= 0.5 s), as in v5
    self.assertFalse(any(c.hold(60, 3.0, v=15.0)))   # release to 60: no pump, the level follows it down
    self.assertEqual(c.level, 60)
    self.assertTrue(any(c.hold(67, 0.1, v=15.0)))    # a rise from the released level fires
    # a release inside the jitter band (6 counts) keeps the level
    c = _C1b()
    c.hold(80, 2.0)
    c.hold(75, 1.0)
    self.assertEqual(c.level, 80)

  def test_release_to_zero_resets_and_the_next_application_bursts_at_once(self):
    c = _C1b()
    c.hold(60, 0.6)
    c.step(0)
    self.assertEqual(c.level, 0)
    self.assertTrue(c.step(2))                # first frame of an application: a burst, however small

  def test_no_pump_without_brake(self):
    p, level, _, _ = brake_pump_c1b_elesys(0, 0.0, 150, 150, 0.0, 0.1)   # even inside a running burst
    self.assertFalse(p)
    self.assertEqual(level, 0)
    p, _, _, _ = brake_pump_c1b_elesys(0, 15.0, 0, 0, -1e9, 5.0)
    self.assertFalse(p)
    p, _, _, _ = brake_pump_c1b_elesys(0, 1.0, 0, 0, -1e9, 5.0)          # not in the crawl run either
    self.assertFalse(p)

  def test_non_finite_speed_counts_as_moving_outside_both_runs(self):
    # as in v5: a NaN vEgo is neither still, crawling nor firm, so a hold still gets its 6 s bound
    c = _C1b()
    on = c.hold(150, 13.0, v=float("nan"))
    self.assertEqual(_C1b.starts(on), 3)

  def test_matches_the_c1weak_reference(self):
    # c1weak's replay rule (final/replay2.py c1x(onset_first=True, min_gap=0.0, crawl=True)), transcribed: the rule
    # its replay table was computed with. A random command/speed trace must give the same pump bit on every frame.
    def ref(cbs, vs, ts):
      out = []
      level = trig = 0.
      last = -1e9
      for ab, ve, t in zip(cbs, vs, ts, strict=True):
        if ab <= 0:
          level = 0.
          out.append(False)
          continue
        if (ve >= 2.5 and ab > 200) or (0.15 <= ve < 2.5 and ab > 100):
          level, trig, last = max(level, ab), ab, t
          out.append(True)
          continue
        db = float(np.interp(ab, [0., 60., 200.], [12., 6., 3.]))
        if t - last < 0.5:
          if ab >= trig + max(2, 0.5 * db):
            last, trig = t, ab
        elif level == 0:
          last, trig = t, ab
        elif (ve >= 0.15 or level < 100 or ab > level + 15) and ab > level + db:
          last, trig = t, ab
        elif ve >= 0.15 and ab >= 100 and t - last >= 6.0:
          last, trig = t, ab
        if ab < level - 6:
          level = ab
        on = t - last < 0.5
        if on:
          level = max(level, ab)
        out.append(on)
      return out
    rng = np.random.default_rng(7)
    n = 30000
    cb = np.zeros(n)
    v = np.zeros(n)
    x, sp = 0.0, 10.0
    for i in range(n):
      if rng.random() < 0.01:
        x = 0.0 if rng.random() < 0.3 else float(rng.choice([rng.uniform(1, 40), rng.uniform(40, 260)]))
      x = max(0.0, x + rng.normal(0, 1.5))
      sp = float(np.clip(sp + rng.normal(0, 0.15) - 0.002 * x, 0.0, 30.0))
      if rng.random() < 0.002:
        sp = float(rng.choice([0.0, 0.1, 1.0, 2.4, 2.5, 12.0]))
      cb[i], v[i] = round(x), sp
    ts = [2 * i * 0.01 for i in range(n)]   # as _C1b computes it, bit for bit
    c = _C1b()
    got = [c.step(int(a), float(b)) for a, b in zip(cb, v, strict=True)]
    self.assertEqual(got, ref(cb, v, ts))
    self.assertGreater(sum(got), 1000)


class TestElesysPumpRuleSelection(unittest.TestCase):
  """FORK(HONDA_ACCORD_9G_AU): CarController picks the pump rule from CP_SP.flags alone, and with the flag clear
  0x1FA is exactly what it was."""

  def _run(self, flags, seconds=40.0):
    from opendbc.can import CANPacker
    from opendbc.car import gen_empty_fingerprint, structs
    from opendbc.car.honda.carcontroller import CarController
    from opendbc.car.honda.interface import CarInterface
    fp = gen_empty_fingerprint()
    fp[0][0x188] = 8   # GEARBOX_AUTO
    fp[0][0x201] = 6   # the comma pedal
    CP = CarInterface.get_params(ELESYS_CAR, fp, [], False, False, False)
    CP_SP = CarInterface.get_params_sp(CP, ELESYS_CAR, fp, [], False, False, False)
    self.assertTrue(CP.openpilotLongitudinalControl)
    CP_SP.flags |= flags
    cc_obj = CarController(DBC[ELESYS_CAR], CP, CP_SP)
    packer = CANPacker(DBC[ELESYS_CAR][Bus.pt])
    off = _as_tuple(packer.make_can_msg("BRAKE_COMMAND", 0, {"BRAKE_PUMP_REQUEST": 0}))[1]
    on = _as_tuple(packer.make_can_msg("BRAKE_COMMAND", 0, {"BRAKE_PUMP_REQUEST": 1}))[1]
    pump_mask = bytes(a ^ b for a, b in zip(off[:7], on[:7], strict=True))
    cs = _FakeCS()
    out = []
    for i in range(round(seconds * 100)):
      t = i * 0.01
      cc = structs.CarControl.new_message()
      cc.enabled = cc.longActive = True
      # 3 s of braking into a stop, then a standstill hold
      cs.out.vEgo = max(0.0, 6.0 - 2.0 * t)
      cs.out.standstill = cs.out.vEgo == 0.0
      cc.actuators.accel = -1.5 if t < 3.0 else -2.0
      _, sends = cc_obj.update(cc.as_reader(), structs_CC_SP(), cs, int(t * 1e9))
      f = [d for a, d, _ in map(_as_tuple, sends) if a == 0x1FA]
      if f:
        out.append((f[0], cc_obj.apply_brake_last, float(cs.out.vEgo), i * 0.01,
                    any(x & m for x, m in zip(f[0][:7], pump_mask, strict=True))))
    return out

  def test_flag_clear_is_v5_and_flag_set_is_c1b(self):
    from opendbc.sunnypilot.car.honda.values_ext import HondaFlagsSP
    v5 = self._run(0)
    c1 = self._run(HondaFlagsSP.ELESYS_PUMP_C1B.value)
    self.assertEqual(len(v5), len(c1))
    self.assertTrue(any(cb > 0 for _, cb, _, _, _ in v5))
    anchor, last = 0, 0.0
    level, trig, last1 = 0, 0, 0.0
    for (f5, cb5, v, ts, p5), (f1, cb1, _, _, p1) in zip(v5, c1, strict=True):
      self.assertEqual(cb5, cb1)              # the pump rule changes only the pump bit
      exp5, anchor, last = brake_pump_hysteresis_elesys(cb5, v, anchor, last, ts)
      exp1, level, trig, last1 = brake_pump_c1b_elesys(cb1, v, level, trig, last1, ts)
      self.assertEqual(p5, exp5, msg=f"v5 at {ts:.2f} s")
      self.assertEqual(p1, exp1, msg=f"C1b at {ts:.2f} s")
      if p5 == p1:
        self.assertEqual(f5, f1)
    # the scenario tells the rules apart: v5 tops a standstill hold up at 30 s, C1b never re-pumps a built hold
    self.assertTrue(any(p for _, _, v, ts, p in v5 if v == 0.0 and ts > 20.0))
    self.assertFalse(any(p for _, _, v, ts, p in c1 if v == 0.0 and ts > 20.0))

  def test_the_retired_c1_flag_runs_v5(self):
    # flag 16 (the retired rule C1) selects nothing: an old CarParamsSP that carries it sends v5's 0x1FA, frame for frame
    from opendbc.sunnypilot.car.honda.values_ext import HondaFlagsSP
    self.assertEqual(self._run(HondaFlagsSP.ELESYS_PUMP_V6.value), self._run(0))

  def test_brake_law_flag_alone_leaves_the_pump_on_v5(self):
    # flag 32 belongs to the brake law; on its own it must not select C1b
    from opendbc.car.honda.carcontroller import CarController
    from opendbc.car.honda.interface import CarInterface
    from opendbc.sunnypilot.car.honda.values_ext import HondaFlagsSP
    self.assertEqual(HondaFlagsSP.ELESYS_PUMP_C1B, 64)
    self.assertEqual(HondaFlagsSP.ELESYS_BRAKE_LAW_V2, 32)
    CP = CarInterface.get_non_essential_params(ELESYS_CAR)
    CP_SP = CarInterface.get_non_essential_params_sp(CP, ELESYS_CAR)
    CP_SP.flags |= HondaFlagsSP.ELESYS_BRAKE_LAW_V2.value | HondaFlagsSP.ELESYS_PUMP_V6.value
    self.assertFalse(CarController(DBC[ELESYS_CAR], CP, CP_SP).elesys_pump_c1b)
    CP_SP.flags |= HondaFlagsSP.ELESYS_PUMP_C1B.value
    self.assertTrue(CarController(DBC[ELESYS_CAR], CP, CP_SP).elesys_pump_c1b)


class TestElesysC1bWithTheSoftStop(unittest.TestCase):
  """FORK(HONDA_ACCORD_9G_AU): the real CarController with the tuner on, so the soft stop is built, and C1b selected.
  The soft stop caps the rolling command at ~125 counts and raises it to the hold after the wheels read zero; C1b must
  pump the capped approach (the crawl run, which the cap was sized for), deliver the rise to the hold, and not top the
  hold up afterwards. The pump is read where the car reads it: BRAKE_PUMP_REQUEST in the 0x1FA the controller sends."""

  def _run(self, flags):
    from unittest import mock
    from opendbc.can import CANPacker
    from opendbc.car import gen_empty_fingerprint, structs
    from opendbc.car.honda.carcontroller import CarController
    from opendbc.car.honda.interface import CarInterface
    from opendbc.sunnypilot.car.honda import dynamic_tuning as dt
    from opendbc.sunnypilot.car.honda import elesys_gas as eg

    class _P:
      store = {"HondaDynamicTuningEnabled": True, eg.GAS_LAW_PARAM: True}

      def get(self, key, block=False, return_default=False):
        return self.store.get(key, dt._PARAM_SPEC[key][0] if key in dt._PARAM_SPEC else None)

      def get_bool(self, key, block=False):
        return bool(self.store.get(key, False))

      def put(self, key, val, block=False):
        pass
    fp = gen_empty_fingerprint()
    fp[0][0x188] = 8
    fp[0][0x201] = 6
    with mock.patch.object(dt, "_open_params", lambda: _P()), mock.patch.object(eg, "_open_params", lambda: _P()):
      CP = CarInterface.get_params(ELESYS_CAR, fp, [], False, False, False)
      CP_SP = CarInterface.get_params_sp(CP, ELESYS_CAR, fp, [], False, False, False)
      CP_SP.flags |= flags
      cc_obj = CarController(DBC[ELESYS_CAR], CP, CP_SP)
    self.assertIsNotNone(cc_obj.soft_stop, "the tuner on builds the soft stop")
    packer = CANPacker(DBC[ELESYS_CAR][Bus.pt])
    off = _as_tuple(packer.make_can_msg("BRAKE_COMMAND", 0, {"BRAKE_PUMP_REQUEST": 0}))[1]
    on = _as_tuple(packer.make_can_msg("BRAKE_COMMAND", 0, {"BRAKE_PUMP_REQUEST": 1}))[1]
    pump_mask = bytes(a ^ b for a, b in zip(off[:7], on[:7], strict=True))
    cs = _FakeCS()
    out = []
    v, stopping = 6.0, False
    for i in range(round(40.0 * 100)):
      t = i * 0.01
      stopping = stopping or v <= 0.9              # stopping entered below the soft stop's 1.2 m/s, as measured
      v = max(0.0, v - (1.0 if not stopping else 0.6) * 0.01)
      cs.out.vEgo = v
      cs.out.vEgoRaw = v if v > 0.3 else 0.0          # XMISSION_SPEED reads 0 below ~0.3 m/s
      cs.out.standstill = cs.out.vEgoRaw == 0.0
      cs.out.aEgo = -0.6 if v > 0 else 0.0
      cc = structs.CarControl.new_message()
      cc.enabled = cc.longActive = True
      cc.orientationNED = [0.0, 0.0, 0.0]
      # the planner eases off into the stop (the creep table adds ~0.75 m/s^2 at 0.9 m/s: ~90 counts at entry, under
      # the 125 cap), then longcontrol's stopping accel, which the cap holds at 125 until the wheels read zero
      cc.actuators.accel = (-1.0 if v > 3.0 else -0.2) if not stopping else -0.8
      cc.actuators.longControlState = structs.CarControl.Actuators.LongControlState.stopping if stopping else \
        structs.CarControl.Actuators.LongControlState.pid
      _, sends = cc_obj.update(cc.as_reader(), structs_CC_SP(), cs, int(t * 1e9))
      f = [d for a, d, _ in map(_as_tuple, sends) if a == 0x1FA]
      if f:
        pump = any(x & m for x, m in zip(f[0][:7], pump_mask, strict=True))
        out.append((t, v, cs.out.standstill, cc_obj.apply_brake_last, cc_obj.pump_level, pump))
    return cc_obj, out

  def test_c1b_delivers_the_soft_stops_hold(self):
    from opendbc.sunnypilot.car.honda.values_ext import HondaFlagsSP
    cc_obj, out = self._run(HondaFlagsSP.ELESYS_PUMP_C1B.value)
    self.assertGreater(len(out), 1900, "a 0x1FA every other frame")
    t_stop = next(t for t, v, *_ in out if v <= 0.9)
    rolling = [cb for t, v, _, cb, *_ in out if t >= t_stop and v > 0.0]
    self.assertTrue(rolling and max(rolling) <= 130, f"the soft stop caps the rolling command: {max(rolling or [0])}")
    held = [cb for t, v, ss, cb, *_ in out if ss and t > 10.0]
    self.assertGreaterEqual(min(held), 180, "the hold the soft stop rises to")
    level_end = out[-1][4]
    self.assertGreaterEqual(level_end, min(held), "C1b delivered the hold (C1's pseudo-code alone left it at the cap)")
    # one burst at standstill, delivering the rise, none for the rest of the 30 s hold
    t_still = next(t for t, v, ss, *_ in out if ss)
    levels_after = [lv for t, _, ss, _, lv, _ in out if ss and t > t_still + 3.0]
    self.assertEqual(min(levels_after), max(levels_after), "no top-up: the delivered level never moves again")
    self.assertFalse(any(p for t, _, ss, _, _, p in out if ss and t > t_still + 3.0), "the 0x1FA pump bit stays off")

  def test_c1b_pumps_the_capped_approach_through_the_crawl_run(self):
    # what the soft stop's 125 cap was sized for: every rolling 0x1FA at 0.15 <= v < 2.5 with the command above 100
    # carries the pump bit - the crawl run C1b restores (C1 had it on for 39% of that zone on route 120's stop)
    from opendbc.sunnypilot.car.honda.values_ext import HondaFlagsSP
    _, out = self._run(HondaFlagsSP.ELESYS_PUMP_C1B.value)
    crawl = [(t, cb, p) for t, v, _, cb, _, p in out if 0.15 <= v < 2.5 and cb > 100]
    self.assertGreater(len(crawl), 50, "the capped approach spends time in the crawl zone above 100 counts")
    self.assertTrue(all(p for _, _, p in crawl), [x for x in crawl if not x[2]][:5])
    # v5 has the same run; with the flag clear the same approach pumps it too
    _, out5 = self._run(0)
    crawl5 = [p for t, v, _, cb, _, p in out5 if 0.15 <= v < 2.5 and cb > 100]
    self.assertTrue(crawl5 and all(crawl5))


class TestElesysGasMultiplier(unittest.TestCase):
  """Golden pedal-multiplier curve -- deliberately duplicated so any change to the deployed
  curve fails a test until it has been re-measured.

  Re-measured 2026-08 against the full 14-route set. Settled-frame plant identification
  (target steady >= 1.5 s, demand = aEgo + g*sin(pitch) + aero(v) regressed on the interceptor
  command, ~200k frames over three routes on two tunes) put the command actually needed at
  1.38x @ 3-6 m/s, 1.30-1.47x @ 6-10, 1.59-1.70x @ 10-15, 1.51x @ 15-20 and 1.50-1.75x @ 20+,
  with actuatorsOutput.gas never reaching 0.9 on any of the 13 engaged routes -- so nothing
  physical was capping it. The top three breakpoints were raised ~1.25x, one measured step
  rather than the full ratio, because the PI and the pitch feedforward close part of the rest.
  Previous golden was [0.55, 0.85, 1.10, 1.25, 1.55, 2.20].

  2026-10: this is now the v1 law -- what runs with HondaElesysGasLawV2 off -- and inside v2 it
  only supplies the 0-3 m/s gain and the offset below ~16.9 m/s. v2's slope comes from the
  measured k table (elesys_gas.py, tested in test_elesys_gas.py), which is NOT monotonic; the
  monotonic and mid-band tests below are about this curve only."""
  GOLD_MULT_BP = [0., 3., 6., 10., 15., 20.]
  GOLD_MULT_V = [0.55, 0.85, 1.20, 1.55, 1.95, 2.75]

  def test_deployed_curve_matches_golden(self):
    for v in list(self.GOLD_MULT_BP) + [1.5, 4.5, 8.0, 12.5, 17.5, 25.0]:
      expect = float(np.interp(v, self.GOLD_MULT_BP, self.GOLD_MULT_V))
      self.assertAlmostEqual(elesys_gas_multiplier(float(v)), expect, places=6, msg=f'v={v}')

  def test_monotonic_increasing(self):
    # gain falloff with speed means the multiplier must only ever grow
    vals = [elesys_gas_multiplier(float(v)) for v in np.arange(0.0, 25.0, 0.25)]
    for a, b in zip(vals[:-1], vals[1:], strict=True):
      self.assertGreaterEqual(b, a - 1e-9)

  def test_mid_band_above_old_curve(self):
    # the 3-14 m/s band delivered only ~50-65% of commanded accel on the old <=1.0 ramp;
    # the fix is specifically that this band now multiplies above it
    self.assertGreaterEqual(elesys_gas_multiplier(0.0), 0.5)
    for v in (6.0, 8.0, 10.0, 12.0):
      old = float(np.interp(v, [0., 10., 15., 20.], [0.5, 1.0, 1.4, 2.1]))
      self.assertGreater(elesys_gas_multiplier(v), old, msg=f'v={v}')

  def test_v2_slope_is_deliberately_not_this_curve(self):
    # The measured pedal response (k = 6.8 m/s^2 per unit at 6 m/s against 5.65 at 3) means v2
    # asks for LESS pedal per m/s^2 at 6 m/s than at 3, and less than this curve at 6-20 m/s:
    # the car over-delivered under it. Do not "fix" v2 to be monotonic like the curve above.
    from opendbc.sunnypilot.car.honda.elesys_gas import elesys_ff_gm
    self.assertLess(elesys_ff_gm(6.0), elesys_ff_gm(3.0))
    for v in (6.0, 10.0, 15.0, 20.0):
      self.assertLess(elesys_ff_gm(v), elesys_gas_multiplier(v), msg=f'v={v}')
    for v in (0.0, 1.5, 3.0):
      self.assertEqual(elesys_ff_gm(v), elesys_gas_multiplier(v), msg=f'v={v}')


class TestBrakeCommandUnitsBit(unittest.TestCase):
  """SET_ME_1 in BRAKE_COMMAND is the cluster units flag on Elesys cars (0 = metric,
  1 = imperial) and a reserved constant 1 on every other Honda."""

  def _frame(self, car, is_metric):
    # fresh packer per frame so counters match and frames are byte-comparable
    try:
      from opendbc.can import CANPacker
      packer = CANPacker(DBC[car][Bus.pt])
    except Exception as e:  # compiled packer not available in this environment
      self.skipTest(f'opendbc.can unavailable: {e}')
    CAN = SimpleNamespace(pt=0)
    CP_SP = SimpleNamespace(flags=0)
    msg = hondacan.create_brake_command(packer, CAN, 100, True, True, False, 0, {}, CP_SP,
                                        is_metric=is_metric, elesys=car in HONDA_ELESYS)
    return bytes(msg.dat if hasattr(msg, 'dat') else msg[1])

  def _units_bit_mask(self, car):
    # locate the SET_ME_1 bit by packing it 0 vs 1 with everything else zero
    from opendbc.can import CANPacker
    a = CANPacker(DBC[car][Bus.pt]).make_can_msg("BRAKE_COMMAND", 0, {"SET_ME_1": 0})
    b = CANPacker(DBC[car][Bus.pt]).make_can_msg("BRAKE_COMMAND", 0, {"SET_ME_1": 1})
    da = bytes(a.dat if hasattr(a, 'dat') else a[1])
    db = bytes(b.dat if hasattr(b, 'dat') else b[1])
    return bytes(x ^ y for x, y in zip(da, db, strict=True))

  def test_elesys_bit_tracks_units(self):
    metric = self._frame(CAR.HONDA_ACCORD_9G_AU, True)
    imperial = self._frame(CAR.HONDA_ACCORD_9G_AU, False)
    mask = self._units_bit_mask(CAR.HONDA_ACCORD_9G_AU)
    diff = bytes(x ^ y for x, y in zip(metric, imperial, strict=True))
    # outside the checksum byte, metric vs imperial frames differ in exactly the SET_ME_1 bit
    self.assertEqual(diff[:7], mask[:7])
    self.assertEqual(sum(bin(b).count('1') for b in mask[:7]), 1)
    # metric = bit clear (matches the stock radar on the metric AU car), imperial = bit set
    idx = next(i for i, b in enumerate(mask[:7]) if b)
    self.assertEqual(metric[idx] & mask[idx], 0)
    self.assertEqual(imperial[idx] & mask[idx], mask[idx])

  def test_other_hondas_unaffected(self):
    # non-Elesys cars: constant 1 regardless of units, so frames are identical
    self.assertEqual(self._frame(CAR.HONDA_CRV, True), self._frame(CAR.HONDA_CRV, False))

class TestElesysGearDecode(unittest.TestCase):
  """GEARBOX_AUTO raw 0 means BOTH Sport and between-detents on this car, so only
  time separates them. Across 473k logged frames every 0 run was a shift transient:
  47 of them, median 3 frames (30 ms), max 520 ms, all below 0.5 m/s -- S was never
  selected during that recording. Since then S has been driven (routes b1, dd, fc), and
  GEAR read 26 on those frames (16,164) against 0 on the transients (2,975), so the
  GEAR == 26 fast path is what fires in practice; the dwell is the fallback."""

  @staticmethod
  def _name(gear):
    """capnp enums are bare ints once assigned; map back through the schema."""
    from opendbc.car import structs
    names = {int(v): k for k, v in structs.CarState.GearShifter.schema.enumerants.items()}
    raw = getattr(gear, "raw", None)
    return names.get(int(raw if raw is not None else gear))

  def _cs(self):
    from opendbc.car import gen_empty_fingerprint
    from opendbc.car.honda.carstate import CarState
    from opendbc.car.honda.interface import CarInterface
    fp = gen_empty_fingerprint()
    fp[0][0x188] = 8
    CP = CarInterface.get_params(CAR.HONDA_ACCORD_9G_AU, fp, [], False, False, False)
    CP_SP = CarInterface.get_params_sp(CP, CAR.HONDA_ACCORD_9G_AU, fp, [], False, False, False)
    self.assertEqual(str(CP.transmissionType), 'automatic')
    return CarState(CP, CP_SP)

  def test_detents_decode(self):
    cs = self._cs()
    for raw, name in ((1, 'park'), (2, 'reverse'), (4, 'neutral'), (8, 'drive')):
      self.assertEqual(self._name(cs.update_gear_elesys(raw, 0)), name, msg=f'raw={raw}')

  def test_shift_transient_holds_last_gear(self):
    # the real worst case: 52 frames (520 ms) of raw 0 between two detents
    cs = self._cs()
    cs.update_gear_elesys(8, 4)                      # in D
    for _ in range(52):
      self.assertEqual(self._name(cs.update_gear_elesys(0, 0)), 'drive')
    self.assertEqual(self._name(cs.update_gear_elesys(4, 3)), 'neutral')

  def test_sustained_zero_becomes_sport(self):
    cs = self._cs()
    cs.update_gear_elesys(8, 4)
    for _ in range(cs.SPORT_DWELL - 1):
      self.assertEqual(self._name(cs.update_gear_elesys(0, 0)), 'drive')
    self.assertEqual(self._name(cs.update_gear_elesys(0, 0)), 'sport')   # dwell reached

  def test_gear_26_is_an_instant_fast_path(self):
    cs = self._cs()
    cs.update_gear_elesys(8, 4)
    self.assertEqual(self._name(cs.update_gear_elesys(0, 26)), 'sport')  # no dwell needed

  def test_leaving_sport_is_immediate_and_rearms(self):
    cs = self._cs()
    cs.update_gear_elesys(8, 4)
    for _ in range(cs.SPORT_DWELL):
      cs.update_gear_elesys(0, 0)
    self.assertEqual(self._name(cs.gear_shifter_last), 'sport')
    self.assertEqual(self._name(cs.update_gear_elesys(8, 4)), 'drive')   # back to D at once
    self.assertEqual(cs.gear_zero_frames, 0)                      # counter re-armed
    for _ in range(10):                                           # a short blip stays D
      self.assertEqual(self._name(cs.update_gear_elesys(0, 0)), 'drive')

  def test_dwell_is_clear_of_the_longest_observed_transient(self):
    cs = self._cs()
    self.assertGreater(cs.SPORT_DWELL * 0.01, 0.52 * 1.5)

  def test_a_gearbox_message_without_gear_does_not_raise(self):
    # Fuzz seed 7944551990633456218, example 21: a fingerprint with 0x191 and no 0x188 makes
    # this car's gearbox message GEARBOX_CVT, which has no GEAR signal, and CarState.update()
    # raised KeyError('GEAR'). The real car always has 0x188, but update() must never raise --
    # a message without GEAR takes upstream's decode instead.
    from opendbc.car import gen_empty_fingerprint
    from opendbc.car.honda.interface import CarInterface
    fp = gen_empty_fingerprint()
    fp[0][0x191] = 8
    CP = CarInterface.get_params(CAR.HONDA_ACCORD_9G_AU, fp, [], False, False, False)
    CP_SP = CarInterface.get_params_sp(CP, CAR.HONDA_ACCORD_9G_AU, fp, [], False, False, False)
    self.assertEqual(str(CP.transmissionType), 'cvt')
    ci = CarInterface(CP, CP_SP)
    for i in range(5):
      ret, _ = ci.update([((i + 1) * int(1e7), [])])
    self.assertIn(self._name(ret.gearShifter), ('unknown', 'park', 'reverse', 'neutral', 'drive', 'sport', 'low'))

  def test_the_real_gearbox_still_takes_the_elesys_decode(self):
    # and the guard must not divert the real car's GEARBOX_AUTO, which does have GEAR
    from opendbc.can import CANParser
    p = CANParser(DBC[ELESYS_CAR][Bus.pt], [], 0)
    self.assertIn("GEAR", p.vl["GEARBOX_AUTO"])
    self.assertNotIn("GEAR", p.vl["GEARBOX_CVT"])


class TestElesysKeyOffSteerStatus(unittest.TestCase):
  """At key-off this EPS sends STEER_STATUS 1 (DRIVER_STEERING) in its last frames, which upstream's rule takes as both
  a temporary and a permanent fault. Across 88 logged routes (187 runs of it) STEER_STATUS 1 appeared only at key-on
  and key-off, always at a standstill, and in P on every route since the gear decode was fixed (03e on); routes 10f,
  114 and 115 showed it as "LKAS Fault: Restart the car" / TAKE CONTROL IMMEDIATELY. carstate.py ignores it ONLY
  at a standstill AND in P, ONLY on HONDA_ELESYS; anywhere else it is the fault it always was."""

  def _ci(self, car):
    from opendbc.can import CANPacker
    from opendbc.car import gen_empty_fingerprint
    from opendbc.car.honda.interface import CarInterface
    fp = gen_empty_fingerprint()
    fp[0][0x188 if car in HONDA_ELESYS else 0x1A3] = 8   # an automatic, as each car's own gearbox frame says
    CP = CarInterface.get_params(car, fp, [], False, False, False)
    CP_SP = CarInterface.get_params_sp(CP, car, fp, [], False, False, False)
    self.assertEqual(str(CP.transmissionType), 'automatic')
    return CarInterface(CP, CP_SP), CANPacker(DBC[car][Bus.pt])

  def _gear_raw(self, car, name):
    from opendbc.can import CANDefine
    dv = CANDefine(DBC[car][Bus.pt]).dv["GEARBOX_AUTO"]["GEAR_SHIFTER"]
    return next(raw for raw, n in dv.items() if n == name)

  def _faults(self, car, steer_status, gear, kph, frames=30):
    ci, packer = self._ci(car)
    gear_values = {"GEAR_SHIFTER": self._gear_raw(car, gear)}
    if car in HONDA_ELESYS:
      gear_values["GEAR"] = 0
    ret = None
    for i in range(frames):
      msgs = [packer.make_can_msg("STEER_STATUS", 0, {"STEER_STATUS": steer_status}),
              packer.make_can_msg("ENGINE_DATA", 0, {"XMISSION_SPEED": kph}),
              packer.make_can_msg("GEARBOX_AUTO", 0, gear_values)]
      ret, _ = ci.update([((i + 1) * int(1e7), [_as_tuple(m) for m in msgs])])
    return ret.standstill, TestElesysGearDecode._name(ret.gearShifter), ret.steerFaultTemporary, ret.steerFaultPermanent

  def test_parked_driver_steering_is_not_a_fault(self):
    self.assertEqual(self._faults(ELESYS_CAR, 1, "P", 0.0), (True, 'park', False, False))

  def test_driver_steering_is_still_a_fault_unless_parked(self):
    # standstill in D (a red light), and moving: both still faults, temporary and permanent as upstream has them
    self.assertEqual(self._faults(ELESYS_CAR, 1, "D", 0.0), (True, 'drive', True, True))
    standstill, gear, tmp, perm = self._faults(ELESYS_CAR, 1, "P", 20.0)
    self.assertFalse(standstill)
    self.assertTrue(tmp and perm)
    for gear in ("R", "N"):
      self.assertEqual(self._faults(ELESYS_CAR, 1, gear, 0.0)[2:], (True, True), msg=gear)

  def test_only_driver_steering_is_ignored_when_parked(self):
    self.assertEqual(self._faults(ELESYS_CAR, 0, "P", 0.0)[2:], (False, False))
    for status in (5, 6, 7):       # FAULT_1, TMP_FAULT, PERMANENT_FAULT: still faults in P
      self.assertTrue(any(self._faults(ELESYS_CAR, status, "P", 0.0)[2:]), msg=f"STEER_STATUS={status}")

  def test_other_hondas_are_unchanged(self):
    self.assertEqual(self._faults(NIDEC_CAR, 1, "P", 0.0)[2:], (True, True))


class TestElesysStockAeb(unittest.TestCase):
  """stockAeb stands openpilot down so the factory CMBS can have the car. The four bits below
  are the ones confirmed on this car by bit-level analysis of a real event; all four read 0
  across 351,809 stock BRAKE_COMMAND frames on bus 2, so none of them can fire spuriously."""

  BITS = ("CMBS_BRAKE", "AEB_REQ_3", "AEB_REQ_2", "AEB_STATUS")

  @staticmethod
  def _stock_aeb(**kw):
    """mirrors the expression in carstate.py"""
    b = {"CMBS_BRAKE": 0, "AEB_REQ_3": 0, "AEB_REQ_2": 0, "AEB_STATUS": 0,
         "FCW": 0, "COMPUTER_BRAKE": 0}
    b.update(kw)
    return bool(b["CMBS_BRAKE"] or b["AEB_REQ_3"] or b["AEB_REQ_2"] or b["AEB_STATUS"] == 1)

  def test_quiet_when_nothing_is_asserted(self):
    self.assertFalse(self._stock_aeb())

  def test_each_confirmed_bit_triggers_on_its_own(self):
    self.assertTrue(self._stock_aeb(CMBS_BRAKE=1))
    self.assertTrue(self._stock_aeb(AEB_REQ_3=1))
    self.assertTrue(self._stock_aeb(AEB_REQ_2=1))
    self.assertTrue(self._stock_aeb(AEB_STATUS=1))

  def test_does_not_wait_for_computer_brake(self):
    # the whole point of the request bits: catch the event before pressure rises
    self.assertTrue(self._stock_aeb(AEB_REQ_3=1, COMPUTER_BRAKE=0))

  def test_warnings_alone_do_not_stand_openpilot_down(self):
    # measured: every non-zero AEB_STATUS / FCW frame in 351,809 stock frames was
    # AEB_STATUS=3 (aeb_prepare) and/or FCW=2 with COMPUTER_BRAKE=0 -- a warning, not braking
    self.assertFalse(self._stock_aeb(AEB_STATUS=3, FCW=2, COMPUTER_BRAKE=0))
    self.assertFalse(self._stock_aeb(AEB_STATUS=2))
    self.assertFalse(self._stock_aeb(FCW=2))

  def test_aeb_req_1_is_not_used(self):
    # bit 29 is the generic Nidec position and never asserts on this car; bit 27 is the real one
    self.assertFalse(self._stock_aeb(AEB_REQ_1=1))

  def test_all_four_bits_are_defined_in_the_dbc(self):
    from opendbc.can import CANParser
    p = CANParser(DBC[ELESYS_CAR][Bus.pt], [], 0)
    sigs = p.vl["BRAKE_COMMAND"]
    for s in self.BITS:
      self.assertIn(s, sigs, msg=f"{s} missing from BRAKE_COMMAND")


class TestElesysTorquePrior(unittest.TestCase):
  """The car's own torqued prior (override.toml) and the offset seed (interface.py).

  The substitute to HONDA_ACCORD (factor 1.689) held torqued at its 1.18 floor on this car, where torque
  1.0 = 2560 on 0x0E4 = 160 serial counts. The prior is also torqued's cache restore key, so these
  numbers only change together with the board's authority / full scale."""

  @staticmethod
  def _toml(name):
    import os
    import tomllib
    from opendbc.car.interfaces import TORQUE_PARAMS_PATH
    with open(os.path.join(os.path.dirname(TORQUE_PARAMS_PATH), name), 'rb') as f:
      return tomllib.load(f)

  @staticmethod
  def _cp(car):
    from opendbc.car.honda.interface import CarInterface
    return CarInterface.get_non_essential_params(car)

  def test_own_prior(self):
    from opendbc.car.interfaces import get_torque_params
    p = get_torque_params()['HONDA_ACCORD_9G_AU']
    self.assertEqual(p['LAT_ACCEL_FACTOR'], 1.25)
    self.assertEqual(p['MAX_LAT_ACCEL_MEASURED'], 1.25)
    self.assertEqual(p['FRICTION'], 0.18)

  def test_honda_accord_prior_unchanged(self):
    # upstream's fleet value for the 2018+ Accord, read straight from params.toml, not re-typed here
    from opendbc.car.interfaces import get_torque_params
    params = self._toml('params.toml')
    want = dict(zip(params['legend'], params['HONDA_ACCORD'], strict=True))
    self.assertEqual(get_torque_params()['HONDA_ACCORD'], want)
    self.assertNotEqual(get_torque_params()['HONDA_ACCORD_9G_AU'], want)

  def test_not_substituted(self):
    # a merge that brings the substitute back would silently restore the 1.689 prior (or, with the
    # override entry, make the loader raise "defined twice")
    self.assertNotIn('HONDA_ACCORD_9G_AU', self._toml('substitute.toml'))
    self.assertTrue('HONDA_ACCORD_9G_AU' in self._toml('override.toml'))
    self.assertNotIn('HONDA_ACCORD_9G_AU', self._toml('params.toml'))

  def test_learnable_window_holds_the_filtered_values(self):
    # torqued clips the raw factor to (1 +- FACTOR_SANITY 0.3) * prior and friction to (1 +- 0.5) * prior.
    # This car's factor rises with speed, 0.67 in town to 1.66 on the highway commute, and no +-30% window
    # holds that (2.5 against 1.86). The 1.25 prior (2026-10-03; 1.1 before) chooses the highway: commuting
    # dominates the engaged steering, and a feedforward that is too strong at speed is the worse error.
    from opendbc.car.interfaces import get_torque_params
    p = get_torque_params()['HONDA_ACCORD_9G_AU']
    lo, hi = 0.7 * p['LAT_ACCEL_FACTOR'], 1.3 * p['LAT_ACCEL_FACTOR']
    self.assertAlmostEqual(lo, 0.875)
    self.assertAlmostEqual(hi, 1.625)
    # Inside: the highway routes. fc/fd/103: a fit on the wire 1.215/1.505/1.39, the controller's own
    # correction 1.28-1.36. The real TorqueEstimator with this prior, chained fc -> fd -> 103 -> 10f from an
    # empty cache: filtered 1.352/1.362/1.459 at the ends of fd/103/10f, 10f's raw median 1.453 and highest
    # 1.590, and no time at either limit.
    for factor in (1.215, 1.28, 1.352, 1.36, 1.362, 1.39, 1.453, 1.459, 1.505, 1.590):
      self.assertTrue(lo < factor < hi, msg=f"{factor} outside {lo}-{hi}")
    # At the ceiling: 10f (the commute) on its own from an empty cache, three methods - a fit on the wire
    # 1.634, the controller's correction 1.65, the TorqueEstimator 1.664. 1.625 is within 3% of all three
    # (that replay with this prior: raw above 1.625 69% of the valid time, filtered 1.578 at the end); the
    # 1.1 prior's 1.43 ceiling sat 12-14% below them, held 100% of the raw and ended at 1.396.
    for factor in (1.634, 1.65, 1.664):
      self.assertLess(abs(factor - hi) / factor, 0.03, msg=f"{factor} vs ceiling {hi}")
    # Clipped at the floor, by design: the town routes (d8/d9/e1/e2, ~60 km/h) give a raw factor down to
    # 0.667, medians 0.791-0.868, and filtered 0.813-0.867 under the 1.1 prior. With this prior, town first
    # (d5..e2 -> fc -> fd -> 103 -> 10f), the raw sits below 0.875 on 97-100% of d9/e1/e2/fc and the
    # filtered factor ends e2 0.882 and fc 0.876, then recovers to 1.353 by the end of 10f.
    for factor in (0.667, 0.791, 0.809, 0.813, 0.867, 0.868):
      self.assertLess(factor, lo, msg=f"{factor} not below the floor {lo}")
    for friction in (0.14, 0.16, 0.18, 0.193, 0.23):
      self.assertTrue(0.5 * p['FRICTION'] <= friction <= 1.5 * p['FRICTION'], msg=f"{friction}")

  def test_car_params_carry_the_prior_and_the_offset_seed(self):
    CP = self._cp(ELESYS_CAR)
    self.assertEqual(CP.lateralTuning.which(), 'torque')
    self.assertAlmostEqual(CP.lateralTuning.torque.latAccelFactor, 1.25, places=6)
    self.assertAlmostEqual(CP.lateralTuning.torque.friction, 0.18, places=6)
    self.assertAlmostEqual(CP.lateralTuning.torque.latAccelOffset, -0.43, places=6)

  def test_other_hondas_keep_a_zero_offset(self):
    # configure_torque_tune() sets 0.0; only the Elesys block seeds it, so every other car is unchanged
    for car in CAR:
      if car in HONDA_ELESYS:
        continue
      CP = self._cp(car)
      if CP.lateralTuning.which() == 'torque':
        self.assertEqual(CP.lateralTuning.torque.latAccelOffset, 0.0, msg=str(car))


def _as_tuple(m):
  return (m[0], bytes(m[1]), m[2]) if isinstance(m, tuple) else (m.address, bytes(m.dat), m.src)


def _controller(car):
  from opendbc.car.honda.carcontroller import CarController
  from opendbc.car.honda.interface import CarInterface
  CP = CarInterface.get_non_essential_params(car)
  CP_SP = CarInterface.get_non_essential_params_sp(CP, car)
  return CarController(car.config.dbc_dict, CP, CP_SP)


def _cc(torque, lat_active=True):
  from opendbc.car import structs
  cc = structs.CarControl.new_message()
  cc.enabled = True
  cc.latActive = lat_active
  cc.actuators.torque = torque
  cc.hudControl.speedVisible = True
  cc.hudControl.setSpeed = 30.0
  return cc.as_reader()


def structs_CC_SP():
  from opendbc.car import structs
  return structs.CarControlSP()


class _FakeCS:
  """What CarController.update() reads from CS (the integration script's double, plus out_sp)."""

  def __init__(self, actuating=True, with_out_sp=True):
    from opendbc.car import structs
    self.out = structs.CarState.new_message()
    self.out.vEgo = 25.0
    self.out.cruiseState.speed = 30.0
    self.out.cruiseState.available = True
    self.v_cruise_factor = 1.0
    self.stock_brake = {"CHIME": 0, "AEB_REQ_1": 0, "AEB_REQ_2": 0, "AEB_STATUS": 0}
    self.acc_hud = {"FCM_OFF": 0, "FCM_OFF_2": 0, "FCM_PROBLEM": 0, "ICONS": 0}
    self.lkas_hud = {}
    self.scm_buttons = {"CRUISE_BUTTONS": 0, "CRUISE_SETTING": 0}
    self.is_metric = True
    self.econ_on = False
    if with_out_sp:
      self.out_sp = structs.CarStateSP()
      self.set_actuating(actuating)

  def set_actuating(self, actuating):
    gw = self.out_sp.linbusGateway
    gw.present = True
    gw.engaged = gw.valid = gw.actuating = actuating


class TestElesysReportedTorque(unittest.TestCase):
  """carOutput's reported torque (new_actuators.torque) is 0.0 while the gateway board is not actuating.

  Only the REPORT changes: torqued fits it, controlsd's steer_limited_by_safety compares against it, the
  torque bar draws it. last_torque, the rate limiter, torqueOutputCan and 0x0E4 must be bit-identical."""

  STEP = 0.03   # STEER_DELTA_UP / DOWN 3 at 100 Hz, every Honda

  def _run(self, car, actuating, requests, cs=None):
    cc_obj = _controller(car)
    cs = cs if cs is not None else _FakeCS()
    out = []
    for i, (act, req) in enumerate(zip(actuating, requests, strict=True)):
      if act is not None:
        cs.set_actuating(act)
      acts, sends = cc_obj.update(_cc(req), structs_CC_SP(), cs, i * int(1e7))
      e4 = [d for a, d, _ in map(_as_tuple, sends) if a in (0xE4, 0x194)]   # STEERING_CONTROL (0x194 on older Nidecs)
      self.assertEqual(len(e4), 1)
      out.append((acts.torque, acts.torqueOutputCan, e4[0], cc_obj.last_torque))
    return out

  def test_not_actuating_reports_zero_and_the_wire_is_unchanged(self):
    n = 120
    req = [1.0] * 60 + [-0.4] * 60
    on = self._run(ELESYS_CAR, [True] * n, req)
    off = self._run(ELESYS_CAR, [False] * n, req)
    for i, (a, b) in enumerate(zip(on, off, strict=True)):
      self.assertEqual(b[0], 0.0, msg=f"frame {i}")
      self.assertAlmostEqual(a[0], a[3], places=6, msg=f"frame {i}: actuating reports last_torque (Float32)")
      self.assertEqual(a[1], b[1], msg=f"frame {i}: torqueOutputCan")
      self.assertEqual(a[2], b[2], msg=f"frame {i}: 0x0E4 bytes")
      self.assertEqual(a[3], b[3], msg=f"frame {i}: last_torque")
    self.assertTrue(any(abs(a[0]) > 0.5 for a in on))
    self.assertTrue(any(abs(a[1]) > 1000 for a in off))   # the command still went out

  def test_resume_continues_from_the_limited_value(self):
    # 10 frames dark while the ramp runs, then the board takes over: the report picks up the ramp
    # where it is (0.33), it does not restart from 0, and the ramp itself never steps by more than 0.03
    seq = [False] * 10 + [True] * 10
    out = self._run(ELESYS_CAR, seq, [1.0] * 20)
    self.assertTrue(all(o[0] == 0.0 for o in out[:10]))
    self.assertAlmostEqual(out[10][0], 11 * self.STEP, places=6)
    self.assertAlmostEqual(out[10][0], out[10][3], places=6)
    for prev, cur in zip(out, out[1:], strict=False):
      self.assertLessEqual(abs(cur[3] - prev[3]), self.STEP + 1e-9)

  def test_default_gateway_state_reports_zero(self):
    # CarStateSP() as it is before the first 0x704: present/valid/actuating all False
    from opendbc.car import structs
    cs = _FakeCS()
    cs.out_sp = structs.CarStateSP()
    out = self._run(ELESYS_CAR, [None] * 20, [0.5] * 20, cs=cs)
    self.assertTrue(all(o[0] == 0.0 for o in out))
    self.assertGreater(abs(out[-1][3]), 0.4)

  def test_other_hondas_are_unaffected(self):
    for car in (CAR.HONDA_CIVIC, CAR.HONDA_ACCORD, CAR.HONDA_CRV):
      out = self._run(car, [False] * 30, [0.6] * 30)
      self.assertTrue(all(abs(o[0] - o[3]) < 1e-6 for o in out), msg=str(car))
      self.assertGreater(abs(out[-1][0]), 0.5, msg=str(car))

  def test_missing_or_odd_gateway_state_behaves_as_before(self):
    # update() must never raise: a CS without out_sp, or with something odd in it, reports last_torque
    class Boom:
      def __bool__(self):
        raise ValueError("odd")

    cases = []
    cs = _FakeCS(with_out_sp=False)
    cases.append(cs)
    cs = _FakeCS(with_out_sp=False)
    cs.out_sp = None  # ty: ignore[invalid-assignment]
    cases.append(cs)
    cs = _FakeCS(with_out_sp=False)
    cs.out_sp = SimpleNamespace()  # ty: ignore[invalid-assignment]
    cases.append(cs)
    cs = _FakeCS(with_out_sp=False)
    cs.out_sp = SimpleNamespace(linbusGateway=SimpleNamespace(actuating=Boom()))  # ty: ignore[invalid-assignment]
    cases.append(cs)
    for cs in cases:
      out = self._run(ELESYS_CAR, [None] * 20, [0.5] * 20, cs=cs)
      self.assertTrue(all(abs(o[0] - o[3]) < 1e-6 for o in out))
      self.assertGreater(abs(out[-1][0]), 0.4)


class TestElesysReportedTorqueSeam(unittest.TestCase):
  """The same rule through the real CarInterface: 0x704 GW_ACTIVE frames -> carstate_ext -> CarController."""

  def setUp(self):
    from opendbc.can import CANPacker
    from opendbc.car.honda.interface import CarInterface
    CP = CarInterface.get_non_essential_params(ELESYS_CAR)
    CP_SP = CarInterface.get_non_essential_params_sp(CP, ELESYS_CAR)
    self.CI = CarInterface(CP, CP_SP)
    self.packer = CANPacker(DBC[ELESYS_CAR][Bus.pt])
    self.i = 0

  def _step(self, gw=None, torque=0.5):
    self.i += 1
    frames = [] if gw is None else [_as_tuple(self.packer.make_can_msg("GW_ACTIVE", 0, gw))]
    _, cs_sp = self.CI.update([(self.i * int(1e7), frames)])
    acts, _ = self.CI.apply(_cc(torque), structs_CC_SP(), self.i * int(1e7))
    return cs_sp.linbusGateway, acts.torque, self.CI.CC.last_torque

  def _hold(self, gw, frames=40):
    out = None
    for _ in range(frames):
      out = self._step(gw)
    return out

  def test_present_valid_engaged_dry_run_and_stale(self):
    # no 0x704 has ever arrived: present (it is, on every Elesys car), not valid, so not actuating
    gw, reported, last = self._hold(None, frames=5)
    self.assertTrue(gw.present)
    self.assertFalse(gw.valid or gw.actuating)
    self.assertEqual(reported, 0.0)
    self.assertGreater(last, 0.1)

    gw, reported, last = self._hold({"ENGAGED": 1, "DRY_RUN": 0})
    self.assertTrue(gw.valid and gw.actuating)
    self.assertAlmostEqual(reported, last, places=6)
    self.assertGreater(reported, 0.4)

    gw, reported, last = self._hold({"ENGAGED": 1, "DRY_RUN": 1})
    self.assertTrue(gw.valid and not gw.actuating)
    self.assertEqual(reported, 0.0)

    gw, reported, last = self._hold({"ENGAGED": 0, "DRY_RUN": 0})
    self.assertFalse(gw.actuating)
    self.assertEqual(reported, 0.0)

    self._hold({"ENGAGED": 1, "DRY_RUN": 0})
    gw, reported, last = self._hold(None, frames=60)  # the board goes quiet: stale, so not valid
    self.assertFalse(gw.valid or gw.actuating)
    self.assertEqual(reported, 0.0)
    self.assertGreater(last, 0.4)                     # the command itself never stopped


class TestElesysSteerDelay(unittest.TestCase):
  """lagd measured ~0.38 s on this car (0.383/0.377 on d3/d4, 0.342 by fd). Both places that read
  steerActuatorDelay bare add 0.2 to it (lagd's initial_lag, and LagdToggle off with the default
  LagdToggleDelay), so the line is 0.18 and both fallbacks land on 0.38 instead of 0.58."""

  LAGD_FALLBACK_ADD = 0.2   # lagd.py initial_lag; LagdToggleDelay default (params_keys.h)

  def test_delay_is_0_18_and_its_fallbacks_are_0_38(self):
    from opendbc.car.honda.interface import CarInterface
    CP = CarInterface.get_non_essential_params(ELESYS_CAR)
    self.assertAlmostEqual(CP.steerActuatorDelay, 0.18, places=6)
    self.assertAlmostEqual(CP.steerActuatorDelay + self.LAGD_FALLBACK_ADD, 0.38, places=6)

  def test_only_the_elesys_block_moves_the_delay(self):
    # With the Elesys gate emptied every car gets upstream's chain. Against that: HONDA_ELESYS differs, and every
    # other Honda is identical - whatever upstream's values are after a merge, not a re-typed 0.1 / 0.15.
    from unittest.mock import patch
    import opendbc.car.honda.interface as honda_interface
    get = honda_interface.CarInterface.get_non_essential_params
    with patch.object(honda_interface, 'HONDA_ELESYS', frozenset()):
      generic = {car: get(car).steerActuatorDelay for car in CAR}
    for car in CAR:
      delay = get(car).steerActuatorDelay
      if car in HONDA_ELESYS:
        self.assertNotAlmostEqual(delay, generic[car], places=6, msg=str(car))
      else:
        self.assertEqual(delay, generic[car], msg=str(car))


class TestElesysTorqueScale(unittest.TestCase):
  """Item 7: openpilot's full scale is the board's full scale. torque 1.0 = 2560 on 0x0E4 = 160 serial counts
  (GW_OP_FULL_SCALE 2560, GW_LIN_AUTHORITY 160); the board clamps at 160 anyway, and openpilot's anti-windup
  and saturation logic are only right if its 1.0 is that 160. A generic branch would give 3840."""

  def test_torque_table_is_2560(self):
    from opendbc.car.honda.interface import CarInterface
    from opendbc.car.honda.values import CarControllerParams
    CP = CarInterface.get_non_essential_params(ELESYS_CAR)
    self.assertEqual(list(CP.lateralParams.torqueBP), [0, 2560])
    self.assertEqual(list(CP.lateralParams.torqueV), [0, 2560])
    params = CarControllerParams(CP)
    self.assertEqual(params.STEER_MAX, 2560)
    self.assertEqual(params.STEER_DELTA_UP, 3)
    self.assertEqual(params.STEER_DELTA_DOWN, 3)

  def test_full_scale_on_the_wire_and_the_domain_bit(self):
    # byte 2 bit 2 is the board's SERIAL_DOMAIN flag; it must stay clear (bits 3:0 are SET_ME_X00_3)
    for req, want in ((1.0, -2560), (-1.0, 2560)):
      cc_obj = _controller(ELESYS_CAR)
      cs = _FakeCS()
      for i in range(60):
        _, sends = cc_obj.update(_cc(req), structs_CC_SP(), cs, i * int(1e7))
      d = next(d for a, d, _ in map(_as_tuple, sends) if a == 0xE4)
      self.assertEqual(int.from_bytes(d[0:2], "big", signed=True), want)
      self.assertEqual(d[2] & 0x04, 0)
      self.assertEqual(d[2] & 0x0F, 0)
      self.assertEqual(d[2] & 0x80, 0x80)   # STEER_TORQUE_REQUEST

  GW_LIN_AUTHORITY = 160   # the board's serial counts at openpilot's full scale (this class's docstring)

  def test_brake_release_steps_on_the_wire(self):
    # the carcontroller comment: under the brake the command is withdrawn by STEER_MAX / BRAKE_RELEASE_FRAMES on
    # 0x0E4 per frame (128 of 2560), which at the board's authority is 8 serial counts per frame - under the 10
    # SP-PROTOCOL-V3 allows and the board's own 40-count step toward zero - and reaches 0 in BRAKE_RELEASE_FRAMES
    from opendbc.car.honda.carcontroller import BRAKE_RELEASE_FRAMES
    from opendbc.car.honda.values import CarControllerParams
    cc_obj = _controller(ELESYS_CAR)
    steer_max = CarControllerParams(cc_obj.CP).STEER_MAX
    cs = _FakeCS()
    wire = []
    for i in range(60 + BRAKE_RELEASE_FRAMES + 5):
      cs.out.brakePressed = i >= 60
      _, sends = cc_obj.update(_cc(1.0), structs_CC_SP(), cs, i * int(1e7))
      d = next(d for a, d, _ in map(_as_tuple, sends) if a == 0xE4)
      wire.append(abs(int.from_bytes(d[0:2], "big", signed=True)))
    release = wire[59:]                     # the last full-scale frame, then the brake
    self.assertEqual(release[0], steer_max)
    self.assertEqual(release[BRAKE_RELEASE_FRAMES], 0)
    self.assertEqual(release[BRAKE_RELEASE_FRAMES - 1], steer_max // BRAKE_RELEASE_FRAMES)
    steps = [a - b for a, b in zip(release, release[1:], strict=False)][:BRAKE_RELEASE_FRAMES]
    self.assertTrue(all(abs(s - steer_max / BRAKE_RELEASE_FRAMES) <= 1 for s in steps), msg=f"{steps}")
    self.assertLessEqual(max(steps) * self.GW_LIN_AUTHORITY / steer_max, 8.1)


if __name__ == "__main__":
  unittest.main()
