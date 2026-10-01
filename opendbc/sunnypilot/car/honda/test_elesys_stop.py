"""FORK(HONDA_ACCORD_9G_AU): the soft final stop (elesys_stop.py), in isolation.

test_dynamic_tuning_integration.py section [19] drives the same code through the real
CarController (the gate, bit-identity elsewhere, the hold, odd inputs); this file pins the ceiling
itself: every timer, the grade term, and the invariants the skeptic review made conditions of
shipping it.

Most episodes ENTER with a light command (60 counts) and then ask for more, which is what the
stopping state looks like on the road: entered with a near-zero accel command (~100 counts with the
creep table), then longcontrol ramps toward stopAccel. The ceiling starts at max(cap, entry), so an
episode that entered at 180 would be held at 180, not at the cap.
"""
import math
import random
import unittest
from types import SimpleNamespace
from unittest import mock

from opendbc.car import structs
from opendbc.sunnypilot.car.honda import elesys_stop as es

LongCtrlState = structs.CarControl.Actuators.LongControlState
DT = es.SOFT_STOP_DT
CAP = int(es.SOFT_STOP_ROLL_CB)
ENTRY = 60


def frames(seconds: float) -> int:
  return int(round(seconds / DT))


class Episode:
  """Feeds soft_stop_ceiling() one 50 Hz frame at a time and keeps what came out."""

  def __init__(self, entry=ENTRY, **kw):
    self.st = None
    self.out, self.ceilings, self.rising = [], [], []
    if entry is not None:
      self.step(entry, **kw)

  def step(self, cmd, wheels_zero=False, a_ego=-0.5, pitch=0.0, stopping=True):
    o, self.st = es.soft_stop_ceiling(stopping, wheels_zero, a_ego, pitch, cmd, self.st)
    self.out.append(o)
    self.ceilings.append(None if self.st is None else self.st.ceiling)
    self.rising.append(self.st is not None and self.st.rising)
    return o

  def run(self, n, cmd, **kw):
    return [self.step(cmd(i) if callable(cmd) else cmd, **kw) for i in range(n)]

  def first_rise(self):
    return next((i for i, r in enumerate(self.rising) if r), None)


class TestSoftStop(unittest.TestCase):
  def test_rolling_is_held_at_the_cap_until_the_wheels_read_zero(self):
    # v=0.6, aEgo=-0.5, the stopping ramp asking for 180: the cap, not 180, until wheel-zero
    ep = Episode()
    out = ep.run(frames(1.5), 180, a_ego=-0.5)
    self.assertEqual(set(out), {CAP})
    self.assertIsNone(ep.first_rise())
    # and the wheels reading zero does not lift it on the spot
    self.assertEqual(ep.step(180, wheels_zero=True), CAP)

  def test_entry_above_the_cap_is_the_ceiling_and_never_walks_down(self):
    ep = Episode(entry=160)
    self.assertEqual(ep.out, [160])                        # never below what entry was commanding
    self.assertEqual(ep.run(10, 200), [160] * 10)          # the ramp cannot add to it
    self.assertEqual(ep.run(5, 100), [100] * 5)            # a lower request passes straight through
    self.assertEqual(ep.run(5, 200), [160] * 5)            # and the ceiling is still 160, not 125
    self.assertTrue(all(c == 160.0 for c in ep.ceilings))

  def test_entry_below_the_cap_passes_through(self):
    ep = Episode(entry=None)
    self.assertEqual([ep.step(c) for c in (0, 30, 90, 124, 125, 126, 200)], [0, 30, 90, 124, 125, 125, 125])

  def test_the_settle_timer_starts_at_wheel_zero_and_holds_the_cap(self):
    ep = Episode()
    ep.run(frames(0.5), 189)
    out = ep.run(frames(1.2), 189, wheels_zero=True)
    held = next(i for i, o in enumerate(out) if o > CAP)
    self.assertGreaterEqual(held * DT, 0.5)                # >= 0.5 s at the cap after wheel-zero
    self.assertLessEqual(held * DT, es.SOFT_STOP_SETTLE)
    self.assertEqual(ep.st.reason, "settle")
    rise = out[held:]
    self.assertEqual(rise[0], CAP + 5)                     # 250 counts/s: 5 per 50 Hz frame
    reached = rise.index(189)
    self.assertLessEqual(reached * DT, 0.3)                # 125 -> 189 in ~0.26 s
    self.assertEqual(set(rise[reached:]), {189})           # then today's hold, exactly
    self.assertFalse(ep.st.armed)                          # and the ceiling is out of the way

  def test_no_escape_while_the_brake_is_still_rising(self):
    # entered with 30 counts and aEgo -0.1 because pressure is still building: the audit's plain
    # 0.4 s escape fired here (ad 394.3, b0 391.0, d3 829.4). Only time AT the ceiling counts.
    ep = Episode(entry=None)
    out = ep.run(frames(1.0), lambda i: min(30 + 5 * i, 189), a_ego=-0.1)
    at_cap = out.index(CAP)                                # the ramp reaches the cap at ~0.38 s
    self.assertGreater(at_cap * DT, 0.35)
    self.assertEqual(max(out), CAP)                        # no rise in the first second
    self.assertIsNone(ep.first_rise())
    # it does escape, but only 0.3 s + 0.4 s after reaching the cap
    ep.run(frames(0.2), 189, a_ego=-0.1)
    self.assertEqual(ep.st.reason, "weak")
    self.assertAlmostEqual((ep.first_rise() - at_cap + 1) * DT, 0.3 + 0.4, delta=1e-9)

  def test_weak_decel_at_the_ceiling_rises(self):
    ep = Episode()
    ep.run(frames(1.0), 180, a_ego=-0.1)
    first, at_cap = ep.first_rise(), ep.out.index(CAP)
    self.assertIsNotNone(first)
    self.assertEqual(ep.st.reason, "weak")
    self.assertAlmostEqual((first - at_cap + 1) * DT, 0.3 + 0.4, delta=1e-9)
    self.assertEqual(ep.out[first], CAP + 5)
    self.assertEqual(set(ep.out[at_cap:first]), {CAP})

  def test_real_deceleration_at_the_ceiling_does_not_escape(self):
    ep = Episode()
    ep.run(frames(1.8), 180, a_ego=-0.3)
    self.assertIsNone(ep.first_rise())

  def test_one_real_deceleration_frame_restarts_the_weak_count(self):
    ep = Episode()
    ep.run(frames(0.3), 180, a_ego=-0.5)                   # 0.3 s at the ceiling
    ep.run(frames(0.3), 180, a_ego=-0.1)
    ep.step(180, a_ego=-0.6)
    ep.run(frames(0.38), 180, a_ego=-0.1)
    self.assertIsNone(ep.first_rise())
    ep.run(frames(0.04), 180, a_ego=-0.1)
    self.assertEqual(ep.st.reason, "weak")

  def test_leaving_the_ceiling_restarts_the_at_ceiling_count(self):
    ep = Episode()
    ep.run(frames(0.28), 180, a_ego=-0.1)
    ep.step(100, a_ego=-0.1)                               # the command drops below the ceiling
    ep.run(frames(0.68), 180, a_ego=-0.1)                  # 0.68 s < 0.3 + 0.4 again
    self.assertIsNone(ep.first_rise())
    ep.step(180, a_ego=-0.1)
    self.assertEqual(ep.st.reason, "weak")

  def test_the_cap_is_grade_aware(self):
    add = es.soft_stop_cap(math.radians(-2.5)) - es.SOFT_STOP_ROLL_CB
    self.assertAlmostEqual(add, 42.0, delta=0.5)          # ~42 counts at -2.5 deg
    per_deg = es.soft_stop_cap(math.radians(-1.0)) - es.SOFT_STOP_ROLL_CB
    self.assertAlmostEqual(per_deg, 17.0, delta=0.3)
    self.assertEqual(es.soft_stop_cap(math.radians(2.5)), es.SOFT_STOP_ROLL_CB)   # uphill: the flat cap
    self.assertEqual(es.soft_stop_cap(0.0), es.SOFT_STOP_ROLL_CB)
    ep = Episode(pitch=math.radians(-2.5))
    self.assertEqual(ep.run(5, 200, pitch=math.radians(-2.5)), [167] * 5)

  def test_a_steeper_downhill_raises_the_ceiling_and_a_gentler_one_does_not_lower_it(self):
    ep = Episode()
    self.assertEqual(ep.run(5, 200, pitch=0.0)[-1], CAP)
    self.assertEqual(ep.run(5, 200, pitch=math.radians(-2.5))[-1], 167)
    self.assertEqual(ep.run(5, 200, pitch=0.0)[-1], 167)
    self.assertEqual(ep.run(5, 200, pitch=math.radians(3.0))[-1], 167)

  def test_no_usable_pitch_means_no_grade_term(self):
    for pitch in (None, float("nan"), float("inf"), -float("inf"), 4.0, "downhill", object()):
      with self.subTest(pitch=pitch):
        self.assertEqual(es.soft_stop_cap(pitch), es.SOFT_STOP_ROLL_CB)
        ep = Episode(pitch=pitch)
        self.assertEqual(ep.run(3, 200, pitch=pitch), [CAP] * 3)

  def test_moving_again_after_wheel_zero_rises_immediately(self):
    ep = Episode()
    ep.run(frames(0.4), 180)
    ep.run(frames(0.2), 180, wheels_zero=True)
    self.assertIsNone(ep.first_rise())
    o = ep.step(180, wheels_zero=False)
    self.assertEqual(ep.st.reason, "moving")
    self.assertEqual(o, CAP + 5)

  def test_entered_at_standstill_is_never_capped(self):
    ep = Episode(entry=189, wheels_zero=True)
    self.assertEqual(ep.run(frames(0.5), 189, wheels_zero=True), [189] * frames(0.5))
    self.assertEqual(ep.run(frames(0.5), 200, wheels_zero=False), [200] * frames(0.5))   # creeping
    self.assertFalse(ep.st.armed)

  def test_outside_the_stopping_gate_there_is_no_cap_and_the_state_resets(self):
    ep = Episode()
    ep.run(10, 180)
    self.assertEqual(ep.step(180, stopping=False), 180)
    self.assertIsNone(ep.st)
    self.assertEqual(ep.step(150), 150)                    # a fresh stop: entry 150 is the ceiling
    self.assertEqual(ep.st.ceiling, 150.0)
    self.assertEqual(ep.st.t, DT)

  def test_max_roll_bounds_the_soft_phase(self):
    ep = Episode()
    ep.run(frames(3.0), 189, a_ego=-0.6)                   # decelerating, never reaching wheel-zero
    first = ep.first_rise()
    self.assertEqual(ep.st.reason, "max_roll")
    self.assertAlmostEqual((first + 1) * DT, es.SOFT_STOP_MAX_ROLL, delta=DT / 2)
    self.assertEqual(set(ep.out[1:first]), {CAP})
    done = next(i for i, c in enumerate(ep.ceilings) if c >= es.SOFT_STOP_DONE_CB)
    self.assertLessEqual((done + 1) * DT, es.SOFT_STOP_MAX_ROLL + (255 - CAP) / es.SOFT_STOP_RISE + 2 * DT)
    self.assertFalse(ep.st.armed)

  def test_rising_is_monotone(self):
    ep = Episode()
    ep.run(frames(0.3), 180)
    ep.run(frames(0.1), 180, wheels_zero=True)
    ep.step(180, wheels_zero=False)                        # moving again: rising
    ep.run(5, 180, a_ego=-2.0, wheels_zero=True)           # strong decel and wheel-zero do not stop it
    ep.run(5, 180, a_ego=-2.0)
    rising = ep.ceilings[ep.first_rise():]
    for a, b in zip(rising, rising[1:], strict=False):
      self.assertAlmostEqual(b - a, es.SOFT_STOP_RISE * DT)

  def test_invariants_on_random_episodes(self):
    # Whatever the inputs: never above the command, never below min(command, entry command), the
    # ceiling never falls, and the whole soft phase ends inside MAX_ROLL + SETTLE + the rise.
    rng = random.Random(20261001)
    bound = es.SOFT_STOP_MAX_ROLL + es.SOFT_STOP_SETTLE + (255 - CAP) / es.SOFT_STOP_RISE + 2 * DT
    for _ in range(400):
      ep = Episode(entry=None)
      entry = None
      p_zero = rng.choice([0.0, 0.05, 0.3])
      for i in range(frames(4.0)):
        cmd = rng.choice([0, rng.randint(0, 255), 189, 254])
        o = ep.step(cmd, wheels_zero=rng.random() < p_zero, a_ego=rng.choice([-1.0, -0.2, 0.1, float("nan")]),
                    pitch=rng.choice([0.0, math.radians(-3), math.radians(3), None, float("nan")]))
        entry = cmd if entry is None else entry
        self.assertLessEqual(o, cmd)
        self.assertGreaterEqual(o, min(cmd, entry))
        if i:
          self.assertGreaterEqual(ep.ceilings[i], ep.ceilings[i - 1])
        if ep.st.armed and (i + 1) * DT > bound:
          self.fail(f"soft phase still armed after {(i + 1) * DT:.2f} s")

  def test_unknown_deceleration_counts_as_not_slowing(self):
    ep = Episode()
    ep.run(frames(0.8), 180, a_ego=float("nan"))
    self.assertEqual(ep.st.reason, "weak")                 # falls back toward today's ramp, never a longer cap


def cs(v_raw=0.6, standstill=False, a_ego=-0.5, gas=False, brake=False, v_ego=None):
  return SimpleNamespace(out=SimpleNamespace(vEgo=v_raw if v_ego is None else v_ego, vEgoRaw=v_raw, standstill=standstill,
                                             aEgo=a_ego, gasPressed=gas, brakePressed=brake))


def cc(state=LongCtrlState.stopping, long_active=True):
  return SimpleNamespace(longActive=long_active, actuators=SimpleNamespace(longControlState=state))


class Tuner:
  def __init__(self, pitch=0.0):
    self.pitch = pitch

  def filtered_pitch(self):
    if isinstance(self.pitch, Exception):
      raise self.pitch
    return self.pitch


class Harness:
  """ElesysSoftStop entered the way the road enters it: one light frame, then the ramp."""

  def __init__(self, c_s=None, tuner=None):
    self.ss = es.ElesysSoftStop()
    self.tuner = Tuner() if tuner is None else tuner
    self.out = [self.ss.update(cc(), cs() if c_s is None else c_s, ENTRY, self.tuner)]

  def run(self, n, c_s, cmd=180, c_c=None):
    self.out = [self.ss.update(cc() if c_c is None else c_c, c_s, cmd, self.tuner) for _ in range(n)]
    return self.out


class TestElesysSoftStop(unittest.TestCase):
  def test_the_wheels_not_vego_start_the_settle_timer(self):
    h = Harness()
    h.run(frames(0.4), cs(v_raw=0.6))
    # vEgo has smoothed below 0.15 but XMISSION_SPEED still reads ~0.3: still rolling, still capped
    self.assertEqual(set(h.run(frames(0.6), cs(v_raw=0.3, v_ego=0.1, a_ego=-0.6))), {CAP})
    self.assertFalse(h.ss.state.wheels_zero_seen)
    self.assertEqual(set(h.run(frames(0.5), cs(v_raw=0.0, v_ego=0.0))), {CAP})   # still settling
    self.assertTrue(h.ss.state.wheels_zero_seen)
    self.assertGreater(max(h.run(frames(0.3), cs(v_raw=0.0, v_ego=0.0))), CAP)

  def test_what_reads_as_the_wheels_at_zero(self):
    self.assertTrue(es.wheels_read_zero(cs(v_raw=0.2, standstill=True)))
    self.assertTrue(es.wheels_read_zero(cs(v_raw=0.0)))
    self.assertFalse(es.wheels_read_zero(cs(v_raw=0.29)))
    self.assertFalse(es.wheels_read_zero(cs(v_raw=0.3, v_ego=0.05)))     # vEgo is not the wheels
    # missing or broken readings count as zero: the fallback is today's behavior, not a longer cap
    self.assertTrue(es.wheels_read_zero(cs(v_raw=float("nan"))))
    self.assertTrue(es.wheels_read_zero(SimpleNamespace(out=SimpleNamespace())))

  def test_pid_state_long_active_false_and_the_pedals_reset_it(self):
    for label, kw_cc, kw_cs in (("pid", dict(state=LongCtrlState.pid), {}),
                                ("starting", dict(state=LongCtrlState.starting), {}),
                                ("off", dict(state=LongCtrlState.off), {}),
                                ("longActive false", dict(long_active=False), {}),
                                ("gas", {}, dict(gas=True)),
                                ("brake", {}, dict(brake=True))):
      with self.subTest(label):
        h = Harness()
        self.assertEqual(set(h.run(10, cs())), {CAP})
        self.assertIsNotNone(h.ss.state)
        self.assertEqual(h.run(1, cs(**kw_cs), c_c=cc(**kw_cc)), [180])
        self.assertIsNone(h.ss.state)

  def test_the_tuner_pitch_feeds_the_cap(self):
    self.assertEqual(Harness(tuner=Tuner(math.radians(-2.5))).run(3, cs(), cmd=200), [167] * 3)
    for pitch in (None, float("nan"), ValueError("no pose")):
      with self.subTest(pitch=pitch):
        self.assertEqual(Harness(tuner=Tuner(pitch)).run(3, cs(), cmd=200), [CAP] * 3)
    ss = es.ElesysSoftStop()
    ss.update(cc(), cs(), ENTRY, None)
    self.assertEqual(ss.update(cc(), cs(), 200, None), CAP)   # no tuner at all: the flat cap

  def test_never_raises_and_never_raises_the_command(self):
    odd = [(None, cs()), (cc(), None), (cc(), SimpleNamespace()), (SimpleNamespace(), cs()),
           (cc(), SimpleNamespace(out=SimpleNamespace(gasPressed=False, brakePressed=False)))]
    for c, s in odd:
      ss = es.ElesysSoftStop()
      for cmd in (0, 125, 180, 254):
        self.assertEqual(ss.update(c, s, cmd, Tuner()), cmd)
    h = Harness()
    with mock.patch.object(es, "soft_stop_ceiling", side_effect=RuntimeError("boom")):
      self.assertEqual(h.run(1, cs()), [180])
    self.assertIsNone(h.ss.state)
    with mock.patch.object(es, "soft_stop_ceiling", return_value=(250, None)):
      self.assertEqual(h.run(1, cs()), [180])              # an answer above the command is refused
    with mock.patch.object(es, "soft_stop_ceiling", return_value=(-3, None)):
      self.assertEqual(h.run(1, cs()), [180])              # and so is a negative one

  def test_one_log_line_per_stop(self):
    with mock.patch.object(es.carlog, "info") as info:
      h = Harness()
      h.run(frames(0.5), cs(v_raw=0.5))
      h.run(frames(1.5), cs(v_raw=0.0))
      lines = [c.args[0] for c in info.call_args_list]
      self.assertEqual(len(lines), 1, lines)
      self.assertTrue(lines[0].startswith("hondastop rise=settle "), lines[0])
      info.reset_mock()
      h = Harness()
      h.run(frames(0.5), cs(v_raw=0.5))
      h.run(1, cs(v_raw=0.5), c_c=cc(state=LongCtrlState.pid))
      lines = [c.args[0] for c in info.call_args_list]
      self.assertEqual(len(lines), 1, lines)
      self.assertTrue(lines[0].startswith("hondastop end=left "), lines[0])
      info.reset_mock()
      h = Harness(c_s=cs(v_raw=0.0))                       # entered at standstill: nothing to say
      h.run(frames(1.0), cs(v_raw=0.0))
      self.assertEqual(info.call_args_list, [])
    with mock.patch.object(es.carlog, "info", side_effect=RuntimeError("log down")):
      h = Harness()
      h.run(frames(0.3), cs(v_raw=0.5))
      h.run(frames(1.0), cs(v_raw=0.0))
      self.assertEqual(h.out[-1], 180)                     # a broken log does not change the command


if __name__ == "__main__":
  unittest.main()
