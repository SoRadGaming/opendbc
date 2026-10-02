"""FORK(HONDA_ACCORD_9G_AU): the VSA fault monitor (vsa_fault.py), through the real CarInterface.

What it must do, all checked on real frames (fixtures/vsa_fault_frames.json.gz: every bus-0 frame of
0x1A4, 0x1EA and 0x1B0 in six windows, cut from the raw rlogs of dongle 15646e8515eda1a7 with LogReader
and grouped exactly as the panda delivered them, one entry per `can` message; 52 KB):

  * a live onset (route 110 episode A, route 112 episode B) sets vsaFault on the very frame
    accFaulted first is, and keeps it;
  * a stored fault at key-on (routes 111, 113) sets vsaStoredFault once the start-up window is
    over, never vsaFault, and route 113's clear at 35.3 km/h clears it;
  * a clean start's bulb check (route 10f) sets nothing;
  * the other lamp state, b3.5 + b4.1 for minutes with braking working (comma_logs route 69),
    sets nothing;
  * no other Honda reports either flag, and no other Honda registers anything new;
  * nothing - missing frames, garbage frames, a monitor that raises - makes CarState.update() raise.
"""
import contextlib
import gzip
import json
import logging
import math
import random
import unittest
from pathlib import Path
from unittest import mock

from opendbc.can import CANParser
from opendbc.car import Bus, structs
from opendbc.car.carlog import carlog
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR, DBC
from opendbc.sunnypilot.car.honda import vsa_fault
from opendbc.sunnypilot.car.honda.vsa_fault import VsaFaultMonitor, STARTUP_WINDOW_FRAMES, STORED_SET_FRAMES, \
  STORED_CLEAR_FRAMES, VSA_SILENT_FRAMES, LIVE_SIGNALS, LAMP_SIGNALS, INERTIAL_INVALID

FIXTURE = Path(__file__).parent / "fixtures" / "vsa_fault_frames.json.gz"
ELESYS = CAR.HONDA_ACCORD_9G_AU
DT_MS = 10.0   # card runs CarState.update() once per panda batch, 100 Hz


def load_scenarios() -> dict:
  with gzip.open(FIXTURE, "rt") as f:
    return json.load(f)


SCENARIOS = load_scenarios()


@contextlib.contextmanager
def quiet_carlog():
  """The replays carry only three of the car's messages, so CANParser warns about every other one; not the point here."""
  level = carlog.level
  carlog.setLevel(logging.CRITICAL)
  try:
    yield
  finally:
    carlog.setLevel(level)


def honda_checksum(addr: int, d: bytes) -> int:
  """The 4-bit Honda checksum over a frame whose last byte's low nibble is still 0 (opendbc/can/dbc.py)."""
  s = 0
  a = addr
  while a:
    s += a & 0xF
    a >>= 4
  for i, x in enumerate(d):
    if i == len(d) - 1:
      x >>= 4
    s += (x & 0xF) + (x >> 4)
  return (8 - s) & 0xF


def make_ci(platform=ELESYS):
  CP = CarInterface.get_non_essential_params(platform)
  CP_SP = CarInterface.get_non_essential_params_sp(CP, platform)
  return CarInterface(CP, CP_SP)


def replay(name: str, platform=ELESYS) -> list[tuple[float, bool, bool, bool]]:
  """Feed a scenario at 100 Hz from its first batch on, delivering each real batch in the 10 ms slot it arrived
  in (an empty update where none did, as card does). Returns (t_s, accFaulted, vsaFault, vsaStoredFault) per
  update; t_s is route time, seconds from the first message of segment 0."""
  batches = SCENARIOS[name]["batches"]
  ci = make_ci(platform)
  out = []
  t = batches[0][0]
  i = 0
  end = batches[-1][0]
  with quiet_carlog():
    while t <= end + DT_MS:
      frames = []
      while i < len(batches) and batches[i][0] < t + DT_MS:
        frames += [(addr, bytes.fromhex(h), 0) for addr, h in batches[i][1]]
        i += 1
      ret, ret_sp = ci.update([(int(t * 1e6), frames)])
      out.append((t / 1000.0, bool(ret.accFaulted), ret_sp.vsaFault, ret_sp.vsaStoredFault))
      t += DT_MS
  return out


def spans(rows, col: int) -> list[tuple[float, float]]:
  res, start, last = [], None, None
  for r in rows:
    if r[col] and start is None:
      start = r[0]
    elif not r[col] and start is not None:
      res.append((start, last))
      start = None
    last = r[0]
  if start is not None:
    res.append((start, last))
  return res


class TestRealFrames(unittest.TestCase):
  def assert_live_onset(self, name: str, onset_s: float):
    rows = replay(name)
    acc = spans(rows, 1)
    live = spans(rows, 2)
    self.assertEqual(len(acc), 1, acc)
    self.assertAlmostEqual(acc[0][0], onset_s, delta=0.011)
    # the property the alert text depends on: no frame has accFaulted without vsaFault
    self.assertTrue(all(r[2] for r in rows if r[1]), "accFaulted on a frame where vsaFault was not yet set")
    self.assertEqual(live, [(acc[0][0], rows[-1][0])], "vsaFault: exactly one span, from the onset frame to the end")
    # the lamp bits follow 20 ms later; this CarState started inside the window, so they count from
    # STARTUP_WINDOW_FRAMES after its first VSA frame
    stored = spans(rows, 3)
    self.assertEqual(len(stored), 1, stored)
    self.assertEqual(stored[0][1], rows[-1][0])

  def test_episode_A_route_110(self):
    self.assert_live_onset("110_episode_A", 1923.16)

  def test_episode_B_route_112(self):
    self.assert_live_onset("112_episode_B", 128.84)

  def test_stored_start_route_111_never_clears_parked(self):
    rows = replay("111_stored_start")
    first_vsa = SCENARIOS["111_stored_start"]["batches"][0][0] / 1000.0
    self.assertFalse(any(r[2] for r in rows), "a stored fault is not a live one")
    stored = spans(rows, 3)
    self.assertEqual(len(stored), 1, stored)
    expected = first_vsa + (STARTUP_WINDOW_FRAMES + STORED_SET_FRAMES - 1) * DT_MS / 1000.0
    self.assertAlmostEqual(stored[0][0], expected, delta=0.03)
    self.assertEqual(stored[0][1], rows[-1][0], "parked, it never clears")

  def test_stored_start_and_the_clear_at_35_kph_route_113(self):
    rows = replay("113_stored_start_and_clear")
    self.assertFalse(any(r[2] for r in rows))
    stored = spans(rows, 3)
    self.assertEqual(len(stored), 1, stored)
    # all fault-lamp bits are gone from the 0x1A4 frame at t=36.898; then STORED_CLEAR_FRAMES more
    self.assertAlmostEqual(stored[0][1], 36.898 + (STORED_CLEAR_FRAMES - 1) * DT_MS / 1000.0, delta=0.03)
    self.assertFalse(rows[-1][3])

  def test_clean_start_bulb_check_route_10f(self):
    rows = replay("10f_clean_start")
    self.assertFalse(any(r[2] or r[3] for r in rows))

  def test_clean_start_bulb_check_with_the_longest_bulb_check_seen(self):
    # 10f's bulb check ends 2.0 s after the VSA's first frame; the longest of 143 clean starts was 3.08 s.
    # The window has to outlast it from the first frame on, whatever card's start time.
    self.assertGreater(STARTUP_WINDOW_FRAMES * DT_MS / 1000.0, 3.08 + 1.0)

  def test_the_other_lamp_state_is_not_a_fault_comma_route_69(self):
    """b3.5 + b4.1 on for 1250 s of this drive while the VSA acknowledged 12,172 frames of braking."""
    rows = replay("comma69_lamp_b3_5_b4_1")
    frames = [bytes.fromhex(h) for _, fr in SCENARIOS["comma69_lamp_b3_5_b4_1"]["batches"] for a, h in fr if a == 0x1A4]
    on = sum(1 for b in frames if b[3] & 0x20 and b[4] & 0x02)
    self.assertGreater(on, 400, "the fixture must actually contain the b3.5+b4.1 state")
    self.assertFalse(any(r[2] or r[3] for r in rows))

  def test_no_other_honda_reports_it(self):
    for platform in (CAR.HONDA_CRV, CAR.HONDA_CIVIC, CAR.ACURA_ILX, CAR.HONDA_ACCORD):
      for name in ("110_episode_A", "113_stored_start_and_clear"):
        rows = replay(name, platform)
        self.assertFalse(any(r[2] or r[3] for r in rows), (platform, name))


class TestDbc(unittest.TestCase):
  """The provisional signals decode the bytes the analysis named, through opendbc's own parser."""

  def decode(self, frames):
    cp = CANParser(DBC[ELESYS][Bus.pt], [("VSA_STATUS", math.nan), ("VEHICLE_DYNAMICS", math.nan),
                                         ("VSA_1AA", math.nan), ("VSA_3D9", math.nan)], 0)
    cp.update([(int(1e9), [(a, bytes.fromhex(h), 0) for a, h in frames])])
    return cp

  def test_live_onset_frame(self):
    # route 110, t=1923.165 / 1923.184: the onset frame, then the frame with the lamps
    cp = self.decode([(0x1A4, "00660c0000000001"), (0x1EA, "000000000000040b")])
    v = cp.vl["VSA_STATUS"]
    self.assertEqual([v[s] for s in LIVE_SIGNALS], [1, 1])
    self.assertEqual([v[s] for s in LAMP_SIGNALS], [0] * len(LAMP_SIGNALS))
    self.assertEqual(cp.vl["VEHICLE_DYNAMICS"][INERTIAL_INVALID], 1)
    cp = self.decode([(0x1A4, "00660ce802000117")])
    v = cp.vl["VSA_STATUS"]
    for s in ("VSA_FAULT_LAMP_B3_3", "VSA_FAULT_LAMP_B3_5", "VSA_FAULT_LAMP_B3_6", "VSA_FAULT_LAMP_B3_7",
              "VSA_FAULT_LAMP_B4_1", "VSA_FAULT_LAMP_B6_0"):
      self.assertEqual(v[s], 1, s)
    self.assertEqual(v["VSA_FAULT_STORED_B4_0"], 0, "b4.0 is a stored-fault bit, not a live one")
    self.assertEqual(v["ESP_DISABLED"], 0)

  def test_stored_fault_frame(self):
    # route 113 after its bulb check: byte 4 = 0x03, b4.0 is what a live onset never has
    frames = [bytes.fromhex(h) for _, fr in SCENARIOS["113_stored_start_and_clear"]["batches"] for a, h in fr if a == 0x1A4]
    stored = next(b for b in frames if b[4] == 0x03)
    cp = self.decode([(0x1A4, stored.hex())])
    self.assertEqual(cp.vl["VSA_STATUS"]["VSA_FAULT_STORED_B4_0"], 1)
    self.assertEqual(cp.vl["VSA_STATUS"]["VSA_FAULT_LAMP_B6_0"], 1)

  def test_bulb_check_frame(self):
    # route 10f's first frame: b3.1-b3.7 minus b3.3, b4.4 - the lamp test, none of the fault-only bits
    cp = self.decode([(0x1A4, "0fff00f61060203b")])
    v = cp.vl["VSA_STATUS"]
    self.assertEqual(v["ESP_DISABLED"], 1)
    self.assertEqual((v["VSA_FAULT_LAMP_B3_6"], v["VSA_FAULT_LAMP_B3_7"]), (1, 1))
    for s in ("VSA_FAULT_LAMP_B3_3", "VSA_FAULT_STORED_B4_0", "VSA_FAULT_LAMP_B6_0") + LIVE_SIGNALS:
      self.assertEqual(v[s], 0, s)

  def test_the_two_new_messages_check_out(self):
    # real frames, route 110 after episode A: Honda checksum and counter as the DBC declares them
    cp = self.decode([(0x1AA, "7fff03000000000c"), (0x3D9, "008502")])
    self.assertNotEqual(cp.ts_nanos["VSA_1AA"]["VSA_FAULT_LIVE_B2_0"], 0, "0x1AA rejected: checksum or layout wrong")
    self.assertNotEqual(cp.ts_nanos["VSA_3D9"]["VSA_FAULT_LAMP_B1_0"], 0, "0x3D9 rejected: checksum or layout wrong")
    self.assertEqual((cp.vl["VSA_1AA"]["VSA_FAULT_LIVE_B2_0"], cp.vl["VSA_1AA"]["VSA_FAULT_LIVE_B2_1"]), (1, 1))
    self.assertEqual((cp.vl["VSA_3D9"]["VSA_FAULT_LAMP_B1_0"], cp.vl["VSA_3D9"]["VSA_FAULT_LAMP_B1_2"]), (1, 1))

  def test_only_this_cars_dbc_has_them(self):
    for platform in (CAR.HONDA_CRV, CAR.HONDA_CIVIC, CAR.ACURA_ILX):
      cp = CANParser(DBC[platform][Bus.pt], [], 0)
      self.assertNotIn("VSA_FAULT_LIVE_B2_2", cp.vl["VSA_STATUS"], platform)


class TestParserRegistration(unittest.TestCase):
  def test_vehicle_dynamics_is_liveness_exempt_on_this_car_only(self):
    ci = make_ci()
    st = ci.can_parsers[Bus.pt].message_states[0x1EA]
    self.assertTrue(st.ignore_alive, "a missing 0x1EA must never cost openpilot its CAN")
    for platform in (CAR.HONDA_CRV, CAR.HONDA_CIVIC):
      ci = make_ci(platform)
      with quiet_carlog():
        for _ in range(3):
          ci.update([(int(1e9), [])])
      self.assertNotIn(0x1EA, ci.can_parsers[Bus.pt].message_states, platform)

  def test_missing_vsa_frames_read_false(self):
    ci = make_ci()
    with quiet_carlog():
      for i in range(300):
        ret, ret_sp = ci.update([(int((i + 1) * 1e7), [])])
        self.assertFalse(ret_sp.vsaFault or ret_sp.vsaStoredFault)


class TestNeverRaises(unittest.TestCase):
  def test_garbage_frames(self):
    """Random lengths and bytes: the parser drops most of them on the checksum, nothing raises."""
    rng = random.Random(20261001)
    ci = make_ci()
    addrs = (0x1A4, 0x1EA, 0x1B0, 0x1AA, 0x3D9)
    with quiet_carlog():
      for i in range(3000):
        frames = [(rng.choice(addrs), bytes(rng.getrandbits(8) for _ in range(rng.randint(0, 8))), 0)
                  for _ in range(rng.randint(0, 6))]
        ret, ret_sp = ci.update([(int((i + 1) * 1e7), frames)])
        self.assertIsInstance(ret_sp.vsaFault, bool)
        self.assertIsInstance(ret_sp.vsaStoredFault, bool)

  def test_random_frames_that_pass_the_checksum(self):
    """Random payloads with a valid Honda counter and checksum reach the monitor as decoded values: nothing raises."""
    rng = random.Random(20261002)
    ci = make_ci()
    counters = {0x1A4: 0, 0x1EA: 0, 0x1B0: 0}
    lengths = {0x1A4: 8, 0x1EA: 8, 0x1B0: 7}
    with quiet_carlog():
      for i in range(1500):
        frames = []
        for addr in counters:
          d = bytearray(rng.getrandbits(8) for _ in range(lengths[addr]))
          counters[addr] = (counters[addr] + 1) & 3
          d[-1] = (counters[addr] << 4)
          d[-1] |= honda_checksum(addr, d)
          frames.append((addr, bytes(d), 0))
        ret, ret_sp = ci.update([(int((i + 1) * 1e7), frames)])
        self.assertIsInstance(ret_sp.vsaFault, bool)
        self.assertIsInstance(ret_sp.vsaStoredFault, bool)
    cp = ci.can_parsers[Bus.pt]
    self.assertGreater(cp.ts_nanos["VSA_STATUS"]["VSA_FAULT_LIVE_B2_2"], 0, "the frames must actually pass the parser")
    self.assertGreater(cp.ts_nanos["VEHICLE_DYNAMICS"]["VSA_FAULT_INERTIAL_INVALID"], 0)

  def test_a_monitor_that_raises_reads_as_no_fault(self):
    ci = make_ci()
    batches = SCENARIOS["110_episode_A"]["batches"]
    with mock.patch.object(vsa_fault.VsaFaultMonitor, "update", side_effect=RuntimeError("boom")), quiet_carlog():
      for t, fr in batches[-50:]:
        ret, ret_sp = ci.update([(int(t * 1e6), [(a, bytes.fromhex(h), 0) for a, h in fr])])
        self.assertFalse(ret_sp.vsaFault or ret_sp.vsaStoredFault)
    self.assertTrue(ret.accFaulted, "the rest of CarState carried on: accFaulted still reads BRAKE_ERROR")

  def test_odd_values_into_the_monitor(self):
    m = VsaFaultMonitor()
    for vsa, dyn in ((None, None), ({}, {}), ({"VSA_FAULT_LIVE_B2_2": float("nan")}, {INERTIAL_INVALID: "x"}),
                     ({s: float("inf") for s in LIVE_SIGNALS + LAMP_SIGNALS}, {INERTIAL_INVALID: None})):
      for i in range(STARTUP_WINDOW_FRAMES + STORED_SET_FRAMES + 5):
        live, stored = m.update(i + 1, vsa, i + 1, dyn, True)
        self.assertIsInstance(live, bool)
        self.assertIsInstance(stored, bool)
        self.assertFalse(live or stored, (vsa, dyn))


class Feed:
  """Drives a VsaFaultMonitor frame by frame. fresh=False repeats the last timestamps: the VSA went quiet."""
  def __init__(self):
    self.m = VsaFaultMonitor()
    self.ts = 0

  def run(self, frames: int, vsa: dict, dyn: dict | None = None, brake_error: bool = False, fresh: bool = True):
    out = []
    for _ in range(frames):
      if fresh:
        self.ts += 1
      out.append(self.m.update(self.ts, vsa, self.ts, dyn or {}, brake_error))
    return out


class TestMonitorTiming(unittest.TestCase):
  LAMPS = {s: 1 for s in LAMP_SIGNALS}

  def test_lamps_inside_the_window_never_count(self):
    f = Feed()
    out = f.run(STARTUP_WINDOW_FRAMES - 1, self.LAMPS) + f.run(1, {})
    self.assertFalse(any(s for _, s in out))

  def test_lamps_after_the_window_count_after_the_debounce(self):
    out = Feed().run(STARTUP_WINDOW_FRAMES + STORED_SET_FRAMES + 10, self.LAMPS)
    first = next(i for i, (_, s) in enumerate(out) if s)
    self.assertEqual(first, STARTUP_WINDOW_FRAMES + STORED_SET_FRAMES - 2)

  def test_a_short_lamp_blip_after_the_window_is_ignored(self):
    f = Feed()
    f.run(STARTUP_WINDOW_FRAMES, {})
    out = f.run(STORED_SET_FRAMES - 1, self.LAMPS) + f.run(5, {})
    self.assertFalse(any(s for _, s in out))

  def test_it_clears_after_the_clear_debounce(self):
    f = Feed()
    f.run(STARTUP_WINDOW_FRAMES + STORED_SET_FRAMES, self.LAMPS)
    self.assertTrue(f.m.vsa_stored_fault)
    out = f.run(STORED_CLEAR_FRAMES - 1, {})
    self.assertTrue(all(s for _, s in out))
    self.assertEqual(f.run(1, {}), [(False, False)])

  def test_silence_reads_false_and_restarts_the_window(self):
    f = Feed()
    f.run(STARTUP_WINDOW_FRAMES + STORED_SET_FRAMES, self.LAMPS)
    self.assertTrue(f.m.vsa_stored_fault)
    # the VSA goes quiet: the same timestamps for VSA_SILENT_FRAMES updates
    out = f.run(VSA_SILENT_FRAMES, self.LAMPS, fresh=False)
    self.assertTrue(out[-2][1], "still stored one frame before the silence counts")
    self.assertEqual(out[-1], (False, False))
    # it comes back with a bulb check: not counted, the window starts again
    out = f.run(STARTUP_WINDOW_FRAMES - 1, self.LAMPS)
    self.assertFalse(any(s for _, s in out))

  def test_never_received_is_false(self):
    m = VsaFaultMonitor()
    for _ in range(1000):
      self.assertEqual(m.update(0, {s: 1 for s in LIVE_SIGNALS + LAMP_SIGNALS}, 0, {INERTIAL_INVALID: 1}, True), (False, False))

  def test_live_needs_no_window(self):
    self.assertEqual(Feed().run(1, {"VSA_FAULT_LIVE_B2_3": 1}), [(True, False)])

  def test_inertial_invalid_is_live_only_with_brake_error_and_after_the_window(self):
    f = Feed()
    dyn = {INERTIAL_INVALID: 1}
    # inside the start-up window it never counts (BRAKE_ERROR is up on the VSA's first frame at key-on)
    self.assertFalse(any(live for live, _ in f.run(STARTUP_WINDOW_FRAMES - 1, {}, dyn, brake_error=True)))
    # after it: only with BRAKE_ERROR, which a stored fault does not carry
    self.assertFalse(any(live for live, _ in f.run(10, {}, dyn, brake_error=False)))
    self.assertTrue(all(live for live, _ in f.run(10, {}, dyn, brake_error=True)))


class TestDefaults(unittest.TestCase):
  def test_struct_defaults_false(self):
    cs = structs.CarStateSP()
    self.assertIs(cs.vsaFault, False)
    self.assertIs(cs.vsaStoredFault, False)


if __name__ == "__main__":
  unittest.main()
