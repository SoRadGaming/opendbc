"""FORK(HONDA_ACCORD_9G_AU): the Elesys radar's decode, pinned on real frames.

REL_SPEED. honda_accord_2015au_radar.dbc (hand-written) carried REL_SPEED at 1/128 m/s until 2026-10-09, which halved
every radar vRel openpilot saw: on bus 1 d(LONG_DIST)/dt was 2.01-2.03 x REL_SPEED, and a stationary object read
-0.50 x vEgo. The scale is 1/64 m/s.

The track slots (2026-10-24, radar_interface.py): an empty slot is raw LONG_DIST 0x7FFF only (returns at 255.0-255.98 m
are real), the update waits for 0x424 (the last track frame of the radar's cycle, which can land in the CAN batch after
0x423), a held NEW_TRACK keeps the slot's trackId only on a frame that continues the same object, and
RadarPoint's measured flag (deprecated.measured) is not COASTING (byte 2 bit 6, FLAG_B22 until then).

These frames are logged ones (routes 15646e8515eda1a7/00000113, 00000141 and 000000dd, bus 1; frames with the same time
arrived in one CAN batch), so a change to the DBC or to radar_interface.py that breaks the decode fails here, whatever
a pack/unpack round trip would say.
"""
import unittest

import numpy as np

from types import SimpleNamespace

from opendbc.can import CANParser
from opendbc.can.dbc import DBC
from opendbc.car.honda.radar_interface import RadarInterface
from opendbc.car.honda.values import CAR

RADAR_DBC = 'honda_accord_2015au_radar'
TRACKS = list(range(0x410, 0x418)) + list(range(0x420, 0x425))

# route 113 t 61.0-63.0 s: slot 0x417 holds a stationary object from 76.3 m to 45.3 m while vEgo is 15.55 m/s (carState,
# flat within 0.5 m/s), with the radar's 0x400 and the last track frames 0x423 and 0x424 around it.
# (ns from the first frame, address, data)
STATIONARY_VEGO = 15.551
STATIONARY = [
  (0, 0x400, "7d0000000000003d"), (0, 0x417, "262a03cf3c330032"), (9758531, 0x423, "7fff000000000038"),
  (9758531, 0x424, "7fff000000000037"),
  (99670439, 0x400, "7d00000000000000"), (99670439, 0x417, "256803d63c26000a"), (109466729, 0x423, "7fff00000000000b"),
  (109466729, 0x424, "7fff00000000000a"),
  (199212858, 0x400, "7d0000000000001f"), (208995399, 0x417, "248e03cf3c2d0013"), (208995399, 0x423, "7fff00000000001a"),
  (208995399, 0x424, "7fff000000000019"),
  (301185301, 0x400, "7d0000000000002e"), (301185301, 0x417, "23db03cd3c200020"), (301185301, 0x423, "7fff000000000029"),
  (301185301, 0x424, "7fff000000000028"),
  (401188496, 0x400, "7d0010000000003c"), (401188496, 0x417, "231803d13c200039"), (401188496, 0x423, "7fff000000000038"),
  (401188496, 0x424, "7fff000000000037"),
  (500912111, 0x400, "7d00000000000000"), (500912111, 0x417, "224a03ca3c1a0007"), (500912111, 0x423, "7fff00000000000b"),
  (500912111, 0x424, "7fff00000000000a"),
  (601049471, 0x400, "7d0000000000001f"), (601049471, 0x417, "218943c43c1a0016"), (601049471, 0x423, "7fff00000000001a"),
  (601049471, 0x424, "7fff000000000019"),
  (701313914, 0x400, "7d0000000000002e"), (701313914, 0x417, "20c843bd3c1a002b"), (701313914, 0x423, "7fff000000000029"),
  (701313914, 0x424, "7fff000000000028"),
  (803692966, 0x400, "7d0000000000003d"), (803692966, 0x417, "1fe103c03c260034"), (803692966, 0x423, "7fff000000000038"),
  (803692966, 0x424, "7fff000000000037"),
  (903142939, 0x400, "7d00000000000000"), (903142939, 0x417, "1f1f03c53c200007"), (903142939, 0x423, "7fff00000000000b"),
  (903142939, 0x424, "7fff00000000000a"),
  (1003839980, 0x400, "7d0000000000001f"), (1003839980, 0x417, "1e5203c13c2d0017"), (1003839980, 0x423, "7fff00000000001a"),
  (1003839980, 0x424, "7fff000000000019"),
  (1103417190, 0x400, "7d0000000000002e"), (1103417190, 0x417, "1d8f03bd3c260023"), (1103417190, 0x423, "7fff000000000029"),
  (1103417190, 0x424, "7fff000000000028"),
  (1202826747, 0x400, "7d0000000000003d"), (1202826747, 0x417, "1cbf03ca3c200038"), (1202826747, 0x423, "7fff000000000038"),
  (1202826747, 0x424, "7fff000000000037"),
  (1305961884, 0x400, "7d00000000000000"), (1305961884, 0x417, "1bfd03d13c200002"), (1305961884, 0x423, "7fff00000000000b"),
  (1305961884, 0x424, "7fff00000000000a"),
  (1405606073, 0x400, "7d0010000000001e"), (1405606073, 0x417, "1b3b03cf3c200012"), (1405606073, 0x423, "7fff00000000001a"),
  (1405606073, 0x424, "7fff000000000019"),
  (1505907025, 0x400, "7d0000000000002e"), (1505907025, 0x417, "1a7703e53c20002a"), (1505907025, 0x423, "7fff000000000029"),
  (1505907025, 0x424, "7fff000000000028"),
  (1606302820, 0x400, "7d0000000000003d"), (1606302820, 0x417, "19b343e63c200035"), (1606302820, 0x423, "7fff000000000038"),
  (1606302820, 0x424, "7fff000000000037"),
  (1705596909, 0x400, "7d00000000000000"), (1705596909, 0x417, "18f143ec3c200001"), (1705596909, 0x423, "7fff00000000000b"),
  (1705596909, 0x424, "7fff00000000000a"),
  (1808025961, 0x400, "7d0000000000001f"), (1808025961, 0x417, "182f43ee3c20001d"), (1808025961, 0x423, "7fff00000000001a"),
  (1808025961, 0x424, "7fff000000000019"),
  (1897975004, 0x400, "7d0000000000002e"), (1907729472, 0x417, "176c43ef3c20002b"), (1907729472, 0x423, "7fff000000000029"),
  (1907729472, 0x424, "7fff000000000028"),
  (1997781378, 0x400, "7d0000000000003d"), (2007556888, 0x417, "16aa43f33c200034"), (2007556888, 0x423, "7fff000000000038"),
  (2007556888, 0x424, "7fff000000000037"),
]

# route 113 t 287.0-288.0 s: B-group slot 0x421 holds an oncoming car closing from 127.3 m to 100.5 m (vEgo 15.2 m/s)
ONCOMING = [
  (0, 0x421, "3fac01d4395a0039"), (100331629, 0x421, "3e5f01da395a0009"), (202363446, 0x421, "3d1201ea395a0019"),
  (302341693, 0x421, "3bc90183394d0023"), (401920779, 0x421, "3a6f012c395a0032"), (502180065, 0x421, "39230132395a000f"),
  (602822523, 0x421, "37ba019a395a0012"), (704969860, 0x421, "367a01ce39600028"), (804591653, 0x421, "3506016e395a0030"),
  (905523900, 0x421, "33ad017c3960000e"), (1005021788, 0x421, "3246015139460014"),
]

# route 141 t 102.515-102.915 s: slot 0x423 holds a car at 58.7 -> 52.4 m. At t=102.715 an oncoming car appears in
# slot 0x424 at 94.68 m (NEW_TRACK), and that 0x424 frame arrived in the CAN batch AFTER its cycle's 0x423 (as 2.3% of
# 0x424 frames do). Updating on 0x423 published that cycle without it, and the 94.68 m frame never reached radard.
LATE_424 = [
  (0, 0x423, "1d5601e53c0d0006"), (0, 0x424, "7fff00000000000a"),
  (90885575, 0x423, "1c9c02463c0d0014"), (90885575, 0x424, "7fff000000000019"),
  (190490476, 0x423, "1bd3028e3c1a002f"),
  (200386810, 0x424, "2f579dcf3b260028"),
  (300110250, 0x423, "1b2a02aa3c2d0030"), (300110250, 0x424, "2e681de53b330038"),
  (400148164, 0x423, "1a3602c13c20000b"), (400148164, 0x424, "2d681e053b3a0200"),
]

# route dd t 804.254-804.764 s: slot 0x412 holds an object at 47.9 m, then (NEW_TRACK, rising) a car 4.5 m to the left
# at 255.34 m, which it tracks on through 255.24, 254.84 and 254.04 m. LONG_DIST < 255 dropped its first two frames.
FAR = [
  (0, 0x412, "17fa5cbb3e200033"), (9758698, 0x424, "7fff000000000037"),
  (100266406, 0x412, "17ac1cbb3e40000b"), (109930677, 0x424, "7fff00000000000a"),
  (199578594, 0x412, "7fac9dc63eed0010"), (210371354, 0x424, "7fff000000000019"),
  (300240261, 0x412, "7f9f1cff3f1a0029"), (310001615, 0x424, "7fff000000000028"),
  (399433177, 0x412, "7f6c1d003f1a003b"), (410246927, 0x424, "7fff000000000037"),
  (499672813, 0x412, "7f051d3b3f060002"), (509402396, 0x424, "7fff00000000000a"),
]

# route 113 t 87.696-88.099 s: slot 0x412 is empty, then an object at 94.04 m with NEW_TRACK, then the SAME object at
# 92.64 m (closing at 11.5 m/s, 0.25 m off the path) with NEW_TRACK still set, then 93.26 and 93.36 m without it.
HELD_SAME = [
  (0, 0x412, "7fff00000000002b"), (0, 0x424, "7fff000000000028"),
  (99875176, 0x412, "2f059d543d200037"), (99875176, 0x424, "7fff000000000037"),
  (203245623, 0x412, "2e529d743d200007"), (203245623, 0x424, "7fff00000000000a"),
  (302871740, 0x412, "2ea11d7a3d130012"), (302871740, 0x424, "7fff000000000019"),
  (402450669, 0x412, "2eae1d983d000028"), (402450669, 0x424, "7fff000000000028"),
]

# route 113 t 31.006-31.510 s: slot 0x411 is empty, then NEW_TRACK on three frames in a row on three DIFFERENT objects
# (90.79 m 12.1 m to the left, 48.95 m 5.1 m to the right, 81.20 m 3.8 m to the right), then the third one at 81.14 m.
HELD_OTHER = [
  (0, 0x411, "7fff00000000003b"), (0, 0x424, "7fff000000000037"),
  (102041088, 0x411, "7fff00000000000e"), (102041088, 0x424, "7fff00000000000a"),
  (203195936, 0x411, "2d6599f83f7a001b"), (203195936, 0x424, "7fff000000000019"),
  (303559335, 0x411, "187a82883f600024"), (303559335, 0x424, "7fff000000000028"),
  (403078786, 0x411, "289a81e03f530031"), (403078786, 0x424, "7fff000000000037"),
  (503671141, 0x411, "289201853f460003"), (503671141, 0x424, "7fff00000000000a"),
]

# route 113 t 63.204-63.306 s: slot 0x417 holds a stationary object 7.9 m to the right at 42.27 m, then is empty (0x7FFF)
EMPTIES = [
  (0, 0x417, "152343fa3c20001f"), (0, 0x424, "7fff000000000019"),
  (91958758, 0x417, "7fff000000000026"),
  (101839944, 0x424, "7fff000000000028"),
]

# route 113 t 56.497-56.799 s: slot 0x410 holds a car at 45.0 -> 43.9 m; on the third frame (43.29 m) COASTING is set:
# the radar extrapolated it, and its REL_SPEED repeats the previous frame's -11.70 m/s.
COASTING = [
  (0, 0x410, "167f1f973d1a0029"), (8753282, 0x424, "7fff000000000028"),
  (99028884, 0x410, "162600093d130034"), (108766008, 0x424, "7fff000000000037"),
  (202368706, 0x410, "15a540433d13000f"), (202368706, 0x424, "7fff00000000000a"),
  (301364205, 0x410, "15f100923ce6001e"), (301364205, 0x424, "7fff000000000019"),
]


def frames(rows):
  return [(t, addr, bytes.fromhex(dat)) for t, addr, dat in rows]


def batches(rows):
  """The rows as the CAN batches they arrived in: consecutive frames with the same time are one batch."""
  out = []
  for t, addr, dat in frames(rows):
    if out and out[-1][0] == t:
      out[-1][1].append((addr, dat, 1))
    else:
      out.append((t, [(addr, dat, 1)]))
  return out


def radar_interface():
  CP = SimpleNamespace(radarUnavailable=False, carFingerprint=CAR.HONDA_ACCORD_9G_AU)
  return RadarInterface(CP, None)


def run(rows):
  """Feed the rows batch by batch: [(t, addresses in the batch, RadarData or None)]."""
  RI = radar_interface()
  return [(t, {a for a, _, _ in batch}, RI.update([(t, batch)])) for t, batch in batches(rows)]


def updates_of(rows, slot):
  """For each published update: the slot's (trackId, dRel, yRel, vRel, measured), or None when it has no point."""
  RI = radar_interface()
  out = []
  for t, batch in batches(rows):
    rr = RI.update([(t, batch)])
    if rr is None:
      continue
    pt = RI.pts.get(slot)
    out.append(None if pt is None else (pt.trackId, round(pt.dRel, 4), round(pt.yRel, 4), round(pt.vRel, 4), pt.deprecated.measured))
    # what was published is what the slot holds
    published = [(p.trackId, round(p.dRel, 4)) for p in rr.points]
    if pt is not None:
      assert (pt.trackId, round(pt.dRel, 4)) in published
  return out


class TestElesysRadarRelSpeed(unittest.TestCase):
  def test_signal_definition_on_all_13_tracks(self):
    dbc = DBC(RADAR_DBC)
    for addr in TRACKS:
      sig = dbc.msgs[addr].sigs['REL_SPEED']
      self.assertEqual((sig.start_bit, sig.size, sig.is_signed, sig.is_little_endian), (37, 14, True, False), hex(addr))
      self.assertEqual(sig.factor, 1 / 64, hex(addr))
      self.assertEqual(sig.offset, 0, hex(addr))

  def test_raw_value(self):
    # bytes 4-5 of the first 0x417 frame are 0x3C33: 14-bit two's complement -973 counts
    cp = CANParser(RADAR_DBC, [(0x417, 10)], 1)
    t, addr, dat = frames(STATIONARY)[1]
    cp.update([(t, [(addr, dat, 1)])])
    self.assertEqual(cp.vl[0x417]['REL_SPEED'], -973 / 64)

  def test_stationary_object_reads_minus_vego(self):
    RI = radar_interface()
    ts, d, v = [], [], []
    for t, addr, dat in frames(STATIONARY):
      rr = RI.update([(t, [(addr, dat, 1)])])
      if rr is not None:
        pt = RI.pts[0x417]
        ts.append(t / 1e9)
        d.append(pt.dRel)
        v.append(pt.vRel)
    self.assertGreaterEqual(len(v), 20)
    # the object stands still, so its range rate is -vEgo (at 1/128 it read about -7.7)
    self.assertAlmostEqual(float(np.median(v)), -STATIONARY_VEGO, delta=0.3)
    # and REL_SPEED is the rate of change of LONG_DIST
    slope = np.polyfit(np.array(ts) - ts[0], d, 1)[0]
    self.assertAlmostEqual(slope / (sum(v) / len(v)), 1.0, delta=0.03)

  def test_b_group_range_rate(self):
    cp = CANParser(RADAR_DBC, [(0x421, 10)], 1)
    ts, d, v = [], [], []
    for t, addr, dat in frames(ONCOMING):
      cp.update([(t, [(addr, dat, 1)])])
      ts.append(t / 1e9)
      d.append(cp.vl[0x421]['LONG_DIST'])
      v.append(cp.vl[0x421]['REL_SPEED'])
    slope = np.polyfit(np.array(ts) - ts[0], d, 1)[0]
    self.assertLess(sum(v) / len(v), -20.0)
    self.assertAlmostEqual(slope / (sum(v) / len(v)), 1.0, delta=0.03)


class TestElesysRadarTracks(unittest.TestCase):
  def test_track_signals_on_all_13_tracks(self):
    dbc = DBC(RADAR_DBC)
    for addr in TRACKS:
      sigs = dbc.msgs[addr].sigs
      self.assertNotIn('FLAG_B22', sigs, hex(addr))
      for name, start, size, factor in (('LONG_DIST', 6, 15, 1 / 128), ('NEW_TRACK', 23, 1, 1), ('COASTING', 22, 1, 1)):
        sig = sigs[name]
        self.assertEqual((sig.start_bit, sig.size, sig.is_signed, sig.factor), (start, size, False, factor), (hex(addr), name))

  def test_the_update_waits_for_0x424(self):
    out = run(LATE_424)
    # one update per radar cycle, on the batch that carries 0x424, never on one that carries 0x423 alone
    self.assertEqual([t for t, addrs, rr in out if rr is not None], [t for t, addrs, _ in out if 0x424 in addrs])
    late = [i for i, (t, addrs, rr) in enumerate(out) if addrs == {0x423}]
    self.assertEqual(len(late), 1)
    self.assertIsNone(out[late[0]][2])
    rr = out[late[0] + 1][2]
    # the cycle whose 0x424 came late is published with that 0x424: the new car at 94.68 m and the 0x423 car at 55.65 m
    self.assertEqual(sorted(round(p.dRel, 2) for p in rr.points), [55.65, 94.68])

  def test_returns_at_255_m_are_tracked(self):
    states = updates_of(FAR, 0x412)
    self.assertEqual([s[1] for s in states], [47.9531, 47.3438, 255.3438, 255.2422, 254.8438, 254.0391])
    ids = [s[0] for s in states]
    # the 47 m object, then (NEW_TRACK rising) the car first seen at 255.34 m, one id from its first frame on
    self.assertEqual(ids[0], ids[1])
    self.assertNotEqual(ids[2], ids[1])
    self.assertEqual(len(set(ids[2:])), 1)

  def test_empty_only_at_raw_7fff(self):
    # 0x424 is 0x7FFF on every frame of FAR: never a point
    self.assertFalse(any(abs(p.dRel - 0x7FFF / 128) < 1e-3 for _, _, rr in run(FAR) if rr is not None for p in rr.points))
    # an occupied slot whose next frame is 0x7FFF loses its point
    self.assertEqual([s if s is None else s[1] for s in updates_of(EMPTIES, 0x417)], [42.2734, None])

  def test_a_held_new_track_on_the_same_object_keeps_its_id(self):
    states = updates_of(HELD_SAME, 0x412)
    self.assertIsNone(states[0])
    self.assertEqual([s[1] for s in states[1:]], [94.0391, 92.6406, 93.2578, 93.3594])
    self.assertEqual(len({s[0] for s in states[1:]}), 1)

  def test_a_held_new_track_on_another_object_is_a_new_id(self):
    states = updates_of(HELD_OTHER, 0x411)
    self.assertEqual(states[:2], [None, None])
    self.assertEqual([s[1] for s in states[2:]], [90.7891, 48.9531, 81.2031, 81.1406])
    ids = [s[0] for s in states[2:]]
    # three objects, each flagged new: three ids, and the third keeps its id once NEW_TRACK clears
    self.assertEqual(len(set(ids[:3])), 3)
    self.assertEqual(ids[2], ids[3])

  def test_measured_is_not_coasting(self):
    states = updates_of(COASTING, 0x410)
    self.assertEqual([s[1] for s in states], [44.9922, 44.2969, 43.2891, 43.8828])
    self.assertEqual([s[4] for s in states], [True, True, False, True])
    self.assertEqual(len({s[0] for s in states}), 1)


if __name__ == "__main__":
  unittest.main()
