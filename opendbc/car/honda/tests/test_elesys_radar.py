"""FORK(HONDA_ACCORD_9G_AU): the Elesys radar's REL_SPEED scale, pinned on real frames.

honda_accord_2015au_radar.dbc (hand-written) carried REL_SPEED at 1/128 m/s until 2026-10-09, which halved every radar
vRel openpilot saw: on bus 1 d(LONG_DIST)/dt was 2.01-2.03 x REL_SPEED, and a stationary object read -0.50 x vEgo.
The scale is 1/64 m/s. These frames are logged ones (route 15646e8515eda1a7/00000113, bus 1), so a change to the DBC
or to radar_interface.py that breaks the scale fails here, whatever a pack/unpack round trip would say.
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

# t 61.0-63.0 s: slot 0x417 holds a stationary object from 76.3 m to 45.3 m while vEgo is 15.55 m/s (carState, flat
# within 0.5 m/s), with the radar's 0x400 and the trigger 0x423 around it. (ns from the first frame, address, data)
STATIONARY_VEGO = 15.551
STATIONARY = [
  (0, 0x400, "7d0000000000003d"), (0, 0x417, "262a03cf3c330032"), (9758531, 0x423, "7fff000000000038"),
  (99670439, 0x400, "7d00000000000000"), (99670439, 0x417, "256803d63c26000a"), (109466729, 0x423, "7fff00000000000b"),
  (199212858, 0x400, "7d0000000000001f"), (208995399, 0x417, "248e03cf3c2d0013"), (208995399, 0x423, "7fff00000000001a"),
  (301185301, 0x400, "7d0000000000002e"), (301185301, 0x417, "23db03cd3c200020"), (301185301, 0x423, "7fff000000000029"),
  (401188496, 0x400, "7d0010000000003c"), (401188496, 0x417, "231803d13c200039"), (401188496, 0x423, "7fff000000000038"),
  (500912111, 0x400, "7d00000000000000"), (500912111, 0x417, "224a03ca3c1a0007"), (500912111, 0x423, "7fff00000000000b"),
  (601049471, 0x400, "7d0000000000001f"), (601049471, 0x417, "218943c43c1a0016"), (601049471, 0x423, "7fff00000000001a"),
  (701313914, 0x400, "7d0000000000002e"), (701313914, 0x417, "20c843bd3c1a002b"), (701313914, 0x423, "7fff000000000029"),
  (803692966, 0x400, "7d0000000000003d"), (803692966, 0x417, "1fe103c03c260034"), (803692966, 0x423, "7fff000000000038"),
  (903142939, 0x400, "7d00000000000000"), (903142939, 0x417, "1f1f03c53c200007"), (903142939, 0x423, "7fff00000000000b"),
  (1003839980, 0x400, "7d0000000000001f"), (1003839980, 0x417, "1e5203c13c2d0017"), (1003839980, 0x423, "7fff00000000001a"),
  (1103417190, 0x400, "7d0000000000002e"), (1103417190, 0x417, "1d8f03bd3c260023"), (1103417190, 0x423, "7fff000000000029"),
  (1202826747, 0x400, "7d0000000000003d"), (1202826747, 0x417, "1cbf03ca3c200038"), (1202826747, 0x423, "7fff000000000038"),
  (1305961884, 0x400, "7d00000000000000"), (1305961884, 0x417, "1bfd03d13c200002"), (1305961884, 0x423, "7fff00000000000b"),
  (1405606073, 0x400, "7d0010000000001e"), (1405606073, 0x417, "1b3b03cf3c200012"), (1405606073, 0x423, "7fff00000000001a"),
  (1505907025, 0x400, "7d0000000000002e"), (1505907025, 0x417, "1a7703e53c20002a"), (1505907025, 0x423, "7fff000000000029"),
  (1606302820, 0x400, "7d0000000000003d"), (1606302820, 0x417, "19b343e63c200035"), (1606302820, 0x423, "7fff000000000038"),
  (1705596909, 0x400, "7d00000000000000"), (1705596909, 0x417, "18f143ec3c200001"), (1705596909, 0x423, "7fff00000000000b"),
  (1808025961, 0x400, "7d0000000000001f"), (1808025961, 0x417, "182f43ee3c20001d"), (1808025961, 0x423, "7fff00000000001a"),
  (1897975004, 0x400, "7d0000000000002e"), (1907729472, 0x417, "176c43ef3c20002b"), (1907729472, 0x423, "7fff000000000029"),
  (1997781378, 0x400, "7d0000000000003d"), (2007556888, 0x417, "16aa43f33c200034"), (2007556888, 0x423, "7fff000000000038"),
]

# t 287.0-288.0 s: B-group slot 0x421 holds an oncoming car closing from 127.3 m to 100.5 m (vEgo 15.2 m/s)
ONCOMING = [
  (0, 0x421, "3fac01d4395a0039"), (100331629, 0x421, "3e5f01da395a0009"), (202363446, 0x421, "3d1201ea395a0019"),
  (302341693, 0x421, "3bc90183394d0023"), (401920779, 0x421, "3a6f012c395a0032"), (502180065, 0x421, "39230132395a000f"),
  (602822523, 0x421, "37ba019a395a0012"), (704969860, 0x421, "367a01ce39600028"), (804591653, 0x421, "3506016e395a0030"),
  (905523900, 0x421, "33ad017c3960000e"), (1005021788, 0x421, "3246015139460014"),
]


def frames(rows):
  return [(t, addr, bytes.fromhex(dat)) for t, addr, dat in rows]


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
    CP = SimpleNamespace(radarUnavailable=False, carFingerprint=CAR.HONDA_ACCORD_9G_AU)
    RI = RadarInterface(CP, None)
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


if __name__ == "__main__":
  unittest.main()
