"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

FORK(BRAKE-LAMP-TEST): the stop-lamp bit test, through the real Honda ELESYS CarController.

What has to hold:
  * param 0 (or unset, or garbage): every frame is byte-for-byte what it is without the test;
  * param N, openpilot holding the car settled at a standstill for DWELL_FRAMES, no pedal: entry
    N's bits are set in the candidate's frame, the checksum is valid, and NOTHING else in any
    frame changes;
  * moving, still rolling (vEgo < 0.1 but no standstill), not yet settled, not braking, a pedal
    pressed, or longitudinal inactive: frames untouched;
  * a param left set is applied at one stop only, until it is written again;
  * no entry can ever set a cancel / fault / AEB / CMBS / hybrid / brake-magnitude bit, or any bit
    in the CMBS byte (1) or the AEB byte (3).
"""
import unittest
from unittest import mock

from opendbc.can.packer import CANPacker
from opendbc.car import structs
from opendbc.car.carlog import carlog
from opendbc.car.honda.carcontroller import CarController
from opendbc.car.honda.hondacan import honda_checksum
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR
from opendbc.sunnypilot.car.honda import brake_lamp_test as blt
from opendbc.sunnypilot.car.honda import dynamic_tuning as dt

LongCtrlState = structs.CarControl.Actuators.LongControlState

PLATFORM = CAR.HONDA_ACCORD_9G_AU
BRAKE_COMMAND = 0x1FA
ACC_HUD = 0x30C
SCM_BUTTONS = 0x1A6

# 0x1FA bits the test must never set (see ALLOWED_BITS): COMPUTER_BRAKE 0-7/14-15, the whole
# CMBS byte 8-15 (pump 8, CMBS 10/12, hybrid pump 11, unnamed 9/13), request 16, cancel 17,
# fault 18, SET_ME_X00_2 19, override 20, the whole AEB byte 24-31 (AEB 24-27/29, the AEB-linked
# 28, unnamed 30, units 31), BRAKE_LIGHTS 39, AEB_STATUS/FCW 40-43, CHIME 45-47, hybrid brake
# 48-55, counter/checksum 56-63
FORBIDDEN_1FA = set(range(21)) | set(range(24, 32)) | set(range(39, 44)) | set(range(45, 64))
# the review's evidence: bit 28 rides with AEB_REQ_1 (29) in real CMBS interventions
AEB_LINKED_1FA = {28, 29}


class FakeParams:
  def __init__(self, value=None):
    self.value = value
    self.reads = 0

  def get(self, key, block=False, return_default=False):
    assert key == blt.PARAM, key
    self.reads += 1
    return self.value

  def get_bool(self, key, block=False):
    return False


class CS:
  def __init__(self):
    self.out = structs.CarState.new_message()
    self.out.cruiseState.speed = 30.0
    self.out.cruiseState.available = True
    self.v_cruise_factor = 1.0
    self.stock_brake = {"CHIME": 0, "AEB_REQ_1": 0, "AEB_REQ_2": 0, "AEB_STATUS": 0}
    self.acc_hud = {"FCM_OFF": 0, "FCM_OFF_2": 0, "FCM_PROBLEM": 0, "ICONS": 0}
    self.lkas_hud = {}
    self.scm_buttons = {"CRUISE_BUTTONS": 0, "CRUISE_SETTING": 0, "MAIN_ON": 1}
    self.is_metric = True


def make_cc(accel, state, long_active=True):
  cc = structs.CarControl.new_message()
  cc.enabled = long_active
  cc.longActive = long_active
  cc.actuators.accel = accel
  cc.actuators.longControlState = state
  cc.hudControl.speedVisible = True
  cc.hudControl.setSpeed = 30.0
  return cc.as_reader()


def build(param_value, platform=PLATFORM, long=True):
  params = FakeParams(param_value)
  with mock.patch.object(dt, "_open_params", lambda: None), mock.patch.object(blt, "_open_params", lambda: params):
    CP = CarInterface.get_non_essential_params(platform)
    CP_SP = CarInterface.get_non_essential_params_sp(CP, platform)
    CP.openpilotLongitudinalControl = long
    CP_SP.enableGasInterceptor = True
    cc = CarController(platform.config.dbc_dict, CP, CP_SP)
  return cc, CP, params


# (name, accel, state, vEgo, standstill, brakePressed, gasPressed, longActive, frames, expect_gate)
# expect_gate means "the physical gate allows it"; the test is then active once that and
# apply_brake > 0 have held for DWELL_FRAMES brake frames in a row (see expected_active()).
# Every phase is an even number of frames: the brake block (and the gate) runs on even frames.
# Nothing drives faster than LATCH_V after the first hold, so the one-stop latch never fires here.
SCRIPT = (
  ("cruising", 0.3, LongCtrlState.pid, 12.0, False, False, False, True, 150, False),
  ("braking, moving", -1.5, LongCtrlState.pid, 6.0, False, False, False, True, 150, False),
  ("creeping to a stop", -0.8, LongCtrlState.stopping, 0.3, False, False, False, True, 80, False),
  # vEgo already under 0.1 but the transmission still turning: the last moments of the stop
  ("stop in progress", -0.8, LongCtrlState.stopping, 0.05, False, False, False, True, 160, False),
  ("standstill hold", -0.8, LongCtrlState.stopping, 0.0, True, False, False, True, 300, True),
  ("hold, driver on the brake", -0.8, LongCtrlState.stopping, 0.0, True, True, False, True, 60, False),
  ("hold again", -0.8, LongCtrlState.stopping, 0.0, True, False, False, True, 160, True),
  ("hold, driver on the gas", -0.8, LongCtrlState.stopping, 0.0, True, False, True, True, 60, False),
  # a hold too short to settle: never active
  ("short hold", -0.8, LongCtrlState.stopping, 0.0, True, False, False, True, 60, True),
  ("hold, standstill flickers", -0.8, LongCtrlState.stopping, 0.0, "flicker", False, False, True, 160, True),
  ("hold again 2", -0.8, LongCtrlState.stopping, 0.0, True, False, False, True, 160, True),
  # still at a standstill while the brake command bleeds off: active exactly as long as it is > 0
  ("pulling away", 1.0, LongCtrlState.pid, 0.0, True, False, False, True, 120, True),
  ("disengaged at a standstill", -0.8, LongCtrlState.off, 0.0, True, False, False, False, 100, False),
)
NEVER_ACTIVE_PHASES = ("cruising", "braking, moving", "creeping to a stop", "stop in progress", "short hold",
                       "hold, standstill flickers", "hold, driver on the brake", "hold, driver on the gas",
                       "disengaged at a standstill")


def standstill_at(standstill, frame: int) -> bool:
  # "flicker": XMISSION_SPEED toggling between 0 and not-quite-0 every 0.2 s
  return (frame // 20) % 2 == 0 if standstill == "flicker" else standstill


def run(cc):
  """Drive the script; return a list of (frame, phase_name, expect_active, lamp_active, apply_brake, sends)."""
  cs = CS()
  out = []
  frame = 0
  for name, accel, state, v, standstill, brake, gas, long_active, n, expect in SCRIPT:
    for _ in range(n):
      cs.out.vEgo = v
      cs.out.standstill = standstill_at(standstill, frame)
      expect_now = expect and cs.out.standstill
      cs.out.brakePressed = brake
      cs.out.gasPressed = gas
      _, sends = cc.update(make_cc(accel, state, long_active), structs.CarControlSP(), cs, frame * int(1e7))
      out.append((frame, name, expect_now, cc.brake_lamp_test.active, cc.apply_brake_last,
                  [(m[0], bytes(m[1]), m[2]) for m in sends]))
      frame += 1
  return out


def expected_active(trace) -> list[bool]:
  """Model of the gate: on each brake (even) frame, active once the physical gate and
  apply_brake > 0 have held for DWELL_FRAMES brake frames in a row; odd frames keep the state."""
  out, dwell, active = [], 0, False
  for f, _name, expect, _active, apply_brake, _sends in trace:
    if f % 2 == 0:
      dwell = dwell + 1 if (expect and apply_brake > 0) else 0
      active = dwell >= blt.DWELL_FRAMES
    out.append(active)
  return out


def diff_bits(a: bytes, b: bytes) -> set[int]:
  return {i * 8 + bit for i in range(len(a)) for bit in range(8) if ((a[i] ^ b[i]) >> bit) & 1}


class TestCandidateList(unittest.TestCase):
  def test_entries_only_use_allowed_bits(self):
    self.assertEqual(blt.ALLOWED_BITS[BRAKE_COMMAND], set(range(64)) - FORBIDDEN_1FA)
    self.assertTrue(blt.ALLOWED_BITS[BRAKE_COMMAND].isdisjoint(AEB_LINKED_1FA))
    self.assertEqual(blt.ALLOWED_BITS[ACC_HUD], {38, 42})
    for i, cand in enumerate(blt.CANDIDATES, 1):
      with self.subTest(entry=i, label=cand.label):
        self.assertIn(cand.addr, (BRAKE_COMMAND, ACC_HUD))
        self.assertTrue(cand.bits)
        self.assertTrue(set(cand.bits) <= blt.ALLOWED_BITS[cand.addr], cand.bits)
        if cand.addr == BRAKE_COMMAND:
          self.assertTrue(set(cand.bits).isdisjoint(FORBIDDEN_1FA))

  def test_risky_entries_are_a_marked_tail(self):
    risky = [c.risky for c in blt.CANDIDATES]
    first = risky.index(True)
    self.assertTrue(all(risky[first:]), "risky entries must all come after the safe ones")
    for cand in blt.CANDIDATES:
      # CRUISE_STATES / SET_ME_X00_3 in 0x1FA and BRAKE_SYSTEM_ICON in 0x30C are CMBS-adjacent unknowns
      cmbs_adjacent = ((cand.addr == BRAKE_COMMAND and bool(set(cand.bits) & {32, 33, 34, 35, 36, 37, 38, 44})) or
                       (cand.addr == ACC_HUD and 38 in cand.bits))
      self.assertEqual(cand.risky, cmbs_adjacent, cand.label)
      if cand.risky:
        self.assertIn("CMBS-adjacent", cand.note)

  def test_labels_fit_the_ui(self):
    labels = [c.label for c in blt.CANDIDATES]
    self.assertEqual(len(labels), len(set(labels)), "labels must be unique")
    for label in labels:
      self.assertTrue(label.isascii() and label.isprintable(), label)
      self.assertLessEqual(len(label), 14, label)
    self.assertEqual(blt.entry_label(0), "off")
    self.assertEqual(blt.entry_label(len(labels) + 1), "off")

  def test_bit_numbers_match_the_dbc(self):
    # the DBC's own signals, packed by the real packer, must equal set_bits() on the same frame --
    # including the recomputed checksum
    packer = CANPacker(PLATFORM.config.dbc_dict["pt"])
    named = [
      (BRAKE_COMMAND, {"SET_ME_X00": 4}, (23,)),
      (BRAKE_COMMAND, {"SET_ME_X00": 2}, (22,)),
      (BRAKE_COMMAND, {"SET_ME_X00": 7}, (21, 22, 23)),
      (BRAKE_COMMAND, {"SET_ME_X00_3": 1}, (44,)),
      (BRAKE_COMMAND, {"CRUISE_STATES": 1}, (32,)),
      (BRAKE_COMMAND, {"CRUISE_STATES": 64}, (38,)),
      (BRAKE_COMMAND, {"BRAKE_LIGHTS": 1}, (39,)),
      (BRAKE_COMMAND, {"COMPUTER_BRAKE_REQUEST": 1}, (16,)),
      (BRAKE_COMMAND, {"AEB_REQ_1": 1}, (29,)),
      (ACC_HUD, {"BOH_4": 1}, (42,)),
      (ACC_HUD, {"BRAKE_SYSTEM_ICON": 1}, (38,)),
    ]
    for addr, extra, bits in named:
      for counter in range(4):
        with self.subTest(addr=hex(addr), extra=extra, counter=counter):
          base = {"COUNTER": counter, "COMPUTER_BRAKE": 120} if addr == BRAKE_COMMAND else {"COUNTER": counter, "PCM_GAS": 10}
          plain = bytes(packer.pack(addr, base))
          ref = bytes(packer.pack(addr, {**base, **extra}))
          self.assertEqual(blt.set_bits(addr, plain, bits), ref)

  def test_checksum_is_valid(self):
    dat = bytes.fromhex("1e0012c0000000" + "20")
    for cand in blt.CANDIDATES:
      out = blt.set_bits(cand.addr, dat, cand.bits)
      self.assertEqual(out[-1] & 0x0F, honda_checksum(cand.addr, None, bytearray(out)))
      self.assertEqual(out[-1] & 0xF0, dat[-1] & 0xF0, "the counter must be left alone")


class TestBrakeLampTestController(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cc, _, _ = build(0)
    cls.reference = run(cc)

  def assert_identical(self, trace, why):
    for (f, name, *_r, sends), (_, _, *_q, ref) in zip(trace, self.reference, strict=True):
      self.assertEqual(sends, ref, f"frame {f} ({name}) differs: {why}")

  def test_script_reaches_a_braked_hold(self):
    for phase in ("stop in progress", "standstill hold", "short hold", "hold, standstill flickers"):
      held = [t for t in self.reference if t[1] == phase]
      self.assertTrue(held, f"the script never reaches {phase!r}")
      self.assertTrue(all(t[4] > 0 for t in held), f"openpilot must be braking in every {phase!r} frame")
    self.assertTrue(any(expected_active(self.reference)), "the dwell is never reached in the script")

  def test_param_off_is_identical(self):
    for value in (None, 0, "0", -1, len(blt.CANDIDATES) + 1, "garbage", 999):
      with self.subTest(value=value):
        cc, _, _ = build(value)
        trace = run(cc)
        self.assertFalse(any(t[3] for t in trace))
        self.assert_identical(trace, f"param {value!r}")

  def test_other_cars_are_untouched(self):
    for platform, long in ((PLATFORM, False), (CAR.HONDA_CIVIC, True)):
      with self.subTest(platform=platform, long=long):
        cc, _, params = build(1, platform=platform, long=long)
        self.assertFalse(cc.brake_lamp_test.applicable)
        self.assertEqual(params.reads, 0, "must not even read the param")

  def test_each_entry_only_in_the_hold(self):
    for entry, cand in enumerate(blt.CANDIDATES, 1):
      with self.subTest(entry=entry, label=cand.label):
        cc, _, _ = build(entry)
        trace = run(cc)
        expected = expected_active(trace)
        seen = 0
        for (f, name, _expect, active, _apply_brake, sends), want, (*_, ref) in zip(trace, expected, self.reference, strict=True):
          self.assertEqual(active, want, f"frame {f} ({name})")
          if name in NEVER_ACTIVE_PHASES:
            self.assertFalse(active, f"frame {f} ({name})")
          self.assertEqual(len(sends), len(ref))
          for (addr, dat, bus), (raddr, rdat, rbus) in zip(sends, ref, strict=True):
            self.assertEqual((addr, bus), (raddr, rbus))
            if active and addr == cand.addr:
              seen += 1
              # exactly the candidate bits, plus whatever the checksum nibble needs
              changed = diff_bits(rdat, dat) - {56, 57, 58, 59}
              self.assertEqual(changed, set(cand.bits) - diff_bits(bytes(len(rdat)), rdat), f"frame {f}")
              for bit in cand.bits:
                self.assertTrue((dat[bit // 8] >> (bit % 8)) & 1)
              self.assertEqual(dat[-1] & 0x0F, honda_checksum(addr, None, bytearray(dat)))
            else:
              self.assertEqual(dat, rdat, f"frame {f} ({name}) 0x{addr:X} changed outside the test window")
        self.assertGreater(seen, 0, "the candidate frame was never modified")

  def test_stand_down_and_brake_value_untouched(self):
    # the SCM_BUTTONS re-send (MAIN_ON=0) and the brake value itself are never altered by any entry
    for entry in range(1, len(blt.CANDIDATES) + 1):
      cc, _, _ = build(entry)
      trace = run(cc)
      for (f, *_x, sends), (*_, ref) in zip(trace, self.reference, strict=True):
        for (addr, dat, _), (_, rdat, _) in zip(sends, ref, strict=True):
          if addr == SCM_BUTTONS:
            self.assertEqual(dat, rdat)
          if addr == BRAKE_COMMAND:
            self.assertEqual((dat[0], dat[1] & 0xC0), (rdat[0], rdat[1] & 0xC0), f"entry {entry} frame {f}")

  def test_param_change_is_picked_up_and_logged(self):
    cc, _, params = build(0)
    cs = CS()
    cs.out.vEgo = 0.0
    cs.out.standstill = True
    hold = make_cc(-0.8, LongCtrlState.stopping)
    frame = 0

    def step(n):
      nonlocal frame
      last = None
      for _ in range(n):
        _, sends = cc.update(hold, structs.CarControlSP(), cs, frame * int(1e7))
        frame += 1
        last = [m for m in sends if m[0] == BRAKE_COMMAND] or last
      return last

    step(100)
    self.assertFalse(cc.brake_lamp_test.active)
    # ERROR, not WARNING: only ERROR reaches errorLogMessage, and so the qlog
    with self.assertLogs(carlog, level="ERROR") as logs:
      params.value = 1
      step(blt.POLL_FRAMES + 2)
      self.assertTrue(cc.brake_lamp_test.active)
      brake = step(2)[0]
      self.assertTrue((brake[1][2] >> 7) & 1, "bit 23 is set")
      params.value = 0
      step(blt.POLL_FRAMES + 2)
      self.assertFalse(cc.brake_lamp_test.active)
    text = "\n".join(logs.output)
    self.assertIn(f"{blt.LOG_TAG} entry 0->1/", text)
    self.assertIn(f"{blt.LOG_TAG} ON entry=1", text)
    self.assertIn(f"{blt.LOG_TAG} frame entry=1 0x1FA ", text)
    self.assertIn(f"{blt.LOG_TAG} entry 1->0/", text)
    self.assertIn(f"{blt.LOG_TAG} OFF entry=0", text)

  def test_param_left_set_applies_at_one_stop_only(self):
    cc, _, params = build(1)
    cs = CS()
    frame = 0

    def drive(accel, state, v, n):
      nonlocal frame
      seen = 0
      cs.out.vEgo = v
      cs.out.standstill = v == 0.0
      for _ in range(n):
        _, sends = cc.update(make_cc(accel, state), structs.CarControlSP(), cs, frame * int(1e7))
        frame += 1
        seen += sum(1 for m in sends if m[0] == BRAKE_COMMAND and (m[1][2] >> 7) & 1)
      return seen

    def stop():
      return drive(-0.8, LongCtrlState.stopping, 0.0, 300)

    with self.assertLogs(carlog, level="ERROR") as logs:
      self.assertGreater(stop(), 0, "first stop: applied")
      self.assertEqual(drive(0.3, LongCtrlState.pid, 12.0, 100), 0)
      self.assertEqual(cc.brake_lamp_test.latched, 1)
      self.assertEqual(stop(), 0, "second stop, param still 1: used up")
      self.assertEqual(drive(0.3, LongCtrlState.pid, 12.0, 100), 0)
      self.assertEqual(stop(), 0, "third stop: still used up")
      params.value = 0                 # off, then the same entry again: re-armed
      stop()
      params.value = 1
      self.assertGreater(stop(), 0, "re-armed by writing the param again")
      # moving slowly (under LATCH_V) does not use it up; driving off does
      drive(0.3, LongCtrlState.pid, 1.5, 100)
      self.assertIsNone(cc.brake_lamp_test.latched)
      drive(0.3, LongCtrlState.pid, 12.0, 100)
      self.assertEqual(cc.brake_lamp_test.latched, 1)
      params.value = 2                 # stepping to the next entry re-arms it too
      self.assertEqual(drive(-0.8, LongCtrlState.stopping, 0.0, 300), 0, "entry 2 is bit 22, not bit 23")
      self.assertTrue(cc.brake_lamp_test.active)
    text = "\n".join(logs.output)
    self.assertIn(f"{blt.LOG_TAG} latched off: entry=1", text)
    self.assertIn(f"{blt.LOG_TAG} re-armed: param 1->0", text)
    self.assertIn(f"{blt.LOG_TAG} re-armed: param 1->2", text)

  def test_other_cars_never_latch(self):
    cc, _, params = build(1, platform=CAR.HONDA_CIVIC)
    self.assertIsNone(cc.brake_lamp_test.latched)
    self.assertEqual(params.reads, 0)


if __name__ == "__main__":
  unittest.main()
