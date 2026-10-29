"""FORK(BRAKE-LAMP-RELAY): the BLR-v1 brake-lamp board's telemetry, from the wire to carStateSP.brakeLampRelay.

The board's bytes are its own: fixtures/blr_golden_frames.json is the firmware's blr/tools/golden_frames.json, copied
byte for byte (board repo S:/Software/EPS-LKAS, branch blr-v1, read at 31c138f). blr_logic_test case 22 packs those
bytes with the SHIPPING blr_pack(), and blr_dbc_test.py decodes them against the firmware's blr/dbc/blr.dbc - so a
decode here that agrees with the vector agrees with the board. What this file holds:

  * the car's generated DBC decodes every golden telemetry frame to exactly the firmware's intended values, and each
    BLR message has exactly the vector's signals - no more, no fewer, so _sunnypilot_blr.dbc is the firmware's copy;
  * whenever the firmware tree is reachable (BLR_FIRMWARE_DIR, or the owner's S: drive from WSL): the copies against
    the firmware's own files - every BO_/SG_/CM_/VAL_ line of the seven telemetry frames against blr/dbc/blr.dbc, and
    the fixture against blr/tools/golden_frames.json, byte for byte. Checked only against each other, a layout change
    made in both of the firmware's files would pass everything above (the BLR-v1 review, lens B);
  * the seven frames are registered liveness-exempt on HONDA_ELESYS only, and no signal is named COUNTER/CHECKSUM;
  * the decoder, through the real CarInterface: status valid for 1 s and then all-zero, identity latched, analog
    gated, a decoder that raises leaves defaults and never raises into CarState.update();
  * THE BOARD NEVER REACHES CarState: with the BLR's frames on the bus - a status claiming DRIVER_BRAKING, PEDAL_CAN
    and EXT_SINK among them - every CarState field, brakePressed included, is identical frame for frame to a run
    without them; and in source, nothing outside the decoder, the parser registration, the struct and the 0x17C
    BRAKE_SWITCH mirror's gate names the board, and panda safety does not mention its IDs at all (DESIGN E.7).
"""
import ast
import contextlib
import json
import logging
import math
import os
import re
import unittest
from pathlib import Path
from unittest import mock

from opendbc.can import CANPacker, CANParser
from opendbc.car import Bus, structs
from opendbc.car.carlog import carlog
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR, DBC
from opendbc.sunnypilot.car.honda import carstate_ext
from opendbc.sunnypilot.car.honda.carstate_ext import BLR_FAULT_SIGNALS, BLR_LATCH_REASONS, BLR_STATUS_STALE_FRAMES, \
  BLR_ANALOG_STALE_FRAMES

HERE = Path(__file__).parent
OPENDBC = HERE.parents[2]   # opendbc/
GOLDEN = HERE / "fixtures" / "blr_golden_frames.json"
ELESYS = CAR.HONDA_ACCORD_9G_AU
CAR_DBC = DBC[ELESYS][Bus.pt]
DT_NS = 10_000_000   # card runs CarState.update() at 100 Hz

TELEMETRY = {0x620: "BLR_STATUS", 0x621: "BLR_ANALOG", 0x622: "BLR_HEALTH", 0x623: "BLR_COUNTERS",
             0x624: "BLR_COUNTERS2", 0x627: "BLR_VERSION", 0x62F: "BLR_BUILD"}
BOOT_CHANNEL = (0x628, 0x629, 0x62A)
COPY_DBC = OPENDBC / "dbc" / "generator" / "honda" / "_sunnypilot_blr.dbc"

# The firmware tree, when it is reachable: the EPS-LKAS repo's blr/ (sunnypilot's test_blr_flasher.py looks in the
# same places).
FIRMWARE_BLR = [Path(p) for p in (os.environ.get("BLR_FIRMWARE_DIR", ""),
                                  "/mnt/s/Software/EPS-LKAS/blr",
                                  "/mnt/s/Software/EPS-LKAS/.claude/worktrees/blr/blr") if p]


def firmware_dir() -> Path | None:
  for p in FIRMWARE_BLR:
    if (p / "dbc" / "blr.dbc").is_file() and (p / "tools" / "golden_frames.json").is_file():
      return p
  return None


def dbc_definition(text: str, ids) -> set[str]:
  """What a DBC says about the messages `ids`, as a set of whitespace-normalised lines: each BO_ line, each of its SG_
  lines (prefixed with its message's ID), and every CM_ BO_ / CM_ SG_ / VAL_ line naming one of them. Order and
  layout are not definition; anything else - the file header, the other messages - is left out."""
  out: set[str] = set()
  msg = None
  for raw in text.splitlines():
    line = " ".join(raw.split())
    if line.startswith("BO_ "):
      msg = int(line.split()[1])
      if msg in ids:
        out.add(line)
    elif line.startswith("SG_ "):
      if msg in ids:
        out.add(f"{msg} {line}")
    else:
      msg = None
      m = re.match(r"(?:CM_ (?:BO_|SG_)|VAL_) (\d+) ", line)
      if m and int(m.group(1)) in ids:
        out.add(line)
  return out


def load_golden() -> dict[str, dict]:
  with open(GOLDEN, encoding="utf-8") as f:
    return {fr["tag"]: fr for fr in json.load(f)["frames"]}


G = load_golden()


def frame(tag: str) -> tuple[int, bytes, int]:
  return G[tag]["id"], bytes.fromhex(G[tag]["hex"]), 0


@contextlib.contextmanager
def quiet_carlog():
  """These runs carry a handful of the car's messages, so CANParser warns about every other one; not the point here."""
  level = carlog.level
  carlog.setLevel(logging.CRITICAL)
  try:
    yield
  finally:
    carlog.setLevel(level)


def make_ci(platform=ELESYS):
  CP = CarInterface.get_non_essential_params(platform)
  CP_SP = CarInterface.get_non_essential_params_sp(CP, platform)
  return CarInterface(CP, CP_SP)


class Run:
  """One CarInterface, stepped at 100 Hz; each step delivers the given frames, or none."""
  def __init__(self, platform=ELESYS):
    self.ci = make_ci(platform)
    self.t = DT_NS

  def step(self, frames=()):
    with quiet_carlog():
      ret, ret_sp = self.ci.update([(self.t, list(frames))])
    self.t += DT_NS
    return ret, ret_sp


class TestBlrDbc(unittest.TestCase):
  def test_the_vector_is_the_firmwares(self):
    """Every telemetry ID has a golden frame, and the boot channel is not part of the copy."""
    ids = {fr["id"] for fr in G.values()}
    assert set(TELEMETRY) <= ids, sorted(set(TELEMETRY) - ids)
    assert "source" in json.loads(GOLDEN.read_text(encoding="utf-8")), "not the firmware's golden_frames.json"

  def test_dbc_decodes_every_golden_frame(self):
    """The byte order is the test, as for GW_VERSION: every BLR field is Intel (@1+), and a wrong one decodes to a
    plausible number. Decoded by the parser card runs, on the car's generated DBC."""
    checked = 0
    for tag, fr in G.items():
      if fr["id"] not in TELEMETRY:
        continue
      name = TELEMETRY[fr["id"]]
      cp = CANParser(CAR_DBC, [(name, math.nan)], 0)
      cp.update([(10**9, [frame(tag)])])
      got = cp.vl[name]
      self.assertEqual(set(got), set(fr["signals"]), f"{tag}: the DBC's {name} and the firmware's vector differ in signals")
      for sig, want in fr["signals"].items():
        self.assertAlmostEqual(got[sig], want, msg=f"{tag} {name}.{sig}: decoded {got[sig]}, the board meant {want}")
        checked += 1
    self.assertGreater(checked, 80)

  def test_the_copies_are_the_firmwares_files(self):
    """Against the firmware's own blr/dbc/blr.dbc and blr/tools/golden_frames.json, when the tree is reachable: the
    seven telemetry frames defined line for line as the firmware defines them, and the vector byte for byte (line
    endings aside). Elsewhere the two copies are checked against each other only, above."""
    fw = firmware_dir()
    if fw is None:
      self.skipTest("the EPS-LKAS blr/ tree is not reachable (set BLR_FIRMWARE_DIR); the copies were checked against each other")
    ids = set(TELEMETRY)
    ours = dbc_definition(COPY_DBC.read_text(encoding="utf-8"), ids)
    theirs = dbc_definition((fw / "dbc" / "blr.dbc").read_text(encoding="utf-8"), ids)
    assert {int(ln.split()[1]) for ln in theirs if ln.startswith("BO_ ")} == ids, "the firmware's DBC lost a frame"
    assert len(theirs) > 100, len(theirs)
    assert ours == theirs, (f"_sunnypilot_blr.dbc is not the firmware's blr.dbc: only here {sorted(ours - theirs)}, " +
                            f"only there {sorted(theirs - ours)}")
    # and nothing but those seven is copied: the boot channel and the firmware's header stay out
    assert dbc_definition(COPY_DBC.read_text(encoding="utf-8"), set(range(0x600, 0x700)) - ids) == set()
    golden = (fw / "tools" / "golden_frames.json").read_bytes().replace(b"\r\n", b"\n")
    assert GOLDEN.read_bytes().replace(b"\r\n", b"\n") == golden, \
      "fixtures/blr_golden_frames.json is not the firmware's blr/tools/golden_frames.json: copy it again"

  def test_boot_channel_is_not_in_the_dbc(self):
    """Offroad-only update traffic: not copied, so nothing can register it."""
    cp = CANParser(CAR_DBC, [], 0)
    for addr in BOOT_CHANNEL:
      assert addr not in cp.dbc.addr_to_msg, hex(addr)

  def test_no_counter_or_checksum_signal(self):
    """In a honda_ DBC those two names make the parser demand a Honda checksum the board does not compute."""
    cp = CANParser(CAR_DBC, [], 0)
    for addr, name in TELEMETRY.items():
      msg = cp.dbc.addr_to_msg[addr]
      assert msg.name == name and msg.size == 8
      assert not {"COUNTER", "CHECKSUM"} & set(msg.sigs), name
      assert all(s.is_little_endian for s in msg.sigs.values()), f"{name}: the board packs LSB first"

  def test_fault_and_latch_names_are_the_dbcs(self):
    """faults bit i is 0x620 bit 16 + i; the latch names are the DBC's VAL_ table."""
    cp = CANParser(CAR_DBC, [], 0)
    sigs = cp.dbc.addr_to_msg[0x620].sigs
    for i, name in enumerate(BLR_FAULT_SIGNALS):
      assert sigs[name].start_bit == 16 + i and sigs[name].size == 1, name
    assert len(BLR_FAULT_SIGNALS) == 8
    val = next(v for v in cp.dbc.vals if v.address == 0x620 and v.name == "LATCH_REASON")
    words = val.def_val.split()
    table = {int(words[i]): words[i + 1] for i in range(0, len(words), 2)}
    assert table == {0: "NONE", **BLR_LATCH_REASONS}, table


class TestBlrDecoder(unittest.TestCase):
  def test_registered_liveness_exempt_on_elesys_only(self):
    cp = make_ci().can_parsers[Bus.pt]
    states = {s.name: s for s in cp.message_states.values()}
    for name in TELEMETRY.values():
      assert name in states, f"{name} is not registered"
      assert states[name].ignore_alive, f"{name} is liveness-checked: a quiet board would cost openpilot its CAN"

    other = make_ci(CAR.HONDA_CIVIC).can_parsers[Bus.pt]
    assert not set(TELEMETRY) & other.addresses
    _, ret_sp = Run(CAR.HONDA_CIVIC).step([frame("status_active"), frame("version")])
    assert ret_sp.brakeLampRelay == structs.CarStateSP.BrakeLampRelay()

  def test_absent_board_reads_as_nothing(self):
    run = Run()
    for _ in range(5):
      _, ret_sp = run.step()
    assert ret_sp.brakeLampRelay == structs.CarStateSP.BrakeLampRelay()

  def test_status_decodes_and_is_valid_for_one_second(self):
    sig = G["status_active"]["signals"]
    run = Run()
    _, ret_sp = run.step([frame("status_active")])
    b = ret_sp.brakeLampRelay
    assert b.valid
    assert (b.state, b.outDrive, b.wouldOn, b.request, b.cmdFresh) == \
      (sig["STATE"], bool(sig["OUT_DRIVE"]), bool(sig["WOULD_ON"]), bool(sig["REQUEST"]), bool(sig["CMD_FRESH"]))
    assert b.faults == sum(sig[n] << i for i, n in enumerate(BLR_FAULT_SIGNALS)) == 0xA0
    assert b.latchReason == sig["LATCH_REASON"]
    assert (b.driverBraking, b.extSink) == (bool(sig["DRIVER_BRAKING"]), bool(sig["EXT_SINK"]))

    # 99 more frames (0.99 s) still valid; the 100th (1.00 s) is not, and everything reads zero with it
    for _ in range(BLR_STATUS_STALE_FRAMES - 1):
      _, ret_sp = run.step()
      assert ret_sp.brakeLampRelay.valid
    _, ret_sp = run.step()
    assert ret_sp.brakeLampRelay == structs.CarStateSP.BrakeLampRelay(), "a stale status must not be the board's last word"

    # and a new frame is valid again, with its own contents
    sig = G["status_latched"]["signals"]
    _, ret_sp = run.step([frame("status_latched")])
    b = ret_sp.brakeLampRelay
    assert b.valid and b.state == 4 and b.latchReason == sig["LATCH_REASON"] and b.driverBraking
    assert b.faults == sum(sig[n] << i for i, n in enumerate(BLR_FAULT_SIGNALS))

  def test_analog_is_its_own_window(self):
    sig = G["analog"]["signals"]
    run = Run()
    _, ret_sp = run.step([frame("analog")])
    b = ret_sp.brakeLampRelay
    assert (b.relaySnsMv, b.lampSnsMv, b.vbatSnsMv) == (sig["RELAY_SNS_MV"], sig["LAMP_SNS_MV"], sig["VBAT_SNS_MV"])
    assert not b.valid, "an analog frame is not a status"
    for _ in range(BLR_ANALOG_STALE_FRAMES):
      _, ret_sp = run.step()
    assert ret_sp.brakeLampRelay.vbatSnsMv == 0

  def test_identity_latches(self):
    """1 Hz identity: silence means nothing changed, as GW_VERSION / GW_BUILD."""
    v, bd = G["version"]["signals"], G["build"]["signals"]
    run = Run()
    _, ret_sp = run.step([frame("version")])
    b = ret_sp.brakeLampRelay
    assert b.fwValid and not b.buildValid, "0x62F has not arrived: buildValid must say so"
    assert (b.fwGitHash, b.fwDirty, b.variant) == (v["GIT_HASH"], bool(v["DIRTY"]), v["VARIANT"])
    assert f"{b.fwGitHash:08x}" == "883a3cd1"
    run.step([frame("build")])
    for _ in range(1000):
      _, ret_sp = run.step()
    b = ret_sp.brakeLampRelay
    assert b.fwValid and b.buildValid and not b.valid
    assert (b.bootPresent, b.bootGitLo24, b.boardUid24, b.optBorEn, b.optDualBank) == \
      (bool(bd["BOOT_PRESENT"]), bd["BOOT_GIT_LO24"], bd["BOARD_UID24"], bool(bd["OPT_BOR_EN"]), bool(bd["OPT_DUAL_BANK"]))

  def test_a_decoder_that_raises_leaves_defaults_and_never_raises(self):
    """An exception in CarState.update() stops card, and with it openpilot's 0x1FA."""
    run = Run()
    with mock.patch.object(carstate_ext, "BLR_FAULT_SIGNALS", (*BLR_FAULT_SIGNALS, "NOT_A_SIGNAL")):
      _, ret_sp = run.step([frame("status_active"), frame("version")])
    assert ret_sp.brakeLampRelay == structs.CarStateSP.BrakeLampRelay()
    _, ret_sp = run.step([frame("status_active")])
    assert ret_sp.brakeLampRelay.valid and ret_sp.brakeLampRelay.fwValid


class TestBlrNeverFeedsBraking(unittest.TestCase):
  """DESIGN E.7. Masking or feeding the pedal from the lamp board would hide a real press, or invent one, at the
  worst moment - so nothing the board says may reach CarState."""

  def test_carstate_is_identical_with_and_without_the_board(self):
    packer = CANPacker(CAR_DBC)
    with_board, without = Run(), Run()
    blr_cycle = [frame("status_active"), frame("status_latched"), frame("analog"), frame("health"), frame("counters"),
                 frame("counters2"), frame("version"), frame("build")]
    # foot off, then BRAKE_PRESSED, then BRAKE_SWITCH alone (two samples in a row count), then off again
    phases = [(0, 0)] * 60 + [(1, 0)] * 60 + [(0, 1)] * 60 + [(0, 0)] * 60
    pressed, board_said = [], set()
    for n, (bp, bs) in enumerate(phases):
      car = [packer.make_can_msg("POWERTRAIN_DATA", 0, {"BRAKE_PRESSED": bp, "BRAKE_SWITCH": bs})]
      blr = [blr_cycle[(n // 10) % len(blr_cycle)]] if n % 10 == 0 else []
      ret_a, sp_a = with_board.step(car + blr)
      ret_b, sp_b = without.step(car)
      # CarState is a capnp builder here: compare the encoded message, every field at once, NaN-safe
      if ret_a.to_bytes() != ret_b.to_bytes():
        self.fail(f"frame {n}: a BLR frame changed CarState: {ret_a.to_dict()} != {ret_b.to_dict()}")
      b = sp_a.brakeLampRelay
      board_said |= {k for k in ("valid", "driverBraking", "extSink", "request", "outDrive", "fwValid") if getattr(b, k)}
      sp_a.brakeLampRelay = structs.CarStateSP.BrakeLampRelay()
      self.assertEqual(sp_a, sp_b, f"frame {n}: a BLR frame changed carStateSP outside brakeLampRelay")
      pressed.append(ret_a.brakePressed)
    # the comparison means something only if the board was heard saying it saw braking, and brakePressed moved -
    # with 0x17C alone
    assert board_said == {"valid", "driverBraking", "extSink", "request", "outDrive", "fwValid"}, board_said
    assert not any(pressed[:60]) and all(pressed[60:120]) and all(pressed[121:180]) and not any(pressed[180:]), pressed

  def test_only_the_decoder_names_the_board(self):
    """Source-level: the board's names appear in the parser registration, the decoder and the struct, and nowhere
    else in opendbc - and in particular not in panda safety, carcontroller or any brake path."""
    allowed = {
      "car/structs.py", "car/honda/carstate.py", "sunnypilot/car/honda/carstate_ext.py",
      "sunnypilot/car/honda/test_blr.py", "dbc/generator/honda/_sunnypilot_blr.dbc",
      # FORK(BRAKE-SWITCH-MIRROR): the one deliberate exception to E.7 (owner decision 2026-10-29), argued in the file
      "sunnypilot/car/honda/brake_switch_mirror.py", "sunnypilot/car/honda/test_brake_switch_mirror.py",
    }
    pattern = re.compile(r"brakeLampRelay|BrakeLampRelay|_update_blr|\bBLR_[A-Z]")
    found = set()
    for path in OPENDBC.rglob("*"):
      if path.suffix not in (".py", ".h", ".c", ".cc", ".dbc", ".capnp") or "_generated" in path.name:
        continue
      if pattern.search(path.read_text(encoding="utf-8", errors="replace")):
        found.add(path.relative_to(OPENDBC).as_posix())
    assert found <= allowed, f"the brake-lamp board is named in {sorted(found - allowed)}: review it against DESIGN E.7"
    assert {"car/honda/carstate.py", "sunnypilot/car/honda/carstate_ext.py"} <= found

    # panda safety: not one of the board's IDs, in hex or decimal, nor its name
    ids = re.compile(r"0x62[0-9a-fA-F]\b|\b(?:15(?:6[89]|7\d|8[0-3]))\b|\bBLR\b")
    for path in (OPENDBC / "safety").rglob("*"):
      if path.suffix in (".h", ".c", ".py"):
        m = ids.search(path.read_text(encoding="utf-8", errors="replace"))
        assert m is None, f"{path.relative_to(OPENDBC)}: {m.group(0)!r} - panda safety must not know the lamp board"

  def test_the_decoder_is_never_handed_carstate(self):
    src = (HERE / "carstate_ext.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_update_blr")
    assert [a.arg for a in fn.args.args] == ["self", "ret_sp", "cp"], "_update_blr must never be handed ret (CarState)"
    assert not any(isinstance(n, ast.Name) and n.id == "ret" for n in ast.walk(fn))
    # every assignment it makes lands in ret_sp.brakeLampRelay or in its own self._blr_* bookkeeping
    for n in ast.walk(fn):
      targets = n.targets if isinstance(n, ast.Assign) else [n.target] if isinstance(n, ast.AugAssign | ast.AnnAssign) else []
      for t in targets:
        if isinstance(t, ast.Attribute):
          root = ast.unparse(t)
          assert root.startswith(("blr.", "ret_sp.brakeLampRelay", "self._blr_")), f"_update_blr assigns {root}"
    # called with exactly that, and only from CarStateExt.update()
    assert src.count("self._update_blr(") == 1 and "self._update_blr(ret_sp, cp)" in src
    # brakeLampRelay is touched nowhere else in the file
    outside = src.replace(ast.get_source_segment(src, fn) or "", "")
    code = "\n".join(line.split("#", 1)[0] for line in outside.splitlines())
    assert "brakeLampRelay" not in code

    # carstate.py names the board's frames only to register them
    cs = (OPENDBC / "car/honda/carstate.py").read_text(encoding="utf-8")
    tree = ast.parse(cs)
    reg = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "get_can_parsers")
    in_reg = {n.value for n in ast.walk(reg) if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.startswith("BLR_")}
    everywhere = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.startswith("BLR_")}
    assert in_reg == everywhere == set(TELEMETRY.values()), (in_reg, everywhere)


if __name__ == "__main__":
  unittest.main()
