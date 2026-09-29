"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

FORK(BRAKE-LAMP-TEST): a bench-style test mode for HONDA_ACCORD_9G_AU (HONDA_ELESYS).

THE PROBLEM. When openpilot brakes (BRAKE_COMMAND 0x1FA on bus 0) the stop lamps stay dark,
although stock ACC braking lights them. openpilot already sets BRAKE_LIGHTS (bit 39) on every
braking frame; stock never sets it. On this car the lamp relay line runs to the ACC unit, which
sits on the far side of the harness split and never sees openpilot's 0x1FA. The log mining
(161 routes) found no 0x1FA bit that stock sets on most braking frames and openpilot does not,
so the expected result of this test is NEGATIVE. It is cheap, so it is still worth running
before building hardware.

WHAT IT DOES. With the param HondaBrakeLampTest = N (1..len(CANDIDATES)), entry N's bits are
OR-ed into openpilot's OWN frame (0x1FA, or 0x30C ACC_HUD), and the Honda checksum is
recomputed. Nothing else changes. It applies only while ALL of these hold:
  * the car is HONDA_ELESYS with openpilot longitudinal;
  * openpilot longitudinal is active (CC.longActive);
  * the car is stopped: CS.out.standstill (XMISSION_SPEED == 0) and vEgo < 0.1 m/s;
  * openpilot is commanding the brake (apply_brake > 0, i.e. the standstill hold);
  * the driver is on neither pedal;
  * all of the above have held for DWELL_FRAMES brake frames (1 s) in a row, so nothing is
    set in the last moments of a stop or while the speed flickers around zero;
  * the test has not been used up (see ONE STOP PER SETTING).
It stops on the first brake frame any of those fails. Otherwise every frame is byte-for-byte
what it would have been with the param at 0.

ONE STOP PER SETTING. The param is cleared by manager at the end of every drive (and at boot),
and once the test has been active the car latches it off as soon as vEgo exceeds LATCH_V: a
value left set is applied at ONE stop, not at every traffic light after it. Writing a different
value (the next entry, or off) re-arms it.

WHAT IT DOES NOT TOUCH (the owner's hard constraint). The stock ACC stand-down stays exactly as
it is: SCM_BUTTONS / MAIN_ON, the ELESYS stand-down safety modes, and the ACC unit's CMBS/AEB
path are not touched. No new message is sent and nothing new goes to bus 2. Brake magnitude,
pump, request, cancel, fault, override, the whole CMBS byte (1), the whole AEB byte (3),
BRAKE_LIGHTS, FCW/AEB_STATUS, chime, hybrid, counter and checksum are excluded by ALLOWED_BITS
below, and a unit test enforces that.

HOW TO RUN IT. Stopped at night, openpilot holding the car, foot hovering over the brake: set
the entry from the comma (Settings > vehicle > lamp test) or from sunnylink, watch the lamps in
the mirror, step to the next. Set 0 ("off") the moment anything appears on the cluster, and
restart the car before continuing. The entries marked risky (the tail of the list) are
CMBS-adjacent unknowns; the comma asks for a slide to confirm before stepping into them. The
param is re-read every POLL_FRAMES control frames (0.2 s).

WHAT LANDS IN THE ROUTE. carlog.error (forwarded to cloudlog by card.py, so it reaches
errorLogMessage and therefore the qlog) gets a line tagged LOG_TAG on every entry change, on
every on/off edge, on the latch and re-arm, and with the first modified frame after each
activation or entry change (original and modified bytes). The frames themselves are in sendcan.
"""
from dataclasses import dataclass

from opendbc.car.carlog import carlog
from opendbc.car.honda.hondacan import honda_checksum

PARAM = "HondaBrakeLampTest"
LOG_TAG = "brakelamptest"

POLL_FRAMES = 20        # control frames (100 Hz); the brake block runs on even frames, so 0.2 s
STANDSTILL_V = 0.1      # m/s
DWELL_FRAMES = 50       # brake frames (50 Hz): settled at a standstill hold for 1 s before applying
LATCH_V = 2.0           # m/s: once the test has been active, moving faster than this uses it up

BRAKE_COMMAND = 0x1FA
ACC_HUD = 0x30C


@dataclass(frozen=True)
class LampCandidate:
  label: str              # shown on the comma 4 and in sunnylink: ASCII only, <= 14 chars
  addr: int
  bits: tuple[int, ...]   # DBC bit numbers (byte * 8 + bit, bit 0 = LSB), all set to 1
  note: str
  risky: bool = False     # CMBS-adjacent or may raise a cluster warning; the comma asks to confirm


# Ranked by the log synthesis (2026-09-29) and the review after it. Entry N is CANDIDATES[N - 1].
# The risky entries are a tail at the end of the list (a unit test keeps them there).
CANDIDATES: tuple[LampCandidate, ...] = (
  LampCandidate("1FA bit 23", BRAKE_COMMAND, (23,),
                "SET_ME_X00 = 4, owner v5 ACC_LEAD_FAR. Stock-only, 2.4% of stock braking frames vs 0.2% otherwise."),
  LampCandidate("1FA bit 22", BRAKE_COMMAND, (22,),
                "SET_ME_X00 = 2, owner v5 ACC_LEAD_TRACK. Stock-only; follows the lead car."),
  LampCandidate("1FA bits 21+22", BRAKE_COMMAND, (21, 22), "SET_ME_X00 = 3, in case the field is an enum."),
  LampCandidate("1FA bits 21+23", BRAKE_COMMAND, (21, 23), "SET_ME_X00 = 5."),
  LampCandidate("1FA bits 22+23", BRAKE_COMMAND, (22, 23), "SET_ME_X00 = 6."),
  LampCandidate("1FA bits 21-23", BRAKE_COMMAND, (21, 22, 23), "SET_ME_X00 = 7."),
  LampCandidate("1FA bit 21", BRAKE_COMMAND, (21,),
                "SET_ME_X00 = 1, ACC_OVERRIDE_STOP. Retest of the 2026-07-18 build; skip if that result is known."),
  LampCandidate("30C bit 42", ACC_HUD, (42,),
                "ACC_HUD BOH_4 / HUD brake text. Weak: stock changed it in 2 of 39 braking episodes."),
  # -- risky tail: the comma asks for a slide to confirm before stepping into any of these --
  LampCandidate("30C bit 38", ACC_HUD, (38,),
                "ACC_HUD BRAKE_SYSTEM_ICON. CMBS-adjacent: next to the FCM flags. May light a BRAKE SYSTEM icon.",
                risky=True),
  LampCandidate("1FA bit 44", BRAKE_COMMAND, (44,),
                "SET_ME_X00_3, never set. CMBS-adjacent, unknown: between FCW (42-43) and CHIME.", risky=True),
  LampCandidate("1FA bit 32", BRAKE_COMMAND, (32,),
                "CRUISE_STATES = 1, never set. CMBS-adjacent, unknown: in 0x30C this bit is ENABLE_MINI_CAR.",
                risky=True),
  LampCandidate("1FA bit 33", BRAKE_COMMAND, (33,),
                "CRUISE_STATES = 2. CMBS-adjacent, unknown (0x30C: RADAR_OBSTRUCTED).", risky=True),
  LampCandidate("1FA bit 34", BRAKE_COMMAND, (34,),
                "CRUISE_STATES = 4. CMBS-adjacent, unknown (0x30C: FCM_PROBLEM).", risky=True),
  LampCandidate("1FA bit 35", BRAKE_COMMAND, (35,),
                "CRUISE_STATES = 8. CMBS-adjacent, unknown (0x30C: FCM_OFF).", risky=True),
  LampCandidate("1FA bit 36", BRAKE_COMMAND, (36,),
                "CRUISE_STATES = 16. CMBS-adjacent, unknown (0x30C: FCM_OFF_2).", risky=True),
  LampCandidate("1FA bit 37", BRAKE_COMMAND, (37,),
                "CRUISE_STATES = 32. CMBS-adjacent, unknown (0x30C: ACC_PROBLEM).", risky=True),
  LampCandidate("1FA bit 38", BRAKE_COMMAND, (38,),
                "CRUISE_STATES = 64. CMBS-adjacent, unknown (0x30C: BRAKE_SYSTEM_ICON).", risky=True),
)

# The only bits any entry may set. Everything else in these frames is either already driven by
# openpilot (brake value, pump, request, override, BRAKE_LIGHTS, units), or is excluded on
# purpose:
#   * byte 1 (bits 8-15), the CMBS byte: pump 8, CMBS_BRAKE 10, hybrid pump 11, CMBS_DISABLED 12,
#     and the unnamed 9 and 13 between them;
#   * 17 CRUISE_CANCEL_CMD, 18 CRUISE_FAULT_CMD, 19 SET_ME_X00_2;
#   * byte 3 (bits 24-31), the AEB byte: AEB_REQ_2 24-26, AEB_REQ_3 27, AEB_REQ_1 29, units 31,
#     and the unnamed 28 and 30. Bit 28 is AEB-linked, not a plain brake flag: across all 161
#     routes stock set it in exactly 4 episodes of 1-2 s, and in 3 of them (routes 08, 41, 5e)
#     on every frame together with AEB_REQ_1 -- stock AEB/CMBS interventions, forwarded by the
#     panda as stock AEB on 41 and 5e. The 4th (route 81, 90 km/h) was bit 28 alone. Setting it
#     would hand the VSA half of an AEB request;
#   * 39 BRAKE_LIGHTS (already set), 40-43 AEB_STATUS / FCW, 45-47 CHIME, 48-55 and 62-63 hybrid
#     brake, 56-61 checksum/counter.
# For 0x30C bytes 0-2 are what the panda checks (PCM speed/gas).
ALLOWED_BITS: dict[int, frozenset[int]] = {
  BRAKE_COMMAND: frozenset({21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 44}),
  ACC_HUD: frozenset({38, 42}),
}


def entry_label(entry: int) -> str:
  if 1 <= entry <= len(CANDIDATES):
    return CANDIDATES[entry - 1].label
  return "off"


def set_bits(addr: int, dat: bytes, bits: tuple[int, ...]) -> bytes:
  """OR the given DBC bit numbers into dat and redo the Honda checksum (low nibble of the
  last byte). The counter is left exactly as the packer wrote it."""
  d = bytearray(dat)
  for bit in bits:
    d[bit // 8] |= 1 << (bit % 8)
  d[-1] = (d[-1] & 0xF0) | honda_checksum(addr, None, d)
  return bytes(d)


def _open_params():
  """Params lives in openpilot; import lazily so opendbc still imports standalone."""
  try:
    from openpilot.common.params import Params
    return Params()
  except Exception:
    return None


def _is_applicable(CP) -> bool:
  try:
    from opendbc.car.honda.values import HONDA_ELESYS
    return bool(CP.openpilotLongitudinalControl) and CP.carFingerprint in HONDA_ELESYS
  except Exception:
    return False


class BrakeLampTest:
  def __init__(self, CP, params=None):
    self.applicable = _is_applicable(CP)
    # Other cars never open Params for this, and never modify a frame.
    self._params = params if params is not None else (_open_params() if self.applicable else None)
    self.entry = 0
    self.active = False
    self.latched: int | None = None   # the param value that was used up; None = armed
    self._used = False                # active at least once since the last (re-)arm
    self._dwell = 0                   # consecutive brake frames at a settled standstill hold
    self._pending_log = False
    self.poll(0)

  @property
  def candidate(self) -> LampCandidate | None:
    return CANDIDATES[self.entry - 1] if 1 <= self.entry <= len(CANDIDATES) else None

  def _read_entry(self) -> int:
    if self._params is None:
      return 0
    try:
      raw = self._params.get(PARAM)
    except Exception:
      return 0
    if raw is None:
      return 0
    try:
      entry = int(raw)
    except (TypeError, ValueError):
      return 0
    return entry if 0 <= entry <= len(CANDIDATES) else 0

  def poll(self, frame: int) -> None:
    if not self.applicable or frame % POLL_FRAMES != 0:
      return
    entry = self._read_entry()
    if self.latched is not None and entry != self.latched:
      carlog.error(f"{LOG_TAG} re-armed: param {self.latched}->{entry}")
      self.latched = None
    if entry != self.entry:
      cand = CANDIDATES[entry - 1] if entry > 0 else None
      desc = f"addr=0x{cand.addr:X} bits={list(cand.bits)}" if cand is not None else ""
      carlog.error(f"{LOG_TAG} entry {self.entry}->{entry}/{len(CANDIDATES)} '{entry_label(entry)}' {desc} " +
                   f"active={self.active} latched={self.latched is not None}")
      self.entry = entry
      self._pending_log = True

  def update(self, frame: int, long_active: bool, v_ego: float, standstill: bool, apply_brake: float,
             brake_pressed: bool, gas_pressed: bool) -> bool:
    """Call once per brake frame, before apply(). Returns whether the test is applying."""
    self.poll(frame)
    held = (self.applicable and bool(long_active) and bool(standstill) and v_ego < STANDSTILL_V
            and apply_brake > 0 and not brake_pressed and not gas_pressed)
    self._dwell = self._dwell + 1 if held else 0

    # one stop per setting: once it has been active, driving off uses it up until the param changes
    if self._used and v_ego > LATCH_V:
      self._used = False
      if self.entry > 0:
        self.latched = self.entry
        carlog.error(f"{LOG_TAG} latched off: entry={self.entry} '{entry_label(self.entry)}' used at one stop, " +
                     f"vEgo={v_ego:.2f}; write the param again to re-arm")

    active = self.entry > 0 and self.latched is None and self._dwell >= DWELL_FRAMES
    if active != self.active:
      carlog.error(f"{LOG_TAG} {'ON' if active else 'OFF'} entry={self.entry} '{entry_label(self.entry)}' " +
                   f"vEgo={v_ego:.2f} standstill={bool(standstill)} brake={apply_brake} " +
                   f"brakePressed={bool(brake_pressed)} gasPressed={bool(gas_pressed)} longActive={bool(long_active)} " +
                   f"latched={self.latched is not None}")
      self._pending_log = active
    self._used = self._used or active
    self.active = active
    return active

  def apply(self, msg):
    """Return msg unchanged unless the test is active and msg is the candidate's frame."""
    cand = self.candidate
    if not self.active or cand is None or msg[0] != cand.addr:
      return msg
    addr, dat, bus = msg[0], msg[1], msg[2]
    new = set_bits(addr, dat, cand.bits)
    if self._pending_log:
      carlog.error(f"{LOG_TAG} frame entry={self.entry} 0x{addr:X} {bytes(dat).hex()} -> {new.hex()}")
      self._pending_log = False
    return addr, new, bus
