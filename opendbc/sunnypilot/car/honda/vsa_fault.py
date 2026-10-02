"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

FORK(HONDA_ACCORD_9G_AU): the VSA's own fault, read off the car's CAN.

On 2026-10-01 this car's VSA (the ABS / stability-control modulator, the unit that carries out
openpilot's brake requests) declared an internal fault twice. Honda i-HDS later read DTC 32-11,
"ABS solenoid valve malfunction", with a freeze frame matching the second onset. openpilot
only ever saw BRAKE_ERROR on 0x1B0, so the driver read "Cruise Fault: Restart the Car" - which
is the wrong advice: the VSA keeps the fault across a key cycle and only clears it once the
car has been driven above about 35 km/h. Forensics: S:/OP/incident-2026-10-01/REPORT.md.

This module turns the VSA's own fault bits into two booleans for carStateSP:

  vsaFault        the VSA is faulting NOW (a live onset, the fault that drops a brake request)
  vsaStoredFault  the VSA's fault lamps are on outside the start-up bulb check - a fault stored
                  from an earlier key cycle, or the lamp state of a live one

THE BITS ARE PROVISIONAL. Nobody has a Honda DBC for them; they are named from timing alone,
across the five routes of 10-01 (110-113 and the clean 10f) and checked against every route on
disk (166 routes, 43.4 h). Notation 0xADDR bB.k = byte B, bit k, k=7 the MSB, which is also the
DBC start bit 8*B+k. The DBC signals are VSA_FAULT_* in _honda_elesys_base.dbc.

  live    0x1A4 b2.2, b2.3   0x00 -> 0x0C in the onset frame of both episodes (1923.165, 128.849);
                             never set on any other route, at any time
          0x1EA b6.2         the VSA flags its own inertial data invalid in the same frame. Also
                             set throughout a stored fault, so it counts as "live" only together
                             with BRAKE_ERROR (accFaulted), which a stored fault does not carry.
                             It is here for timing: 0x1A4 arrives one panda batch after 0x1B0 on
                             about 4% of cycles, 0x1A4 and 0x1EA together never (0 of 19,198 on
                             routes 10f-113), so with it vsaFault is already true on the frame
                             accFaulted first is - which is what lets selfdrived name the VSA in
                             the disengagement alert, created on that one frame
  stored  0x1A4 b3.3, b3.6, b3.7, b6.0   the lamp bits, +20 ms after a live onset, and from 2.26 s
                             after key-on on a stored start; b3.6 and b3.7 are also part of the
                             start-up bulb check (clean starts: b3.4-b3.7 until at most 3.08 s
                             after the first 0x1A4 frame, over 143 starts) - hence the window
          0x1A4 b4.0         set only on a stored start (0x01 from the second frame, 0.26 s), never
                             on a live onset; cleared with the stored fault (113 t=36.877)
  NOT USED  0x1A4 b3.5, b4.1 they are lamp bits of this fault too, but they also sit on, together,
                             for minutes at a time on 45 earlier routes (comma_logs 00-87, 4.6 h
                             in all) in which the VSA acknowledged openpilot's brake requests
                             thousands of times (COMPUTER_BRAKING on 12,172 frames of that state on
                             route 69 alone) - a different lamp state, not a brake that does not work
            0x1AA b2.0, b2.1 live-onset bits as well, but b2.1 alone flickers on three earlier
                             routes; and 0x1A4 + 0x1EA already cover the onset frame
            0x3D9 b1.0, b1.2 a 5 Hz echo of the lamp state, 140 ms late

This monitor, replayed over all 166 routes at card's 100 Hz from each route's first VSA frame, sets
a flag on 110-113 only: vsaFault from each onset frame, vsaStoredFault 5.5 s after key-on on 111
and 113 (until 113's clear) and 0.5 s after each live onset's lamps.

NEVER RAISES. CarStateExt.update() calls this every frame from CarState.update(); an exception
there stops card, and with card stops openpilot's 0x1FA - the VSA sets BRAKE_ERROR about a
second later. Anything unexpected reads as no fault.
"""
import math

# VSA_STATUS (0x1A4)
LIVE_SIGNALS = ("VSA_FAULT_LIVE_B2_2", "VSA_FAULT_LIVE_B2_3")
LAMP_SIGNALS = ("VSA_FAULT_LAMP_B3_3", "VSA_FAULT_LAMP_B3_6", "VSA_FAULT_LAMP_B3_7", "VSA_FAULT_LAMP_B6_0",
                "VSA_FAULT_STORED_B4_0")
# VEHICLE_DYNAMICS (0x1EA)
INERTIAL_INVALID = "VSA_FAULT_INERTIAL_INVALID"

# Frames are CarState.update() calls, 100 Hz: card runs once per panda batch.
#
# The start-up bulb check lights b3.4-b3.7 for up to 3.08 s after the VSA's first frame (143 clean
# starts). Lamp bits are not counted until STARTUP_WINDOW_FRAMES after the first 0x1A4 frame this
# CarState has seen, or after the VSA's frames resume from a silence. card starts some seconds
# after key-on, so on the car the margin is larger than the 1.9 s it is here.
STARTUP_WINDOW_FRAMES = 500   # 5.0 s
STORED_SET_FRAMES = 50        # 0.5 s of lamp bits before vsaStoredFault goes True
STORED_CLEAR_FRAMES = 50      # 0.5 s without them before it goes False again
# 0x1A4 and 0x1EA are 50 Hz. Half a second without a new frame is a missing message: both flags
# read False, and the bulb-check window starts again with the next frame (a quick key cycle with
# openpilot still running restarts the VSA, and its bulb check, under the same CarState).
VSA_SILENT_FRAMES = 50


def _bit(sigs, name: str) -> bool:
  """One provisional bit out of a CANParser signal dict. Anything but a finite non-zero number is False."""
  try:
    v = sigs[name]
  except (KeyError, TypeError, IndexError):
    return False
  try:
    v = float(v)
  except (TypeError, ValueError):
    return False
  return math.isfinite(v) and v != 0.0


class VsaFaultMonitor:
  def __init__(self):
    self.vsa_fault = False
    self.vsa_stored_fault = False
    self._vsa_ts = 0
    self._vsa_silent = VSA_SILENT_FRAMES   # nothing seen yet
    self._dyn_ts = 0
    self._dyn_silent = VSA_SILENT_FRAMES
    self._session = 0                      # frames since the VSA's frames (re)started
    self._lamp_on = 0                      # consecutive counted frames with a lamp bit
    self._lamp_off = 0                     # consecutive frames without one

  def _reset_stored(self) -> None:
    self.vsa_stored_fault = False
    self._session = 0
    self._lamp_on = 0
    self._lamp_off = 0

  def update(self, vsa_ts: int, vsa, dyn_ts: int, dyn, brake_error: bool) -> tuple[bool, bool]:
    """One CarState frame.

    vsa_ts / dyn_ts: the CANParser ts_nanos of VSA_STATUS / VEHICLE_DYNAMICS (0 = never received).
    vsa / dyn: their signal dicts (cp.vl[...]). brake_error: carState.accFaulted, i.e. 0x1B0
    BRAKE_ERROR_1 or BRAKE_ERROR_2 on this car. Returns (vsa_fault, vsa_stored_fault).
    """
    # freshness, counted in frames because CarState.update() is not handed a clock
    if vsa_ts and vsa_ts != self._vsa_ts:
      self._vsa_ts = vsa_ts
      self._vsa_silent = 0
    else:
      self._vsa_silent = min(self._vsa_silent + 1, VSA_SILENT_FRAMES)
    if dyn_ts and dyn_ts != self._dyn_ts:
      self._dyn_ts = dyn_ts
      self._dyn_silent = 0
    else:
      self._dyn_silent = min(self._dyn_silent + 1, VSA_SILENT_FRAMES)

    if self._vsa_silent >= VSA_SILENT_FRAMES:
      self.vsa_fault = False
      self._reset_stored()
      return False, False

    self._session = min(self._session + 1, STARTUP_WINDOW_FRAMES)
    settled = self._session >= STARTUP_WINDOW_FRAMES

    # 0x1EA b6.2 with BRAKE_ERROR counts only once the start-up window is over: at key-on BRAKE_ERROR is up on
    # the VSA's first frame and b6.2 (on a stored start) from its second, and a batch boundary between them
    # would otherwise read as a one-frame live fault. The 0x1A4 bits need no window - they are never set at start-up.
    inertial_invalid = settled and self._dyn_silent < VSA_SILENT_FRAMES and _bit(dyn, INERTIAL_INVALID)
    self.vsa_fault = any(_bit(vsa, s) for s in LIVE_SIGNALS) or (inertial_invalid and bool(brake_error))

    lamps = settled and any(_bit(vsa, s) for s in LAMP_SIGNALS)
    if lamps:
      self._lamp_on = min(self._lamp_on + 1, STORED_SET_FRAMES)
      self._lamp_off = 0
      if self._lamp_on >= STORED_SET_FRAMES:
        self.vsa_stored_fault = True
    else:
      self._lamp_off = min(self._lamp_off + 1, STORED_CLEAR_FRAMES)
      self._lamp_on = 0
      if self._lamp_off >= STORED_CLEAR_FRAMES:
        self.vsa_stored_fault = False

    return self.vsa_fault, self.vsa_stored_fault
