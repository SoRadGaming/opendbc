"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

FORK(HONDA_ACCORD_9G_AU): brake320 - this car's brake ceiling is 320 counts, not every Honda's 255 (owner decision
2026-10-24). Everything that decides it is here; carcontroller.py, carstate_ext.py and card call in.

THE MEASUREMENT (push13 brakecap study, today's car, level road, July-October 2026; steady deceleration at a held
COMPUTER_BRAKE): 100 counts -1.1 m/s^2, 150 -1.76, 200 -2.2, 240-255 -2.45 (95% CI -2.34..-2.62), 320 about -2.95. About
1.0 m/s^2 per 100 counts from 60 to 200, 0.7 above 180; lag about 0.4 s. The Elesys radar's own request ceiling is 320
counts with the pump bit on (its CMBS brakes of 1-7 July sit at 256-320); values above 320 on bus 2 are only its pump-off
release step-downs (1022 -> 682 -> 340 -> 0 and the like, 111 of them), never a request. So 255 -> 320 is about
+0.5 m/s^2, still inside ISO 15622's -3.5; it would have handled route 143 (a stopped car at night, 5% downhill) without
the driver. The VSA logged one 32-11 solenoid fault (2026-10-01, S:/OP/incident-2026-10-01); the owner took the risk.

WHAT CHANGES, AND WHAT DOES NOT
  panda      honda.h max_brake is 320 with HondaSafetyFlags.ELESYS_BRAKE_320 (128) AND the stand-down (32), and not in
             stock ACC mode (64); 255 for any other param, every other Nidec and every Bosch init. interface.py sets 128
             beside 32 on this car with openpilot longitudinal; stock ACC mode clears both (param 68, as before).
  one rule   panda_max_brake(CP) below is honda.h's rule on the same CarParams, and the CarController never sends above
             it: a 0x1FA panda refused would be a gap, and the VSA answers a ~1 s gap with BRAKE_ERROR. Every other car
             gets no ElesysBrakeCeiling at all and runs the old lines (255), frame for frame.
  scale      unchanged: 2.6 m/s^2 per 256 counts (compute_gb_honda_elesys(), CarControllerParams.NIDEC_BRAKE_MAX), which
             is the measured slope. NIDEC_BRAKE_MAX 256 -> 320 would make every request 25% harder, more than the brake
             learner can take back (its floor is 0.85). Instead the brake FRACTION may pass 1.0 on this car, up to
             BRAKE_FRAC_MAX_320 = 1.25 (= 320 / 256): a request beyond -2.6 m/s^2 net asks for more, up to -3.25.
  EARNED     both the fraction's clip and the count's clip are earned by the REQUEST: request_fraction(), the brake
             fraction of what is asked - the plan's accel plus the pitch feedforward, -a / 2.6 - without the creep
             offset (an idle-torque correction below 5 m/s, which on its own took a -1.5 request at a crawl to a
             fraction of 1.01). Up to EARN_START_FRAC (2.45 / 2.6: the old ceiling's measured -2.45) the clips are the
             old 1.0 and 255, so the frame is push12's exactly; from there they rise linearly, the fraction's to 1.25 at
             a full request (earned_fraction()) and the count's to 320 at BRAKE_LAST_MAX_320 (1.24, what
             actuator_hysteresis() leaves of one; earned_ceiling()). The same clip bounds the learned brake gain (up to
             1.6x), which the 255 clip used to: a runaway gain on an ordinary request still stops at 255, as before.
  fallback   any VSA / ABS / brake fault or warning, seen once, takes the ceiling back to 255 for the rest of the drive
             (Brake320Fallback): carStateSP.vsaFault or vsaStoredFault (vsa_fault.py), or - once the VSA's start-up window
             is over - carState.accFaulted (0x1B0 BRAKE_ERROR) or espDisabled (0x1A4 ESP_DISABLED, the VSA lamp; lit by
             the start-up bulb check for up to 1.7 s on almost every route, never later on 201 logged routes). One
             error line (errorLogMessage, in the qlog) and customReserved14 record it.
  telemetry  customReserved14 (card, 1 Hz and on change): the ceiling, the fallback and why, the highest count sent in the
             last second and in the drive; and a `hondabrake` logMessage line for any second that sent above 255.

NEVER RAISES. All of this runs inside CarState.update() and CarController.update(); an exception there stops card, and
with it 0x1FA - BRAKE_ERROR a second later. Anything unexpected reads as 255.
"""
import math

import numpy as np

from opendbc.car import structs
from opendbc.car.carlog import carlog
from opendbc.car.honda.values import HondaSafetyFlags

SafetyModel = structs.CarParams.SafetyModel

BRAKE_SCALE = 256                 # counts per unit of brake fraction: CarControllerParams.NIDEC_BRAKE_MAX, unchanged
BRAKE_MAX_255 = 255               # panda's max_brake on every Nidec Honda, and the CarController's NIDEC_BRAKE_MAX - 1
BRAKE_MAX_320 = 320               # panda's max_brake on this car with ELESYS_BRAKE_320: the Elesys radar's request ceiling
BRAKE_FRAC_MAX_320 = BRAKE_MAX_320 / BRAKE_SCALE   # 1.25: compute_gb_honda_elesys()'s brake clip with brake320
BRAKE_HYST_GAP = 0.01             # actuator_hysteresis()'s brake_hyst_gap: a rising request settles this far below itself
BRAKE_LAST_MAX_320 = BRAKE_FRAC_MAX_320 - BRAKE_HYST_GAP   # 1.24: brake_last at a full request, 317 counts at gain 1.0
OLD_CEILING_DECEL = 2.45          # m/s^2: the steady deceleration measured at 240-255 counts (95% CI 2.34-2.62)
EARN_START_FRAC = OLD_CEILING_DECEL / 2.6   # 0.942: the brake fraction of a -2.45 request (2.6 m/s^2 per 1.0)

WINDOW_FRAMES = 100               # CarController frames (100 Hz) per telemetry window: 1 s
LOG_TAG = "hondabrake"


def _ceiling_of(cfg) -> int:
  """honda.h's honda_nidec_init() rule for one safety config."""
  if cfg.safetyModel != SafetyModel.hondaNidec:
    return BRAKE_MAX_255
  p = int(cfg.safetyParam)
  f = HondaSafetyFlags
  on = bool(p & f.ELESYS_BRAKE_320.value) and bool(p & f.ELESYS_SCM_STANDDOWN.value) and not (p & f.ELESYS_STOCK_ACC.value)
  return BRAKE_MAX_320 if on else BRAKE_MAX_255


def panda_max_brake(CP) -> int:
  """panda's max_brake for this CarParams: 320 on hondaNidec with ELESYS_BRAKE_320 and ELESYS_SCM_STANDDOWN and without
  ELESYS_STOCK_ACC, else 255 (honda.h). With more than one safety config, the lowest of them. Unreadable is 255."""
  try:
    cfgs = list(CP.safetyConfigs)
    return min(_ceiling_of(c) for c in cfgs) if cfgs else BRAKE_MAX_255
  except Exception:
    return BRAKE_MAX_255


def request_fraction(accel) -> float:
  """The brake fraction a request asks for at the brake law's scale: -accel / 2.6, 0 for a positive or non-finite one.
  `accel` is the plan's (CarControl.actuators.accel) plus the pitch feedforward, before the creep offset."""
  try:
    a = float(accel)
  except (TypeError, ValueError):
    return 0.0
  return max(-a, 0.0) / 2.6 if math.isfinite(a) else 0.0


def earned_fraction(req_frac, ceiling: int) -> float:
  """compute_gb_honda_elesys()'s brake clip for a request of req_frac: 1.0 up to EARN_START_FRAC, then linearly up to
  BRAKE_FRAC_MAX_320 at a full request. 1.0 whenever `ceiling` is 255, and for a non-finite req_frac."""
  try:
    f = float(req_frac)
    if ceiling <= BRAKE_MAX_255 or not math.isfinite(f):
      return 1.0
    return float(np.interp(f, [EARN_START_FRAC, BRAKE_FRAC_MAX_320], [1.0, BRAKE_FRAC_MAX_320]))
  except Exception:
    return 1.0


def earned_ceiling(req_frac, ceiling: int) -> int:
  """The highest count a request of req_frac may send: 255 up to EARN_START_FRAC, then linearly up to `ceiling` at
  BRAKE_LAST_MAX_320. 255 whenever `ceiling` is, and for a non-finite req_frac."""
  try:
    f = float(req_frac)
    if ceiling <= BRAKE_MAX_255 or not math.isfinite(f):
      return BRAKE_MAX_255
    return int(np.interp(f, [EARN_START_FRAC, BRAKE_LAST_MAX_320], [BRAKE_MAX_255, ceiling]))
  except Exception:
    return BRAKE_MAX_255


class Brake320Fallback:
  """The VSA fallback, decided in CarState on every frame from card's start (CarStateExt._update_vsa_fault()), so a
  fault before controls start counts too. Latched for the life of this CarState - card's, i.e. the key cycle; a card
  restart starts clean, as a new key cycle does. `settled` is the VSA fault monitor's start-up window (5 s from the
  VSA's first frame, restarted when its frames resume after a silence): accFaulted and espDisabled count only after it,
  because the start-up bulb check lights ESP_DISABLED (and could BRAKE_ERROR) for up to 1.7 s of every key cycle.
  vsaFault and vsaStoredFault already handle the start-up themselves (vsa_fault.py)."""

  def __init__(self):
    self.reason = ""            # "" until latched: the signals that latched it, '+'-joined
    self.frame = 0              # CarState frames seen
    self.latched_frame = 0      # the frame it latched on (1-based), 0 if it has not

  def update(self, acc_faulted, esp_disabled, vsa_fault, vsa_stored_fault, settled, monitor_error=False) -> str:
    try:
      self.frame += 1
      if self.reason:
        return self.reason
      ok = bool(settled)
      hits = [name for name, on in (("vsaFault", bool(vsa_fault)), ("vsaStoredFault", bool(vsa_stored_fault)),
                                    ("accFaulted", ok and bool(acc_faulted)), ("espDisabled", ok and bool(esp_disabled)),
                                    ("vsaMonitorError", bool(monitor_error))) if on]
      if hits:
        self._latch("+".join(hits))
    except Exception:
      if not self.reason:
        self._latch("error")
    return self.reason

  def _latch(self, reason: str) -> None:
    self.reason = reason
    self.latched_frame = self.frame
    try:
      carlog.error(f"{LOG_TAG} fallback={reason} frame={self.frame}: brake ceiling {BRAKE_MAX_320} -> {BRAKE_MAX_255} " +
                   "for the rest of this drive")
    except Exception:
      pass


class ElesysBrakeCeiling:
  """The CarController's side, built only where panda_max_brake(CP) is 320: the ceiling it applies this frame (320, or
  255 once the CarState's Brake320Fallback has latched), the earned clips for this frame's request, and the telemetry
  of what went on the wire."""

  def __init__(self, CP):
    self.allowed = panda_max_brake(CP)
    self.counts = self.allowed          # BRAKE_MAX_255 from the frame the fallback latched
    self.req_frac = 0.0                 # this frame's request_fraction()
    self.fallback_reason = ""
    self.fallback_frame = 0             # the CarState frame it latched on
    self.frames = 0                     # CarController frames
    self.win_max = 0                    # the current 1 s window
    self.win_above = 0
    self.last_max = 0                   # the last complete window
    self.last_above = 0
    self.drive_max = 0
    self.drive_above = 0

  def set_request(self, accel) -> float:
    """This frame's request (the plan's accel plus the pitch feedforward; 0 when longitudinal is not active). Returns
    the brake fraction's clip for compute_gb_honda_elesys() and the brake block."""
    self.req_frac = request_fraction(accel)
    return self.frac_max

  @property
  def frac_max(self) -> float:
    return earned_fraction(self.req_frac, self.counts)

  def update(self, CS) -> None:
    """Once per CarController frame, before the brake law: take the CarState's fallback. Never raises."""
    try:
      fb = getattr(CS, "brake320_fallback", None)
      reason = str(getattr(fb, "reason", "") or "")
      if reason and not self.fallback_reason:
        self.fallback_reason = reason
        self.fallback_frame = int(getattr(fb, "latched_frame", 0) or 0)
        self.counts = BRAKE_MAX_255
    except Exception:
      if not self.fallback_reason:
        self.fallback_reason = "error"
      self.counts = BRAKE_MAX_255

  def earned_counts(self) -> int:
    return earned_ceiling(self.req_frac, self.counts)

  def sent(self, counts) -> None:
    """One 0x1FA, as it went on the wire."""
    try:
      c = int(counts)
    except Exception:
      return
    self.win_max = max(self.win_max, c)
    self.drive_max = max(self.drive_max, c)
    if c > BRAKE_MAX_255:
      self.win_above += 1
      self.drive_above += 1

  def tick(self) -> None:
    """The end of a CarController frame: close the 1 s window, and say so in the route if it sent above 255."""
    try:
      self.frames += 1
      if self.frames % WINDOW_FRAMES:
        return
      self.last_max, self.last_above = self.win_max, self.win_above
      self.win_max = self.win_above = 0
      if self.last_max > BRAKE_MAX_255:
        carlog.info(f"{LOG_TAG} max={self.last_max} above255={self.last_above} ceil={self.counts} " +
                    f"allowed={self.allowed} fallback={self.fallback_reason or '-'}")
    except Exception:
      pass


def brake320_status(CI) -> dict | None:
  """For card's customReserved14: None unless this car's CarState or CarController carries brake320 (every other car,
  and this car in stock ACC mode). Field names are the capnp's. Never raises."""
  try:
    ceil = getattr(getattr(CI, "CC", None), "brake320", None)
    fb = getattr(getattr(CI, "CS", None), "brake320_fallback", None)
    if ceil is None and fb is None:
      return None
    allowed = ceil.allowed if ceil is not None else panda_max_brake(getattr(CI, "CP", None))
    reason = (ceil.fallback_reason if ceil is not None else "") or str(getattr(fb, "reason", "") or "")
    frame = (ceil.fallback_frame if ceil is not None and ceil.fallback_reason else 0) or int(getattr(fb, "latched_frame", 0) or 0)
    return {
      "allowedCounts": int(allowed),
      "ceilingCounts": int(BRAKE_MAX_255 if reason else allowed),
      "vsaFallback": bool(reason),
      "vsaFallbackReason": reason,
      "vsaFallbackFrame": int(frame),
      "maxCountsLastSecond": int(ceil.last_max) if ceil is not None else 0,
      "sendsAbove255LastSecond": int(ceil.last_above) if ceil is not None else 0,
      "maxCountsDrive": int(ceil.drive_max) if ceil is not None else 0,
      "sendsAbove255Drive": int(ceil.drive_above) if ceil is not None else 0,
    }
  except Exception:
    return None
