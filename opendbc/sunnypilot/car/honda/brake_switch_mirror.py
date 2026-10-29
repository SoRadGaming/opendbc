"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

FORK(BRAKE-SWITCH-MIRROR): the one gate of the 0x17C BRAKE_SWITCH mirror.

THE PROBLEM. openpilot brakes through BRAKE_COMMAND (0x1FA) and the BLR-v1 board lights the brake lamps off that
frame's BRAKE_LIGHTS bit (FORK.md section D) - the car's own brake pedal was never touched, so POWERTRAIN_DATA
(0x17C) still says BRAKE_SWITCH = 0 the whole time. The Elesys radar on bus 2 therefore sees lamps lit with no
brake switch behind them, which is the pair it objects to. The mirror (hondacan.create_brake_switch_on, sent from
CarController) re-sends 0x17C to the radar with every signal copied and BRAKE_SWITCH forced on while the lamps
really are lit.

ON FOR THIS VARIANT IN SOFTWARE, with no toggle, like the stand-down and brake320 (owner decision 2026-10-29):
honda/interface.py sets the panda's bit for HONDA_ELESYS with openpilot longitudinal, stock ACC mode clears it, and
brake_switch_mirror_on() below reads it back off the same param so the CarController and the panda can never disagree.

THE GATE ANSWERS ONE QUESTION: is the board driving the lamps RIGHT NOW? Not "did openpilot ask for lamps" - the
owner's rule is that the bit may only claim what is physically true, so the claim is keyed on the board's own
report of its output, which already folds in the board's freshness, its latches and its lamp sense.

DELIBERATE EXCEPTION TO BLR DESIGN E.7. The board's rule is diagnostics only: nothing it sends may feed
brakePressed, brakeLights, a controls decision or panda safety, and test_blr.py pins that in the source by listing
the files allowed to name it. This module is on that list on purpose (owner decision, 2026-10-29) and is the ONLY
bridge from the board into anything openpilot transmits. What it does and does not reach:

  * it reaches ONE BIT of ONE frame openpilot sends to the radar, and nothing else;
  * it does NOT reach brakePressed, brakeLights, any CarState field, the longitudinal plan, the brake command or
    panda safety - panda still does not know the board's IDs exist (test_blr.py checks that separately);
  * it is READ-ONLY and fail-safe: anything missing, stale, latched or dark answers False, and the mirror then
    copies the car's own BRAKE_SWITCH through untouched, i.e. the radar sees exactly today's wire.

So the worst a lying or dead board can do is leave the frame as the car sent it, or claim a brake switch the car's
own brake lamps are already showing. It can never take braking away or add any.
"""

from opendbc.car import structs
from opendbc.car.honda.values import HondaSafetyFlags
from opendbc.sunnypilot.car.honda.carstate_ext import BLR_FAULT_SIGNALS

SafetyModel = structs.CarParams.SafetyModel


def _mirror_of(cfg) -> bool:
  """honda.h's honda_nidec_init() rule for one safety config."""
  if cfg.safetyModel != SafetyModel.hondaNidec:
    return False
  p = int(cfg.safetyParam)
  f = HondaSafetyFlags
  return (bool(p & f.ELESYS_BRAKE_SWITCH_MIRROR.value) and bool(p & f.ELESYS_SCM_STANDDOWN.value) and
          not (p & f.ELESYS_STOCK_ACC.value))


def brake_switch_mirror_on(CP) -> bool:
  """Is the panda mirroring 0x17C for this CarParams: hondaNidec with ELESYS_BRAKE_SWITCH_MIRROR and
  ELESYS_SCM_STANDDOWN and without ELESYS_STOCK_ACC (honda.h), i.e. this car with openpilot longitudinal. The
  CarController asks the same question the panda does, off the same param - the shape of brake_ceiling's
  panda_max_brake(). With more than one safety config, every one of them. Unreadable is off."""
  try:
    cfgs = list(CP.safetyConfigs)
    return all(_mirror_of(c) for c in cfgs) if cfgs else False
  except Exception:
    return False

# 0x620's own diagnoses that mean the lamps are NOT lit however OUT_DRIVE reads: the lamp sense is dark while we
# drive (D_LAMP_DARK), and the relay coil is open so nothing downstream can be (D_COIL_OPEN). Bit positions come
# from BLR_FAULT_SIGNALS so the board's byte order stays defined in one place (carstate_ext.py).
NO_LIGHT_FAULTS = ("D_LAMP_DARK", "D_COIL_OPEN")
NO_LIGHT_MASK = sum(1 << BLR_FAULT_SIGNALS.index(name) for name in NO_LIGHT_FAULTS)


def brake_lamp_driven(CS) -> bool:
  """True only while the BLR-v1 board reports it is driving lit brake lamps.

  BLR_STATUS (0x620) is 10 Hz plus one on every change and carries a 1 s freshness of its own (blr.valid), so this
  trails the lamps by a frame or two at an edge and never by more than that.

  Fail-safe to False, and it must never raise inside CarController.update(): no board, a stale board, a latched
  board or a dark lamp all mean "do not claim the brake switch".
  """
  try:
    blr = getattr(getattr(CS, "out_sp", None), "brakeLampRelay", None)
    if blr is None or not getattr(blr, "valid", False):
      return False
    if not getattr(blr, "outDrive", False):
      return False
    return not (int(getattr(blr, "faults", 0) or 0) & NO_LIGHT_MASK)
  except Exception:
    return False
