"""
FORK(BRAKE-SWITCH-MIRROR): POWERTRAIN_DATA (0x17C) to the Elesys radar with BRAKE_SWITCH on while the lamps are lit.

openpilot brakes through 0x1FA and the BLR-v1 board lights the brake lamps off that frame, but the pedal never moved,
so the car's own 0x17C still says BRAKE_SWITCH = 0 - and the radar objects to lit lamps with no brake switch behind
them. CarController re-sends 0x17C to the radar (bus 2) through hondacan.create_brake_switch_on(): every signal copied,
BRAKE_SWITCH forced on while brake_lamps_lit().

It goes out every frame, not only while the lamps are on. The radar checks 0x17C's COUNTER, so the panda blocks the
car's own copy to bus 2 for the whole drive and openpilot is the only source (safety/modes/honda.h) - a gap in this
stream is a gap at the radar, the same bargain the 0x1A6 stand-down makes. Nothing is sent until the car's own 0x17C
has been decoded (CarState.powertrain_data is None until then), since a copy of nothing is all zeros: engine-off.

On for HONDA_ELESYS with openpilot longitudinal, with no toggle, like the stand-down and brake320. honda/interface.py
sets the safety param bit, stock ACC mode clears it (sunnypilot/car/interfaces.py), and brake_switch_mirror_on() reads
it back, so the CarController and the panda cannot disagree about who owns the radar's 0x17C.

brake_lamps_lit() is a deliberate exception to the lamp board's DESIGN E.7, diagnostics only (owner decision
2026-10-29), and the only place the board reaches anything openpilot sends. It decides this one bit and nothing else,
and it is fail-safe: anything missing, stale or faulted is "not lit", and the car's own bit goes through unchanged.
"""
from opendbc.car import structs
from opendbc.car.honda.values import HondaSafetyFlags
from opendbc.sunnypilot.car.honda.carstate_ext import BLR_FAULT_SIGNALS

# the board's own diagnoses that the lamps are dark, however OUT_DRIVE reads
NO_LIGHT_MASK = sum(1 << BLR_FAULT_SIGNALS.index(name) for name in ("D_LAMP_DARK", "D_COIL_OPEN"))


def brake_switch_mirror_on(CP) -> bool:
  """honda_nidec_init()'s rule, on every safety config: hondaNidec, the mirror and stand-down bits, not stock ACC."""
  f = HondaSafetyFlags
  try:
    cfgs = list(CP.safetyConfigs)
    return bool(cfgs) and all(c.safetyModel == structs.CarParams.SafetyModel.hondaNidec and
                              bool(c.safetyParam & f.ELESYS_BRAKE_SWITCH_MIRROR) and
                              bool(c.safetyParam & f.ELESYS_SCM_STANDDOWN) and
                              not (c.safetyParam & f.ELESYS_STOCK_ACC) for c in cfgs)
  except Exception:
    return False


def brake_lamps_lit(CS) -> bool:
  """The board reports a fresh status, its output driven, and no dark-lamp diagnosis. Never raises."""
  try:
    blr = CS.out_sp.brakeLampRelay
    return bool(blr.valid and blr.outDrive and not (int(blr.faults) & NO_LIGHT_MASK))
  except Exception:
    return False
