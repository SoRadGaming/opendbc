"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from enum import StrEnum

from opendbc.car import Bus, structs
from opendbc.car.honda.values import HONDA_ELESYS
from opendbc.can.parser import CANParser
from opendbc.sunnypilot.car.honda.values_ext import HondaFlagsSP

# GW_ACTIVE is low priority on the board and has been observed dropping, so the window is
# 500 ms rather than the 300 ms used for SP_HUD_STATUS in the other direction. carstate runs
# at 100 Hz, so that is 50 frames.
LINBUS_GW_STALE_FRAMES = 50


class CarStateExt:
  def __init__(self, CP, CP_SP):
    self.CP = CP
    self.CP_SP = CP_SP
    self._linbus_gw_stale = LINBUS_GW_STALE_FRAMES
    self._linbus_gw_ts = 0

  def update(self, ret: structs.CarState, ret_sp: structs.CarStateSP,
             can_parsers: dict[StrEnum, CANParser]) -> None:
    cp = can_parsers[Bus.pt]
    cp_cam = can_parsers[Bus.cam]

    if self.CP.carFingerprint in HONDA_ELESYS:
      self._update_linbus_gateway(ret_sp, cp)

    if self.CP_SP.flags & HondaFlagsSP.NIDEC_HYBRID:
      ret.accFaulted = bool(cp.vl["HYBRID_BRAKE_ERROR"]["BRAKE_ERROR_1"] or cp.vl["HYBRID_BRAKE_ERROR"]["BRAKE_ERROR_2"])
      ret.stockAeb = bool(cp_cam.vl["BRAKE_COMMAND"]["AEB_REQ_1"] and cp_cam.vl["BRAKE_COMMAND"]["COMPUTER_BRAKE_HYBRID"] > 1e-5)

    if self.CP_SP.flags & HondaFlagsSP.HYBRID_ALT_BRAKEHOLD:
      ret.brakeHoldActive = cp.vl["BRAKE_HOLD_HYBRID_ALT"]["BRAKE_HOLD_ACTIVE"] == 1

    if self.CP_SP.enableGasInterceptor:
      # Same threshold as panda, equivalent to 1e-5 with previous DBC scaling
      gas = (cp.vl["GAS_SENSOR"]["INTERCEPTOR_GAS"] + cp.vl["GAS_SENSOR"]["INTERCEPTOR_GAS2"]) // 2
      ret.gasPressed = gas > 492

  def _update_linbus_gateway(self, ret_sp: structs.CarStateSP, cp: CANParser) -> None:
    """Decode GW_ACTIVE (0x704) from the aftermarket LIN-bus gateway.

    Staleness is counted in frames rather than against a clock, because CarState.update()
    is not handed a timestamp. ts_nanos only advances when a frame actually arrives, so a
    ts that stops moving is a gateway that stopped talking -- which must read as NOT
    actuating, or openpilot would keep integrating against a car that is no longer
    following it. That is the failure this whole protocol exists to prevent.
    """
    gw = cp.vl["GW_ACTIVE"]          # registered liveness-exempt in get_can_parsers()
    ts = cp.ts_nanos["GW_ACTIVE"]["ENGAGED"]
    if ts != self._linbus_gw_ts:
      self._linbus_gw_ts = ts
      self._linbus_gw_stale = 0
    else:
      self._linbus_gw_stale = min(self._linbus_gw_stale + 1, LINBUS_GW_STALE_FRAMES)

    valid = self._linbus_gw_stale < LINBUS_GW_STALE_FRAMES and ts != 0
    engaged = bool(gw["ENGAGED"])
    dry_run = bool(gw["DRY_RUN"])

    ret_sp.linbusGateway.present = True
    ret_sp.linbusGateway.engaged = engaged
    ret_sp.linbusGateway.dryRun = dry_run
    ret_sp.linbusGateway.valid = valid
    # The one flag consumers read. In a dry run the board still sets ENGAGED -- it reports
    # what it WOULD do -- so ENGAGED alone would tell openpilot it is in control when it is
    # not, and the integrator would wind up exactly as it did before this protocol existed.
    ret_sp.linbusGateway.actuating = engaged and not dry_run and valid
