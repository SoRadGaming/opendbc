"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import numpy as np

from opendbc.car import structs
from opendbc.car.can_definitions import CanData
from opendbc.car.honda.values import HONDA_ELESYS  # FORK(HONDA_ELESYS)
from opendbc.sunnypilot.car import create_gas_interceptor_command
# FORK(HONDA_ELESYS): this car's gas law lives in elesys_gas.py. The v1 names are re-exported here
# because test_elesys.py and the docs have always found them in this module.
from opendbc.sunnypilot.car.honda.elesys_gas import ELESYS_GAS_BP, ELESYS_GAS_V, ElesysGasLaw, elesys_gas_multiplier  # noqa: F401


class GasInterceptorCarController:
  def __init__(self, CP: structs.CarParams, CP_SP: structs.CarParamsSP):
    self.CP = CP
    self.CP_SP = CP_SP

    self.gas = 0.
    self.interceptor_gas_cmd = 0.
    # FORK(HONDA_ELESYS): reads HondaElesysGasLawV2 here, once, when CarController is built (ignition)
    self.elesys_gas = ElesysGasLaw() if (CP.carFingerprint in HONDA_ELESYS and CP_SP.enableGasInterceptor) else None

  def update(self, CC: structs.CarControl, CS: structs.CarState, gas: float, brake: float, wind_brake: float,
             packer, frame: int, tuner=None) -> list[CanData]:
    can_sends = []

    if self.CP_SP.enableGasInterceptor:
      # way too aggressive at low speed without this
      gas_mult = np.interp(CS.out.vEgo, [0., 10.], [0.4, 1.0])
      # send exactly zero if apply_gas is zero. Interceptor will send the max between read value and apply_gas.
      # This prevents unexpected pedal range rescaling
      # Sending non-zero gas when OP is not enabled will cause the PCM not to respond to throttle as expected
      # when you do enable.
      if self.elesys_gas is not None:
        # FORK(HONDA_ELESYS): this car's own law, v1 or v2 (elesys_gas.py). 0.0 when not longActive; never raises.
        self.gas = self.elesys_gas.update(CC, CS, gas, brake, wind_brake)
      elif CC.longActive:
        self.gas = float(np.clip(gas_mult * (gas - brake + wind_brake * 3 / 4), 0., 1.))
      else:
        self.gas = 0.0
      can_sends.append(create_gas_interceptor_command(packer, self.gas, frame // 2))

      # FORK: the dynamic tuner no longer learns a pedal gain; it only counts per-drive-mode data
      # (dynamic_tuning.py, observe_pedal). Never raises.
      if tuner is not None:
        tuner.observe_pedal(CC, CS, self.gas, self.elesys_gas.law if self.elesys_gas is not None else "nidec")

    return can_sends
