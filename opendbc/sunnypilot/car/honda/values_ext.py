"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from enum import IntFlag


class HondaFlagsSP(IntFlag):
  NIDEC_HYBRID = 1
  EPS_MODIFIED = 2
  HYBRID_ALT_BRAKEHOLD = 4
  ELESYS_STOCK_ACC = 8  # FORK(HONDA_ACCORD_9G_AU): HondaElesysStockAcc, set by _initialize_honda (sunnypilot/car/interfaces.py)
  # FORK(HONDA_ACCORD_9G_AU): HondaElesysPumpV6 / HondaElesysBrakeLawV2, set by _initialize_honda with openpilot long
  ELESYS_PUMP_V6 = 16  # the brake pump rule C1 (carcontroller.brake_pump_c1_elesys); clear = v5
  ELESYS_BRAKE_LAW_V2 = 32  # the measured brake law; clear = today's law


class HondaSafetyFlagsSP:
  NIDEC_HYBRID = 1
  GAS_INTERCEPTOR = 2
