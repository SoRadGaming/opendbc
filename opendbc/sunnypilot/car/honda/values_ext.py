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
  # FORK(HONDA_ACCORD_9G_AU): HondaElesysBrakeLawV2 / HondaElesysPumpC1b, set by _initialize_honda with openpilot long.
  # 16 is RESERVED, never set: the retired brake pump rule C1 (batch 3, HondaElesysPumpV6, 2026-10-05 to 2026-10-06).
  # Routes that ran C1 carry it, and the route tools read it as C1; the controller ignores it (runs v5). Do not reuse.
  ELESYS_PUMP_V6 = 16
  ELESYS_BRAKE_LAW_V2 = 32  # the measured brake law; clear = today's law
  ELESYS_PUMP_C1B = 64  # the brake pump rule C1b (carcontroller.brake_pump_c1b_elesys); clear = v5


class HondaSafetyFlagsSP:
  NIDEC_HYBRID = 1
  GAS_INTERCEPTOR = 2
