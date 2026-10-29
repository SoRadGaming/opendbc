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
  # FORK(HONDA_ACCORD_9G_AU): HondaElesysPumpC1b, set by _initialize_honda with openpilot long.
  # 16 is RESERVED, never set: the retired brake pump rule C1 (batch 3, HondaElesysPumpV6, 2026-10-05 to 2026-10-06).
  # Routes that ran C1 carry it, and the route tools read it as C1; the controller ignores it (runs v5). Do not reuse.
  ELESYS_PUMP_V6 = 16
  # 32 is RESERVED, never set: the retired brake law v2 (batch 3, HondaElesysBrakeLawV2, 2026-10-05 to 2026-10-09, off
  # by default and never accepted). Routes with it set carry it, and brake_route_check reads it; the controller ignores
  # it (today's brake law runs). Do not reuse.
  ELESYS_BRAKE_LAW_V2 = 32
  ELESYS_PUMP_C1B = 64  # the brake pump rule C1b (carcontroller.brake_pump_c1b_elesys); clear = v5
  # FORK(BRAKE-SWITCH-MIRROR): no bit here. The 0x17C mirror is on for this variant in software, with no toggle, and
  # the CarController reads it off CarParams' safety param - HondaSafetyFlags.ELESYS_BRAKE_SWITCH_MIRROR (256), set
  # by honda/interface.py - exactly as brake320's ceiling is read. 128 is free.


class HondaSafetyFlagsSP:
  NIDEC_HYBRID = 1
  GAS_INTERCEPTOR = 2
