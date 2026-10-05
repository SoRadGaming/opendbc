"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import json
import os
import numpy as np
from typing import NamedTuple
from collections.abc import Callable

from opendbc.car import structs
from opendbc.car.can_definitions import CanRecvCallable, CanSendCallable
from opendbc.car.carlog import carlog
from opendbc.car.honda.values import HONDA_ELESYS, HondaSafetyFlags  # FORK(HONDA_ACCORD_9G_AU): stock ACC mode
from opendbc.car.hyundai.values import HyundaiFlags
from opendbc.car.subaru.values import SubaruFlags
from opendbc.car.toyota.values import ToyotaSafetyFlags
from opendbc.sunnypilot.car.hyundai.enable_radar_tracks import enable_radar_tracks as hyundai_enable_radar_tracks
from opendbc.sunnypilot.car.hyundai.longitudinal.helpers import LongitudinalTuningType
from opendbc.sunnypilot.car.honda.values_ext import HondaFlagsSP, HondaSafetyFlagsSP  # FORK(HONDA_ACCORD_9G_AU)
from opendbc.sunnypilot.car.hyundai.values import HyundaiFlagsSP
from opendbc.sunnypilot.car.subaru.values_ext import SubaruFlagsSP, SubaruSafetyFlagsSP
from opendbc.sunnypilot.car.tesla.values import MadsScreenButtonType, TeslaFlagsSP, TeslaSafetyFlagsSP
from opendbc.sunnypilot.car.toyota.values import ToyotaFlagsSP


class LatControlInputs(NamedTuple):
  lateral_acceleration: float
  roll_compensation: float
  vego: float
  aego: float


TorqueFromLateralAccelCallbackTypeTorqueSpace = Callable[[LatControlInputs, structs.CarParams.LateralTorqueTuning, bool], float]


class CarInterfaceBaseSP:
  @staticmethod
  def torque_from_lateral_accel_linear_in_torque_space(latcontrol_inputs: LatControlInputs, torque_params: structs.CarParams.LateralTorqueTuning,
                                                        gravity_adjusted: bool) -> float:
    # The default is a linear relationship between torque and lateral acceleration (accounting for road roll and steering friction)
    return latcontrol_inputs.lateral_acceleration / float(torque_params.latAccelFactor)

  def torque_from_lateral_accel_in_torque_space(self) -> TorqueFromLateralAccelCallbackTypeTorqueSpace:
    return self.torque_from_lateral_accel_linear_in_torque_space


class NanoFFModel:
  def __init__(self, weights_loc: str, platform: str):
    self.weights_loc = weights_loc
    self.platform = platform
    self.load_weights(platform)

  def load_weights(self, platform: str):
    with open(self.weights_loc) as fob:
      self.weights = {k: np.array(v) for k, v in json.load(fob)[platform].items()}

  def relu(self, x: np.ndarray):
    return np.maximum(0.0, x)

  def forward(self, x: np.ndarray):
    assert x.ndim == 1
    x = (x - self.weights['input_norm_mat'][:, 0]) / (self.weights['input_norm_mat'][:, 1] - self.weights['input_norm_mat'][:, 0])
    x = self.relu(np.dot(x, self.weights['w_1']) + self.weights['b_1'])
    x = self.relu(np.dot(x, self.weights['w_2']) + self.weights['b_2'])
    x = self.relu(np.dot(x, self.weights['w_3']) + self.weights['b_3'])
    x = np.dot(x, self.weights['w_4']) + self.weights['b_4']
    return x

  def predict(self, x: list[float], do_sample: bool = False):
    x = self.forward(np.array(x))
    if do_sample:
      pred = np.random.laplace(x[0], np.exp(x[1]) / self.weights['temperature'])
    else:
      pred = x[0]
    pred = pred * (self.weights['output_norm_mat'][1] - self.weights['output_norm_mat'][0]) + self.weights['output_norm_mat'][0]
    return pred


def setup_interfaces(CI, CP: structs.CarParams, CP_SP: structs.CarParamsSP,
                     params_list: list[dict[str, str]] | None = None,
                     can_recv: CanRecvCallable | None = None, can_send: CanSendCallable | None = None) -> None:
  if params_list is None:
    params_list = []

  params_dict = {k: v for param in params_list for k, v in param.items()}

  _initialize_custom_longitudinal_tuning(CI, CP, CP_SP, params_dict)
  _initialize_coop_steering(CP, CP_SP, params_dict)
  _initialize_tesla_mads_screen_button(CP, CP_SP, params_dict)
  _initialize_radar_tracks(CP, CP_SP, can_recv, can_send)
  _initialize_stop_and_go(CP, CP_SP, params_dict)
  _initialize_toyota(CP, CP_SP, params_dict)
  _initialize_honda(CP, CP_SP, params_dict)  # FORK(HONDA_ACCORD_9G_AU): stock ACC mode


def _initialize_custom_longitudinal_tuning(CI, CP: structs.CarParams, CP_SP: structs.CarParamsSP,
                                           params_dict: dict[str, str]) -> None:

  # Hyundai Custom Longitudinal Tuning
  if CP.brand == 'hyundai':
    hyundai_longitudinal_tuning = int(params_dict.get("HyundaiLongitudinalTuning", 0))
    if hyundai_longitudinal_tuning == LongitudinalTuningType.DYNAMIC:
      CP_SP.flags |= HyundaiFlagsSP.LONG_TUNING_DYNAMIC.value
    if hyundai_longitudinal_tuning == LongitudinalTuningType.PREDICTIVE:
      CP_SP.flags |= HyundaiFlagsSP.LONG_TUNING_PREDICTIVE.value

  _ = CI.get_longitudinal_tuning_sp(CP, CP_SP)


def _initialize_coop_steering(CP: structs.CarParams, CP_SP: structs.CarParamsSP,
                              params_dict: dict[str, str]) -> None:
  if CP.brand == 'tesla':
    coop_steering = int(params_dict.get("TeslaCoopSteering", 0)) == 1
    if coop_steering:
      CP_SP.flags |= TeslaFlagsSP.COOP_STEERING.value


def _initialize_tesla_mads_screen_button(CP: structs.CarParams, CP_SP: structs.CarParamsSP,
                                         params_dict: dict[str, str]) -> None:
  if CP.brand == 'tesla' and CP_SP.flags & TeslaFlagsSP.HAS_VEHICLE_BUS:
    selection = int(params_dict.get("TeslaMadsScreenButton", MadsScreenButtonType.OFF))
    if selection == MadsScreenButtonType.THREE_FINGER:
      CP_SP.flags |= TeslaFlagsSP.MADS_SCREEN_BUTTON_3_FINGER.value
      CP_SP.safetyParam |= TeslaSafetyFlagsSP.MADS_SCREEN_BUTTON_3_FINGER
    elif selection == MadsScreenButtonType.FOUR_FINGER:
      CP_SP.flags |= TeslaFlagsSP.MADS_SCREEN_BUTTON_4_FINGER.value
      CP_SP.safetyParam |= TeslaSafetyFlagsSP.MADS_SCREEN_BUTTON_4_FINGER
    elif selection == MadsScreenButtonType.FIVE_FINGER:
      CP_SP.flags |= TeslaFlagsSP.MADS_SCREEN_BUTTON_5_FINGER.value
      CP_SP.safetyParam |= TeslaSafetyFlagsSP.MADS_SCREEN_BUTTON_5_FINGER


def _initialize_radar_tracks(CP: structs.CarParams, CP_SP: structs.CarParamsSP,
                             can_recv: CanRecvCallable | None = None, can_send: CanSendCallable | None = None) -> None:
  if can_recv is None or can_send is None or os.environ.get("REPLAY"):
    return

  if CP.brand == 'hyundai':
    if CP.flags & HyundaiFlags.MANDO_RADAR and (CP.radarUnavailable or CP_SP.flags & HyundaiFlagsSP.ENHANCED_SCC):
      tracks_enabled = hyundai_enable_radar_tracks(can_recv, can_send, bus=0, addr=0x7d0)
      CP.radarUnavailable = not tracks_enabled


def _initialize_stop_and_go(CP: structs.CarParams, CP_SP: structs.CarParamsSP, params_dict: dict[str, str]) -> None:
  if CP.brand == 'subaru' and not CP.flags & (SubaruFlags.GLOBAL_GEN2 | SubaruFlags.HYBRID):
    stop_and_go = int(params_dict.get("SubaruStopAndGo", 0)) == 1
    stop_and_go_manual_parking_brake = int(params_dict.get("SubaruStopAndGoManualParkingBrake", 0)) == 1

    if stop_and_go:
      CP_SP.flags |= SubaruFlagsSP.STOP_AND_GO.value
    if stop_and_go_manual_parking_brake:
      CP_SP.flags |= SubaruFlagsSP.STOP_AND_GO_MANUAL_PARKING_BRAKE.value
    if stop_and_go or stop_and_go_manual_parking_brake:
      CP_SP.safetyParam |= SubaruSafetyFlagsSP.STOP_AND_GO


def _initialize_toyota(CP: structs.CarParams, CP_SP: structs.CarParamsSP, params_dict: dict[str, str]) -> None:
  if CP.brand == 'toyota':
    toyota_stock_long = int(params_dict.get("ToyotaEnforceStockLongitudinal", 0)) == 1
    toyota_stop_and_go_hack = int(params_dict.get("ToyotaStopAndGoHack", 0)) == 1

    if toyota_stock_long:
      CP_SP.flags |= ToyotaFlagsSP.STOCK_LONGITUDINAL.value
      CP.alphaLongitudinalAvailable = False
      CP.openpilotLongitudinalControl = False
      CP.safetyConfigs[0].safetyParam |= ToyotaSafetyFlags.STOCK_LONGITUDINAL.value

    if toyota_stop_and_go_hack and CP.openpilotLongitudinalControl:
      CP_SP.flags |= ToyotaFlagsSP.STOP_AND_GO_HACK.value


# FORK(HONDA_ACCORD_9G_AU): stock ACC mode (HondaElesysStockAcc). The car's own ACC (the Elesys radar, panda bus 2)
# does gas and brake, openpilot steers only, and the panda forwards every frame both ways. The one writer of the mode:
# _get_params/_get_params_sp have already decided everything from "openpilot long" (the stand-down bit, the
# interceptor, pcmCruise), so all of it is undone here, in one place, from one param - never a half state. With the
# param off this is a no-op and CarParams/CarParamsSP are byte-identical to before. minEnableSpeed stays 19 mph.
# See docs/fork/CAR-HONDA-ACCORD-9G-AU.md, "Stock ACC mode".
def _initialize_honda(CP: structs.CarParams, CP_SP: structs.CarParamsSP, params_dict: dict[str, str]) -> None:
  if CP.brand == 'honda' and CP.carFingerprint in HONDA_ELESYS:
    if int(params_dict.get("HondaElesysStockAcc", 0)) == 1:
      CP_SP.flags |= HondaFlagsSP.ELESYS_STOCK_ACC.value
      CP.openpilotLongitudinalControl = False
      CP.pcmCruise = True
      CP.autoResumeSng = False
      # the pedal passes the driver's foot through with no 0x200; the panda also forces its interceptor off
      CP_SP.enableGasInterceptor = False
      CP_SP.safetyParam &= ~HondaSafetyFlagsSP.GAS_INTERCEPTOR
      # bit 32 (the SCM_BUTTONS stand-down) was set by _get_params; bits 32 and 64 together are never sent
      safety_param = CP.safetyConfigs[-1].safetyParam & ~HondaSafetyFlags.ELESYS_SCM_STANDDOWN.value
      CP.safetyConfigs[-1].safetyParam = safety_param | HondaSafetyFlags.ELESYS_STOCK_ACC.value
      carlog.warning("Honda ELESYS stock ACC mode: openpilot longitudinal off, all frames forwarded")
    # FORK(HONDA_ACCORD_9G_AU): the brake pump rule and the brake law, openpilot long only - never in stock ACC mode,
    # where nothing longitudinal is sent. Read here, once, so the route's CarParamsSP records what it ran; the
    # controller reads only CP_SP.flags. A key missing from params_dict (opendbc without openpilot) is off, i.e. the
    # rule and law before these settings; on the device card always passes both, with params_keys.h's defaults.
    elif CP.openpilotLongitudinalControl:
      if _param_is_on(params_dict.get("HondaElesysPumpV6")):
        CP_SP.flags |= HondaFlagsSP.ELESYS_PUMP_V6.value
      if _param_is_on(params_dict.get("HondaElesysBrakeLawV2")):
        CP_SP.flags |= HondaFlagsSP.ELESYS_BRAKE_LAW_V2.value


# FORK(HONDA_ACCORD_9G_AU): a BOOL param as card hands it over (bool from Params, or "1"/1/b"1"); anything else is off
def _param_is_on(value) -> bool:
  if isinstance(value, bytes):
    value = value.decode(errors="ignore")
  if isinstance(value, str):
    return value.strip() == "1"
  return isinstance(value, int) and value == 1   # bool is an int: True == 1
