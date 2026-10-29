#!/usr/bin/env python3
import math
import numpy as np
from opendbc.car import ACCELERATION_DUE_TO_GRAVITY, get_safety_config, structs, uds
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.disable_ecu import disable_ecu
from opendbc.car.honda.hondacan import CanBus
from opendbc.car.honda.values import CarControllerParams, HondaFlags, CAR, HondaSafetyFlags, HONDA_BOSCH, HONDA_BOSCH_CANFD, HONDA_BOSCH_RADARLESS
from opendbc.car.honda.values import HONDA_ELESYS  # FORK(HONDA_ACCORD_9G_AU): HONDA_ELESYS
from opendbc.car.honda.carcontroller import CarController
from opendbc.car.honda.carstate import CarState
from opendbc.car.honda.radar_interface import RadarInterface
from opendbc.car.interfaces import CarInterfaceBase

from opendbc.sunnypilot.car.honda.values_ext import HondaFlagsSP, HondaSafetyFlagsSP

TransmissionType = structs.CarParams.TransmissionType

# FORK(HONDA_ACCORD_9G_AU): the longitudinal PID's floor on this car (round 8, proposal 9).
#
# compute_gb_honda_elesys() put full brake at net -2.6 m/s^2 (brake = -net / 2.6; -3.25 since brake320, below), so a PID
# output below that asks for nothing more: the brake is already at its ceiling. Upstream's NIDEC_ACCEL_MIN of -4.0 only gives the
# integrator room to wind while the brake is railed. Measured once in 42 engaged minutes (drives 13d-143): route 143
# t172.9-174.6, the stopped car at night. The PID sat clipped at -4.0 for 1.7 s with its integrator wound to -1.33
# while 0x1FA was already at full brake (plan -2.1 to -2.9, achieved -1.65 to -1.92). Had the plan let go there, that
# integrator would have kept the brake on until it unwound.
#
# The floor is -3.2 on the level and downhill, and follows a climb. The dynamic tuner's pitch feedforward (hill_accel,
# carcontroller.py) is added AFTER the PID, and on a climb it is positive: it takes braking back. A flat -3.2 is full
# brake only up to about 6% (hill_accel 0.6); on a 10% climb it left about 205 of 254 counts (round 8 review, the
# real CarController). So on a climb the floor is lowered by the climb's own term, g*sin(pitch), from the same pose
# the tuner filters (controlsd passes CC.orientationNED's pitch): floor + hill_accel stays at or below -3.2, full
# brake, up to about 8%; from there the floor is upstream's -4.0, so on no grade does it brake less than before round
# 8. Without a pose (none yet, or not finite) it is -4.0 as well. Above the floor the PID is upstream's to the bit.
# The integrator stops winding as soon as the output reaches the floor: replayed on 143 (a slight downhill, so -3.2)
# it froze at -0.81 instead of -1.33, the brake command was full on all 265 frames either way, and on 13d, 141 and 142
# not one frame changed. Not in stock ACC mode (HondaElesysStockAcc), where nothing longitudinal is sent: its
# carControl stays bit-identical too. Every other car keeps NIDEC_ACCEL_MIN.
#
# brake320 (2026-10-24): -3.2 -> -3.25. Full brake is now net -3.25 (brake fraction 1.25 = 320 / 256 counts at the
# unchanged 2.6 m/s^2 per 256; brake_ceiling.py), so -3.2 stopped 0.05 short of it (fraction 1.231, 312 counts instead
# of 317 at gain 1.0). -3.25 is exactly full brake again on the level, and the climb term still follows (full brake up to
# about 7.6%, where -4.0 binds). NOT the -3.85 the old 0.6 margin would suggest: that margin dates from before the floor
# followed a climb, and below full brake it is only room for the integrator to wind while 0x1FA is already full - the
# 143 windup this floor was built to stop (-1.33 at -4.0 against -0.81 at -3.2). The closed-loop sim (brake320 study)
# brakes no harder with -3.85, and releases later. After brake320's VSA fallback (255 again, fraction 1.0) the room
# below full brake is 0.65 instead of 0.6.
ELESYS_PID_ACCEL_MIN = -3.25  # m/s^2, on the level and downhill


def elesys_pid_floor_applies(CP, CP_SP) -> bool:
  """Does get_pid_accel_limits() take this car's floor: HONDA_ELESYS with the gas interceptor, stock ACC mode off."""
  return (CP.carFingerprint in HONDA_ELESYS and bool(getattr(CP_SP, "enableGasInterceptor", False)) and
          not (int(getattr(CP_SP, "flags", 0)) & HondaFlagsSP.ELESYS_STOCK_ACC.value))


def elesys_pid_accel_min(CP, CP_SP, pitch=0.0) -> float:
  """The PID's negative limit for a Nidec car with the gas interceptor. On HONDA_ELESYS outside stock ACC mode:
  ELESYS_PID_ACCEL_MIN, lowered on a climb by g*sin(pitch) (pitch in radians, nose-up positive, as orientationNED[1]),
  never below NIDEC_ACCEL_MIN, and NIDEC_ACCEL_MIN without a usable pitch (None or not finite). NIDEC_ACCEL_MIN
  everywhere else."""
  if not (CP.carFingerprint in HONDA_ELESYS and not (int(getattr(CP_SP, "flags", 0)) & HondaFlagsSP.ELESYS_STOCK_ACC.value)):
    return CarControllerParams.NIDEC_ACCEL_MIN
  try:
    p = float(pitch)
  except (TypeError, ValueError):
    return CarControllerParams.NIDEC_ACCEL_MIN
  if not math.isfinite(p) or abs(p) >= math.pi / 2:
    return CarControllerParams.NIDEC_ACCEL_MIN
  uphill = max(ACCELERATION_DUE_TO_GRAVITY * math.sin(p), 0.0)
  return max(CarControllerParams.NIDEC_ACCEL_MIN, ELESYS_PID_ACCEL_MIN - uphill)


class CarInterface(CarInterfaceBase):
  CarState = CarState
  CarController = CarController
  RadarInterface = RadarInterface

  DRIVABLE_GEARS = (structs.CarState.GearShifter.sport,)

  @staticmethod
  def get_pid_accel_limits(CP, CP_SP, current_speed, cruise_speed):
    if CP.carFingerprint in HONDA_BOSCH:
      return CarControllerParams.BOSCH_ACCEL_MIN, CarControllerParams.BOSCH_ACCEL_MAX
    elif CP_SP.enableGasInterceptor:
      # FORK(HONDA_ACCORD_9G_AU): this car's PID floor is ELESYS_PID_ACCEL_MIN, not -4.0 (elesys_pid_accel_min())
      return elesys_pid_accel_min(CP, CP_SP), CarControllerParams.NIDEC_ACCEL_MAX
    else:
      # NIDECs don't allow acceleration near cruise_speed,
      # so limit limits of pid to prevent windup
      ACCEL_MAX_VALS = [CarControllerParams.NIDEC_ACCEL_MAX, 0.2]
      ACCEL_MAX_BP = [cruise_speed - 2., cruise_speed - .2]
      return CarControllerParams.NIDEC_ACCEL_MIN, np.interp(current_speed, ACCEL_MAX_BP, ACCEL_MAX_VALS)

  @staticmethod
  def _get_params(ret: structs.CarParams, candidate, fingerprint, car_fw, alpha_long, is_release, docs) -> structs.CarParams:
    ret.brand = "honda"

    CAN = CanBus(ret, fingerprint)

    if ret.flags & HondaFlags.BOSCH:
      cfgs = [get_safety_config(structs.CarParams.SafetyModel.hondaBosch)]
      if ret.flags & HondaFlags.BOSCH_CANFD and CAN.pt >= 4:
        cfgs.insert(0, get_safety_config(structs.CarParams.SafetyModel.noOutput))
      ret.safetyConfigs = cfgs

      ret.radarUnavailable = True
      # Disable the radar and let openpilot control longitudinal
      # WARNING: THIS DISABLES AEB!
      # If Bosch radarless, this blocks ACC messages from the camera
      # TODO: get radar disable working on Bosch CANFD
      ret.alphaLongitudinalAvailable = not (ret.flags & HondaFlags.BOSCH_CANFD)
      ret.openpilotLongitudinalControl = alpha_long and not (ret.flags & HondaFlags.BOSCH_CANFD)
      ret.pcmCruise = not ret.openpilotLongitudinalControl
    else:
      ret.safetyConfigs = [get_safety_config(structs.CarParams.SafetyModel.hondaNidec)]
      ret.openpilotLongitudinalControl = True

      ret.pcmCruise = True

    if candidate == CAR.HONDA_CRV_5G:
      ret.enableBsm = 0x12f8bfa7 in fingerprint[CAN.radar]

    # Detect Bosch cars with new HUD msgs
    if any(0x33DA in f for f in fingerprint.values()):
      ret.flags |= HondaFlags.BOSCH_EXT_HUD.value

    if 0x184 in fingerprint[CAN.pt]:
      ret.flags |= HondaFlags.HYBRID.value

    # FORK(HONDA_ELESYS): this car's gearbox is the LEGACY GEARBOX_AUTO, 0x188 (392), from
    # _gearbox_legacy.dbc -- the only platform that imports it. The check below only looks for
    # 0x191 / 0x1A3, so the car fell through to `manual`, and carstate.py then hardcodes
    # gearShifter = drive|reverse off REVERSE_LIGHT. Confirmed on route 15646e8515eda1a7: 0x188
    # is on bus 0 at 100 Hz and decodes cleanly (GEAR_SHIFTER 27|4 and GEAR 36|5 agree on
    # P/R/N/D across 43.7 min of driving), so the gear was on the wire and simply not read.
    # Consequences of leaving it: the car reported "drive" while sitting in PARK, and S was
    # invisible -- which is what the dynamic tuner needs in order not to learn the pedal map of
    # one gear against another.
    # Scoped to HONDA_ELESYS on purpose. ACURA_RDX also has 392 with neither 0x191 nor 0x1A3 and
    # is misdetected the same way, but that is an upstream platform with no data here; widening
    # this test would silently change its gear reporting too.
    if candidate in HONDA_ELESYS and 0x188 in fingerprint[CAN.pt]:
      ret.transmissionType = TransmissionType.automatic
    elif all(msg not in fingerprint[CAN.pt] for msg in (0x191, 0x1A3)):
      ret.transmissionType = TransmissionType.manual
    elif 0x191 in fingerprint[CAN.pt] and candidate != CAR.ACURA_RDX:
      # Traditional CVTs, gearshift position in GEARBOX_CVT
      ret.transmissionType = TransmissionType.cvt
    else:
      # Traditional autos, direct-drive EVs and eCVTs, gearshift position in GEARBOX_AUTO
      ret.transmissionType = TransmissionType.automatic

    ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0], [0]]
    ret.lateralTuning.pid.kiBP, ret.lateralTuning.pid.kpBP = [[0.], [0.]]
    ret.lateralTuning.pid.kf = 0.00006  # conservative feed-forward
    ret.steerActuatorDelay = 0.1

    if ret.flags & HondaFlags.BOSCH:
      ret.longitudinalActuatorDelay = 0.5 # s
      if ret.flags & HondaFlags.BOSCH_RADARLESS:
        ret.stopAccel = CarControllerParams.BOSCH_ACCEL_MIN  # stock uses -4.0 m/s^2 once stopped but limited by safety model
    else:
      # default longitudinal tuning for all hondas
      ret.longitudinalTuning.kiBP = [0., 5., 35.]
      ret.longitudinalTuning.kiV = [1.2, 0.8, 0.5]

      if candidate in HONDA_ELESYS:
        ret.longitudinalActuatorDelay = 0.6
        # vEgoStopping (0.8 on this car) is in CarParams.deprecated upstream and assigning it raises;
        # the 0.8 m/s stopping speed is carried per fingerprint in openpilot/sunnypilot/selfdrive/controls/lib/stopping_tune.py instead.
        # FORK(HONDA_ELESYS): the default -2.0 lands on top of the ~1.15 m/s^2 creep offset that
        # compute_gb_honda_elesys already adds, so the standstill hold commanded cb 253 of 255 --
        # near max hydraulic pressure, held for 7.8 min of a 65 min drive. The car does not need
        # it: 32 frames of motion in 46,815 hold frames (routes 15646e8515eda1a7 1f+20, incl.
        # graded stops). -0.8 lands the hold near cb 192 and still leaves ~0.8 m/s^2 of holding
        # margin ON TOP of the creep cancellation, which covers roughly an 8% grade.
        # If a stop ever creeps, raise this back toward -1.2 before touching the creep table.
        ret.stopAccel = -0.8

    # Disable control if EPS mod detected
    for fw in car_fw:
      if fw.ecu == "eps" and b"," in fw.fwVersion:
        ret.dashcamOnly = True

    if candidate == CAR.HONDA_CIVIC:
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 2560], [0, 2560]]
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[1.1], [0.33]]

    elif candidate in (CAR.HONDA_CIVIC_BOSCH, CAR.HONDA_CIVIC_BOSCH_DIESEL):
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]  # TODO: determine if there is a dead zone at the top end
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.8], [0.24]]
      if candidate == CAR.HONDA_CIVIC_BOSCH:
          CarControllerParams.BOSCH_GAS_LOOKUP_V = [0, 750]

    elif candidate == CAR.HONDA_CIVIC_2022:
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]  # TODO: determine if there is a dead zone at the top end
      ret.lateralTuning.pid.kpBP, ret.lateralTuning.pid.kpV = [[0, 10], [0.05, 0.5]]
      ret.lateralTuning.pid.kiBP, ret.lateralTuning.pid.kiV = [[0, 10], [0.0125, 0.125]]

    elif candidate == CAR.HONDA_ACCORD:
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]  # TODO: determine if there is a dead zone at the top end
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.6], [0.18]]

    elif candidate == CAR.ACURA_ILX:
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 3840], [0, 3840]]  # TODO: determine if there is a dead zone at the top end
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.8], [0.24]]

    elif candidate in (CAR.HONDA_CRV, CAR.HONDA_CRV_EU):
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 1000], [0, 1000]]  # TODO: determine if there is a dead zone at the top end
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.8], [0.24]]
      ret.wheelSpeedFactor = 1.025

    elif candidate == CAR.HONDA_CRV_5G:
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 3840], [0, 3840]]
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.64], [0.192]]
      ret.wheelSpeedFactor = 1.025

    elif candidate == CAR.HONDA_CRV_HYBRID:
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]  # TODO: determine if there is a dead zone at the top end
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.6], [0.18]]
      ret.wheelSpeedFactor = 1.025

    elif candidate == CAR.HONDA_FIT:
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]  # TODO: determine if there is a dead zone at the top end
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.2], [0.05]]

    elif candidate == CAR.HONDA_FREED:
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.2], [0.05]]

    elif candidate in (CAR.HONDA_HRV, CAR.HONDA_HRV_3G):
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]
      if candidate == CAR.HONDA_HRV:
        ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.16], [0.025]]
        ret.wheelSpeedFactor = 1.025
      else:
        ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.8], [0.24]]  # TODO: can probably use some tuning

    elif candidate == CAR.ACURA_RDX:
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 1000], [0, 1000]]  # TODO: determine if there is a dead zone at the top end
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.8], [0.24]]

    elif candidate == CAR.ACURA_RDX_3G:
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 3840], [0, 3840]]
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.2], [0.06]]

    elif candidate == CAR.HONDA_ODYSSEY:
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.28], [0.08]]
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]  # TODO: determine if there is a dead zone at the top end

    elif candidate == CAR.HONDA_ODYSSEY_TWN:
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.28], [0.08]]
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 32767], [0, 32767]]  # TODO: determine if there is a dead zone at the top end

    elif candidate == CAR.HONDA_PILOT:
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]  # TODO: determine if there is a dead zone at the top end
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.38], [0.11]]

    elif candidate == CAR.HONDA_RIDGELINE:
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]  # TODO: determine if there is a dead zone at the top end
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.38], [0.11]]

    elif candidate in (CAR.HONDA_INSIGHT, CAR.HONDA_NBOX_2G):
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]  # TODO: determine if there is a dead zone at the top end
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.6], [0.18]]

    elif candidate in (CAR.HONDA_E, CAR.HONDA_E_ADVANCE):
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]  # TODO: determine if there is a dead zone at the top end
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.6], [0.18]] # TODO: can probably use some tuning

    elif candidate == CAR.HONDA_ODYSSEY_5G_MMR:
      # Stock camera sends up to 2560 during LKA operation and up to 3840 during RDM operation
      # Steer motor torque does rise a little above 2560, but not linearly, RDM also applies one-sided brake drag
      #ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 2560, 3072], [0, 2560, 3840]]
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 2560], [0, 2560]]
      CarInterfaceBase.configure_torque_tune(candidate, ret.lateralTuning)
      ret.steerActuatorDelay = 0.15
      CarControllerParams.BOSCH_GAS_LOOKUP_V = [0, 2000]
      if not ret.openpilotLongitudinalControl:
        # When using stock ACC, the radar intercepts and filters steering commands the EPS would otherwise accept
        ret.minSteerSpeed = 70. * CV.KPH_TO_MS

    elif candidate in (CAR.ACURA_TLX_2G_MMR, CAR.ACURA_MDX_4G):
      ret.steerActuatorDelay = 0.15
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 3840], [0, 3840]]
      CarInterfaceBase.configure_torque_tune(candidate, ret.lateralTuning)

    # TODO-SP: remove when https://github.com/commaai/opendbc/pull/2687 is merged
    elif candidate == CAR.HONDA_CLARITY:
      pass

    else:
      ret.steerActuatorDelay = 0.15
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 2560], [0, 2560]]
      CarInterfaceBase.configure_torque_tune(candidate, ret.lateralTuning)

    if candidate in HONDA_ELESYS:
      # FORK(HONDA_ELESYS): the steering request does not go out on CAN to a camera-fed EPS.
      # It goes to the gateway board, which retimes it onto the car's 9600-baud LKAS serial
      # link at the camera's own cadence, and the EPS then runs its own torque loop on it.
      # That whole chain is measured, not guessed: openpilot's own delay learner (lateralDelay)
      # converged to 0.383 s on route 000000d3 and 0.377 s on 000000d4, and to 0.342 s (12
      # blocks) by route fd; while it was unestimated its running estimate sat at 0.36-0.39.
      #
      # So the car's delay is ~0.38 s, and this line is 0.18 because nothing reads it bare:
      # lagd normally overrides it frame by frame, and both places it does not add 0.2 s.
      # Before the learner has blocks (no usable cache) lagd publishes initial_lag =
      # steerActuatorDelay + 0.2; with the LagdToggle off, LagdToggle.update() returns
      # steerActuatorDelay + LagdToggleDelay (default 0.2). With 0.38 here both fallbacks were
      # 0.58, and lagd sat at 0.580 "unestimated" from route d7 to partway through fd (about
      # 5 h: it learns only above 50 mph). 0.18 lands both on the measured 0.38. lagd's cache
      # key is fingerprint + VERSION, so this does not discard a learned value. The only other
      # readers are the UI's delay display and the torque extension's first-frame jerk timing.
      ret.steerActuatorDelay = 0.18

      # FORK(HONDA_ELESYS): keep lateral alive at a stop so the cluster keeps its
      # lane graphic, which is what the stock camera does. controlsd computes
      #   standstill = abs(vEgo) <= max(minSteerSpeed, 0.3) or CS.standstill
      #   CC.latActive = ... and (not standstill or CP.steerAtStandstill)
      # so with this False, latActive - and therefore STEER_TORQUE_REQUEST on
      # 0x0E4 - drops at every red light, the gateway board reads that as "not
      # armed" and blanks 0x33D. Stock keeps the dashed lanes up throughout.
      #
      # It does NOT let anything steer at a standstill. The board has its own
      # guard: below GW_STANDSTILL_CPH (5.00 km/h) it raises the engage-guard
      # inhibit, target is zero by construction, and 0x704 reports STANDSTILL.
      # This only keeps openpilot's REQUEST alive so the graphic can follow it.
      ret.steerAtStandstill = True

      # FORK(HONDA_ELESYS): seed the torque controller's lateral-accel offset with the value this
      # car learns, instead of configure_torque_tune()'s 0.0. On routes f2/fc/fd/103 the logged
      # feedforward is exactly desired - roll*g - offset, and torqued's offset sits at -0.42 to -0.50
      # (replay: -0.39 to -0.47): it cancels about 0.4 m/s^2 of road crossfall in the device roll.
      # torqued starts its offset from this value (FORK hunk in torqued.py) whenever it has no valid
      # cache - which the HONDA_ACCORD_9G_AU prior in override.toml forces once - and the torque
      # controllers start from it too. Without the seed the feedforward loses ~0.43 m/s^2 (~60
      # serial counts) until torqued is valid again (23-58 min of driving in the replays), more
      # than the 0.25 the integrator may carry into a takeover (LINBUS_I_CARRY_MAX).
      # sunnypilot's re-run of configure_torque_tune() (EnforceTorqueControl or NNLC on) would reset
      # it to 0.0; its interfaces.py keeps it (FORK hunk there, 2026-10).
      if ret.lateralTuning.which() == 'torque':
        ret.lateralTuning.torque.latAccelOffset = -0.43

    if candidate == CAR.ACURA_RDX_3G_MMR:
      CarControllerParams.BOSCH_GAS_LOOKUP_V = [0, 2000] # alpha longitudinal pedal tuning
      ret.dashcamOnly = is_release  # TODO: release from dashcam when there's enough driving data for torqued/paramsd to converge

    # These cars use alternate user brake msg (0x1BE)
    if 0x1BE in fingerprint[CAN.pt] and (candidate in (CAR.HONDA_ACCORD, CAR.HONDA_HRV_3G, CAR.ACURA_RDX_3G, CAR.ACURA_MDX_4G) or
                                          ret.flags & HondaFlags.BOSCH_CANFD):
      ret.flags |= HondaFlags.BOSCH_ALT_BRAKE.value

    if ret.flags & HondaFlags.BOSCH_ALT_BRAKE:
      ret.safetyConfigs[-1].safetyParam |= HondaSafetyFlags.ALT_BRAKE.value
    if ret.flags & HondaFlags.NIDEC_ALT_SCM_MESSAGES:
      ret.safetyConfigs[-1].safetyParam |= HondaSafetyFlags.NIDEC_ALT.value
    if ret.openpilotLongitudinalControl and ret.flags & HondaFlags.BOSCH:
      ret.safetyConfigs[-1].safetyParam |= HondaSafetyFlags.BOSCH_LONG.value
    if ret.flags & HondaFlags.BOSCH_RADARLESS:
      ret.safetyConfigs[-1].safetyParam |= HondaSafetyFlags.RADARLESS.value
    if ret.flags & HondaFlags.BOSCH_CANFD:
      ret.safetyConfigs[-1].safetyParam |= HondaSafetyFlags.BOSCH_CANFD.value
    if candidate in HONDA_ELESYS and ret.openpilotLongitudinalControl:
      ret.safetyConfigs[-1].safetyParam |= HondaSafetyFlags.ELESYS_SCM_STANDDOWN.value
      # FORK(HONDA_ACCORD_9G_AU): brake320 (owner decision 2026-10-24), the brake ceiling 320 counts instead of 255 in
      # panda and in the CarController alike (opendbc/sunnypilot/car/honda/brake_ceiling.py). Stock ACC mode clears it
      # with the stand-down (sunnypilot/car/interfaces.py _initialize_honda())
      ret.safetyConfigs[-1].safetyParam |= HondaSafetyFlags.ELESYS_BRAKE_320.value
      # FORK(BRAKE-SWITCH-MIRROR): the 0x17C BRAKE_SWITCH mirror to the radar (owner decision 2026-10-29). On for this
      # variant in software, like the stand-down above and brake320: openpilot re-sends POWERTRAIN_DATA on bus 2 with
      # BRAKE_SWITCH forced on while the lamp board is driving lit brake lamps, and the panda blocks the car's own copy
      # to the radar. There is no toggle; stock ACC mode clears it with the other two
      # (sunnypilot/car/interfaces.py _initialize_honda()), and the CarController reads it back off this param
      # (opendbc/sunnypilot/car/honda/brake_switch_mirror.py brake_switch_mirror_on())
      ret.safetyConfigs[-1].safetyParam |= HondaSafetyFlags.ELESYS_BRAKE_SWITCH_MIRROR.value

    # min speed to enable ACC. if car can do stop and go, then set enabling speed
    # to a negative value, so it won't matter. Otherwise, add 0.5 mph margin to not
    # conflict with PCM acc
    ret.autoResumeSng = bool(ret.flags & HondaFlags.BOSCH) or candidate == CAR.HONDA_CIVIC
    if ret.autoResumeSng:
      ret.minEnableSpeed = -1.
    elif candidate in (CAR.HONDA_ODYSSEY_TWN, CAR.HONDA_ACCORD_9G_AU):
      ret.minEnableSpeed = 19. * CV.MPH_TO_MS
    else:
      ret.minEnableSpeed = 25.51 * CV.MPH_TO_MS

    ret.steerLimitTimer = 0.8
    ret.radarDelay = 0.1

    return ret

  @staticmethod
  def _get_params_sp(stock_cp: structs.CarParams, ret: structs.CarParamsSP, candidate, fingerprint: dict[int, dict[int, int]],
                     car_fw: list[structs.CarParams.CarFw], alpha_long: bool, is_release_sp: bool, docs: bool) -> structs.CarParamsSP:
    CAN = CanBus(stock_cp, fingerprint)

    for fw in car_fw:
      if fw.ecu == "eps" and b"," in fw.fwVersion:
        ret.flags |= HondaFlagsSP.EPS_MODIFIED.value
        stock_cp.dashcamOnly = False

    if bool(stock_cp.flags & HondaFlags.NIDEC) and bool(stock_cp.flags & HondaFlags.HYBRID):
      ret.flags |= HondaFlagsSP.NIDEC_HYBRID.value
      ret.safetyParam |= HondaSafetyFlagsSP.NIDEC_HYBRID
      # some hybrids use a different brake hold
      if 0x223 in fingerprint[CAN.pt]:
        ret.flags |= HondaFlagsSP.HYBRID_ALT_BRAKEHOLD.value

    if candidate == CAR.HONDA_CIVIC:
      if ret.flags & HondaFlagsSP.EPS_MODIFIED:
        # stock request input values:     0x0000, 0x00DE, 0x014D, 0x01EF, 0x0290, 0x0377, 0x0454, 0x0610, 0x06EE
        # stock request output values:    0x0000, 0x0917, 0x0DC5, 0x1017, 0x119F, 0x140B, 0x1680, 0x1680, 0x1680
        # modified request output values: 0x0000, 0x0917, 0x0DC5, 0x1017, 0x119F, 0x140B, 0x1680, 0x2880, 0x3180
        # stock filter output values:     0x009F, 0x0108, 0x0108, 0x0108, 0x0108, 0x0108, 0x0108, 0x0108, 0x0108
        # modified filter output values:  0x009F, 0x0108, 0x0108, 0x0108, 0x0108, 0x0108, 0x0108, 0x0400, 0x0480
        # note: max request allowed is 4096, but request is capped at 3840 in firmware, so modifications result in 2x max
        stock_cp.lateralParams.torqueBP, stock_cp.lateralParams.torqueV = [[0, 2560, 8000], [0, 2560, 3840]]
        stock_cp.lateralTuning.pid.kpV, stock_cp.lateralTuning.pid.kiV = [[0.3], [0.1]]

    elif candidate in (CAR.HONDA_CIVIC_BOSCH, CAR.HONDA_CIVIC_BOSCH_DIESEL):
      if ret.flags & HondaFlagsSP.EPS_MODIFIED:
        stock_cp.lateralParams.torqueBP, stock_cp.lateralParams.torqueV = [[0, 2564, 8000], [0, 2564, 3840]]
        stock_cp.lateralTuning.pid.kpV, stock_cp.lateralTuning.pid.kiV = [[0.3], [0.09]]  # 2.5x Modded EPS

    elif candidate == CAR.HONDA_CIVIC_2022:
      if ret.flags & HondaFlagsSP.EPS_MODIFIED:
        stock_cp.lateralParams.torqueBP, stock_cp.lateralParams.torqueV = [[0, 2564, 8000], [0, 2564, 3840]]
        stock_cp.lateralTuning.pid.kpV, stock_cp.lateralTuning.pid.kiV = [[0.3], [0.09]]  # 2.5x Modded EPS

    elif candidate == CAR.HONDA_ACCORD:
      if ret.flags & HondaFlagsSP.EPS_MODIFIED:
        stock_cp.lateralTuning.pid.kpV, stock_cp.lateralTuning.pid.kiV = [[0.3], [0.09]]

    elif candidate == CAR.HONDA_CRV_5G:
      if ret.flags & HondaFlagsSP.EPS_MODIFIED:
        # stock request input values:     0x0000, 0x00DB, 0x01BB, 0x0296, 0x0377, 0x0454, 0x0532, 0x0610, 0x067F
        # stock request output values:    0x0000, 0x0500, 0x0A15, 0x0E6D, 0x1100, 0x1200, 0x129A, 0x134D, 0x1400
        # modified request output values: 0x0000, 0x0500, 0x0A15, 0x0E6D, 0x1100, 0x1200, 0x1ACD, 0x239A, 0x2800
        stock_cp.lateralParams.torqueBP, stock_cp.lateralParams.torqueV = [[0, 2560, 10000], [0, 2560, 3840]]
        stock_cp.lateralTuning.pid.kpV, stock_cp.lateralTuning.pid.kiV = [[0.21], [0.07]]

    elif candidate == CAR.HONDA_CLARITY:
      stock_cp.autoResumeSng = True
      stock_cp.minEnableSpeed = -1
      if ret.flags & HondaFlagsSP.EPS_MODIFIED:
        for fw in car_fw:
          if fw.ecu == "eps" and b"-" not in fw.fwVersion and b"," in fw.fwVersion:
            stock_cp.lateralTuning.pid.kf = 0.00004
            stock_cp.lateralParams.torqueBP, stock_cp.lateralParams.torqueV = [[0, 5760, 15360], [0, 2560, 3840]]
            stock_cp.lateralTuning.pid.kpV, stock_cp.lateralTuning.pid.kiV = [[0.1575], [0.05175]]
          elif fw.ecu == "eps" and b"-" in fw.fwVersion and b"," in fw.fwVersion:
            stock_cp.lateralParams.torqueBP, stock_cp.lateralParams.torqueV = [[0, 5760, 10240], [0, 2560, 3840]]
            stock_cp.lateralTuning.pid.kpV, stock_cp.lateralTuning.pid.kiV = [[0.3], [0.1]]
      else:
        stock_cp.lateralParams.torqueBP, stock_cp.lateralParams.torqueV = [[0, 2560], [0, 2560]]
        stock_cp.lateralTuning.pid.kpV, stock_cp.lateralTuning.pid.kiV = [[0.8], [0.24]]

    if candidate in HONDA_BOSCH:
      pass
    else:
      ret.enableGasInterceptor = 0x201 in fingerprint[CAN.pt]
      stock_cp.pcmCruise = not ret.enableGasInterceptor

    if ret.enableGasInterceptor and candidate not in HONDA_BOSCH:
      ret.safetyParam |= HondaSafetyFlagsSP.GAS_INTERCEPTOR

    stock_cp.autoResumeSng = stock_cp.autoResumeSng or ret.enableGasInterceptor
    # FORK(HONDA_ACCORD_9G_AU): HONDA_ELESYS keeps its 19 mph minEnableSpeed with the interceptor
    stock_cp.minEnableSpeed = -1. if ret.enableGasInterceptor and candidate not in HONDA_ELESYS else stock_cp.minEnableSpeed

    ret.intelligentCruiseButtonManagementAvailable = candidate in (HONDA_BOSCH - HONDA_BOSCH_CANFD) or \
                                                     (candidate in HONDA_BOSCH_CANFD and not is_release_sp)

    return ret

  @staticmethod
  def init(CP, CP_SP, can_recv, can_send, communication_control=None):
    if CP.carFingerprint in (HONDA_BOSCH - HONDA_BOSCH_RADARLESS) and CP.openpilotLongitudinalControl:
      # 0x80 silences response
      if communication_control is None:
        communication_control = bytes([uds.SERVICE_TYPE.COMMUNICATION_CONTROL, 0x80 | uds.CONTROL_TYPE.DISABLE_RX_DISABLE_TX,
                                       uds.MESSAGE_TYPE.NORMAL_AND_NETWORK_MANAGEMENT])
      disable_ecu(can_recv, can_send, bus=CanBus(CP).pt, addr=0x18DAB0F1, com_cont_req=communication_control)

  @staticmethod
  def deinit(CP, can_recv, can_send):
    communication_control = bytes([uds.SERVICE_TYPE.COMMUNICATION_CONTROL, 0x80 | uds.CONTROL_TYPE.ENABLE_RX_ENABLE_TX,
                                   uds.MESSAGE_TYPE.NORMAL_AND_NETWORK_MANAGEMENT])
    CarInterface.init(CP, can_recv, can_send, communication_control)
