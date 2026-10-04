#!/usr/bin/env python3
import unittest
import numpy as np

from opendbc.car.honda.values import HondaSafetyFlags
from opendbc.safety import ALTERNATIVE_EXPERIENCE
from opendbc.safety.tests.libsafety import libsafety_py
import opendbc.safety.tests.common as common
from opendbc.car.structs import CarParams
from opendbc.safety.tests.common import CANPackerSafety, MAX_WRONG_COUNTERS, make_msg
from opendbc.safety.tests.gas_interceptor_common import GasInterceptorSafetyTest

from opendbc.sunnypilot.car.honda.values_ext import HondaSafetyFlagsSP

HONDA_N_COMMON_TX_MSGS = [[0xE4, 0], [0x194, 0], [0x1FA, 0], [0x30C, 0], [0x33D, 0]]


class Btn:
  NONE = 0
  MAIN = 1
  CANCEL = 2
  SET = 3
  RESUME = 4

# Honda safety has several different configurations tested here:
#  * Nidec
#    * normal (PCM-enable)
#    * alt SCM messages  (PCM-enable)
#    * gas interceptor (button-enable)
#    * gas interceptor with alt SCM messages (button-enable)
#  * Bosch
#    * Bosch with Longitudinal Support
#  * Bosch Radarless
#    * Bosch Radarless with Longitudinal Support


class HondaButtonEnableBase(common.CarSafetyTest):

  # override these inherited tests since we're using button enable
  def test_disable_control_allowed_from_cruise(self):
    pass

  def test_enable_control_allowed_from_cruise(self):
    pass

  def test_cruise_engaged_prev(self):
    pass

  def test_buttons_with_main_off(self):
    for btn in (Btn.SET, Btn.RESUME, Btn.CANCEL):
      self.safety.set_controls_allowed(1)
      self._rx(self._acc_state_msg(False))
      self._rx(self._button_msg(btn, main_on=False))
      self.assertFalse(self.safety.get_controls_allowed())

  def test_set_resume_buttons(self):
    """
      Both SET and RES should enter controls allowed on their falling edge.
    """
    for main_on in (True, False):
      self._rx(self._acc_state_msg(main_on))
      for btn_prev in range(8):
        for btn_cur in range(8):
          self._rx(self._button_msg(Btn.NONE))
          self.safety.set_controls_allowed(0)
          for _ in range(10):
            self._rx(self._button_msg(btn_prev))
            self.assertFalse(self.safety.get_controls_allowed())

          # should enter controls allowed on falling edge and not transitioning to cancel or main
          should_enable = (main_on and
                           btn_cur != btn_prev and
                           btn_prev in (Btn.RESUME, Btn.SET) and
                           btn_cur not in (Btn.CANCEL, Btn.MAIN))

          self._rx(self._button_msg(btn_cur, main_on=main_on))
          self.assertEqual(should_enable, self.safety.get_controls_allowed(), msg=f"{main_on=} {btn_prev=} {btn_cur=}")

  def test_main_cancel_buttons(self):
    """
      Both MAIN and CANCEL should exit controls immediately.
    """
    for btn in (Btn.MAIN, Btn.CANCEL):
      self.safety.set_controls_allowed(1)
      self._rx(self._button_msg(btn, main_on=True))
      self.assertFalse(self.safety.get_controls_allowed())

  def test_disengage_on_main(self):
    self.safety.set_controls_allowed(1)
    self._rx(self._acc_state_msg(True))
    self.assertTrue(self.safety.get_controls_allowed())
    self._rx(self._acc_state_msg(False))
    self.assertFalse(self.safety.get_controls_allowed())

  def test_rx_hook(self):

    # TODO: move this test to common
    # checksum checks
    for msg_type in ["btn", "gas", "speed"]:
      self.safety.set_controls_allowed(1)
      if msg_type == "btn":
        msg = self._button_msg(Btn.SET)
      if msg_type == "gas":
        msg = self._user_gas_msg(0)
      if msg_type == "speed":
        msg = self._speed_msg(0)
      self.assertTrue(self._rx(msg))
      if msg_type != "btn":
        msg[0].data[4] = 0  # invalidate checksum
        msg[0].data[5] = 0
        msg[0].data[6] = 0
        msg[0].data[7] = 0
        self.assertFalse(self._rx(msg))
        self.assertFalse(self.safety.get_controls_allowed())

    # counter
    # reset wrong_counters to zero by sending valid messages
    for i in range(MAX_WRONG_COUNTERS + 1):
      self.__class__.cnt_speed += 1
      self.__class__.cnt_button += 1
      self.__class__.cnt_powertrain_data += 1
      if i < MAX_WRONG_COUNTERS:
        self.safety.set_controls_allowed(1)
        self._rx(self._button_msg(Btn.SET))
        self._rx(self._speed_msg(0))
        self._rx(self._user_gas_msg(0))
      else:
        self.assertFalse(self._rx(self._button_msg(Btn.SET)))
        self.assertFalse(self._rx(self._speed_msg(0)))
        self.assertFalse(self._rx(self._user_gas_msg(0)))
        self.assertFalse(self.safety.get_controls_allowed())

    # restore counters for future tests with a couple of good messages
    for _ in range(2):
      self.safety.set_controls_allowed(1)
      self._rx(self._button_msg(Btn.SET, main_on=True))
      self._rx(self._speed_msg(0))
      self._rx(self._user_gas_msg(0))
    self._rx(self._button_msg(Btn.SET, main_on=True))
    self.assertTrue(self.safety.get_controls_allowed())


class HondaPcmEnableBase(common.CarSafetyTest):

  def test_buttons(self):
    """
      Buttons should only cancel in this configuration,
      since our state is tied to the PCM's cruise state.
    """
    for controls_allowed in (True, False):
      for main_on in (True, False):
        # not a valid state
        if controls_allowed and not main_on:
          continue

        for btn in (Btn.SET, Btn.RESUME, Btn.CANCEL):
          self.safety.set_controls_allowed(controls_allowed)
          self._rx(self._acc_state_msg(main_on))

          # btn + none for falling edge
          self._rx(self._button_msg(btn, main_on=main_on))
          self._rx(self._button_msg(Btn.NONE, main_on=main_on))

          if btn == Btn.CANCEL:
            self.assertFalse(self.safety.get_controls_allowed())
          else:
            self.assertEqual(controls_allowed, self.safety.get_controls_allowed())


class HondaBase(common.CarSafetyTest):
  MAX_BRAKE = 255
  PT_BUS: int | None = None  # must be set when inherited
  STEER_BUS: int | None = None  # must be set when inherited
  BUTTONS_BUS: int | None = None  # must be set when inherited, tx on this bus, rx on PT_BUS

  RELAY_MALFUNCTION_ADDRS = {0: (0xE4, 0x194)}  # STEERING_CONTROL

  cnt_speed = 0
  cnt_button = 0
  cnt_brake = 0
  cnt_powertrain_data = 0
  cnt_acc_state = 0

  def _powertrain_data_msg(self, cruise_on=None, brake_pressed=None, gas_pressed=None):
    # preserve the state
    if cruise_on is None:
      # or'd with controls allowed since the tests use it to "enable" cruise
      cruise_on = self.safety.get_cruise_engaged_prev() or self.safety.get_controls_allowed()
    if brake_pressed is None:
      brake_pressed = self.safety.get_brake_pressed_prev()
    if gas_pressed is None:
      gas_pressed = self.safety.get_gas_pressed_prev()

    values = {
      "ACC_STATUS": cruise_on,
      "BRAKE_PRESSED": brake_pressed,
      "PEDAL_GAS": gas_pressed,
      "COUNTER": self.cnt_powertrain_data % 4
    }
    self.__class__.cnt_powertrain_data += 1
    return self.packer.make_can_msg_safety("POWERTRAIN_DATA", self.PT_BUS, values)

  def _pcm_status_msg(self, enable):
    return self._powertrain_data_msg(cruise_on=enable)

  def _speed_msg(self, speed):
    values = {"XMISSION_SPEED": speed, "COUNTER": self.cnt_speed % 4}
    self.__class__.cnt_speed += 1
    return self.packer.make_can_msg_safety("ENGINE_DATA", self.PT_BUS, values)

  def _acc_state_msg(self, main_on):
    values = {"MAIN_ON": main_on, "COUNTER": self.cnt_acc_state % 4}
    self.__class__.cnt_acc_state += 1
    return self.packer.make_can_msg_safety("SCM_FEEDBACK", self.PT_BUS, values)

  def _button_msg(self, buttons, main_on=False, bus=None):
    bus = self.PT_BUS if bus is None else bus
    values = {"CRUISE_BUTTONS": buttons, "COUNTER": self.cnt_button % 4}
    self.__class__.cnt_button += 1
    return self.packer.make_can_msg_safety("SCM_BUTTONS", bus, values)

  def _user_brake_msg(self, brake):
    return self._powertrain_data_msg(brake_pressed=brake)

  def _user_gas_msg(self, gas):
    return self._powertrain_data_msg(gas_pressed=gas)

  def _send_steer_msg(self, steer):
    values = {"STEER_TORQUE": steer}
    return self.packer.make_can_msg_safety("STEERING_CONTROL", self.STEER_BUS, values)

  def _send_brake_msg(self, brake):
    # must be implemented when inherited
    raise NotImplementedError

  def test_disengage_on_brake(self):
    self.safety.set_controls_allowed(1)
    self._rx(self._user_brake_msg(1))
    self.assertFalse(self.safety.get_controls_allowed())

  def test_steer_safety_check(self):
    self.safety.set_controls_allowed(0)
    self.assertTrue(self._tx(self._send_steer_msg(0x0000)))
    self.assertFalse(self._tx(self._send_steer_msg(0x1000)))

  def _lkas_button_msg(self, lkas_button=False, setting_btn=0):
    values = {"CRUISE_SETTING": 1 if lkas_button else setting_btn, "COUNTER": self.cnt_button % 4}
    self.__class__.cnt_button += 1
    return self.packer.make_can_msg_safety("SCM_BUTTONS", self.PT_BUS, values)

  def test_enable_control_allowed_with_mads_button(self):
    """Tests MADS button state transitions and internal button press state."""
    for enable_mads in (True, False):
      with self.subTest("enable_mads", mads_enabled=enable_mads):
        self.safety.set_mads_params(enable_mads, False, False)

        # Verify initial state
        self._rx(self._lkas_button_msg(False, 0))
        self.assertEqual(0, self.safety.get_mads_button_press())  # NOT_PRESSED
        self.assertFalse(self.safety.get_controls_allowed_lateral())

        # Verify press sets correct internal state
        self._rx(self._lkas_button_msg(False, 1))
        self.assertEqual(1, self.safety.get_mads_button_press())  # PRESSED
        self.assertEqual(enable_mads, self.safety.get_controls_allowed_lateral())

        # Verify release sets correct internal state
        self._rx(self._lkas_button_msg(False, 0))
        self.assertEqual(0, self.safety.get_mads_button_press())  # NOT_PRESSED
        self.assertEqual(enable_mads, self.safety.get_controls_allowed_lateral())

        # Test invalid values - should not change button press state
        for invalid_setting in (2, 3):
          self._rx(self._lkas_button_msg(False, invalid_setting))
          self.assertEqual(0, self.safety.get_mads_button_press())  # Should remain NOT_PRESSED
          self.assertEqual(enable_mads, self.safety.get_controls_allowed_lateral())

        # Verify we can still transition after invalid values
        self._rx(self._lkas_button_msg(False, 1))
        self.assertEqual(1, self.safety.get_mads_button_press())
        self._rx(self._lkas_button_msg(False, 0))
        self.assertEqual(0, self.safety.get_mads_button_press())


# ********************* Honda Nidec **********************


class TestHondaNidecSafetyBase(HondaBase):
  TX_MSGS = HONDA_N_COMMON_TX_MSGS
  FWD_BLACKLISTED_ADDRS = {2: [0xE4, 0x194, 0x33D, 0x30C]}
  RELAY_MALFUNCTION_ADDRS = {0: (0xE4, 0x194, 0x33D, 0x30C)}

  PT_BUS = 0
  STEER_BUS = 0
  BUTTONS_BUS = 0

  MAX_GAS = 198

  BRAKE_SIG = "COMPUTER_BRAKE"

  def setUp(self):
    self.packer = CANPackerSafety("honda_civic_touring_2016_can_generated")
    self.safety = libsafety_py.libsafety
    self.safety.set_safety_hooks(CarParams.SafetyModel.hondaNidec, 0)
    self.safety.init_tests()

  def _send_brake_msg(self, brake, aeb_req=0, bus=0):
    values = {self.BRAKE_SIG: brake, "AEB_REQ_1": aeb_req}
    return self.packer.make_can_msg_safety("BRAKE_COMMAND", bus, values)

  def _rx_brake_msg(self, brake, aeb_req=0):
    return self._send_brake_msg(brake, aeb_req, bus=2)

  def _send_acc_hud_msg(self, pcm_gas, pcm_speed):
    # Used to control ACC on Nidec without pedal
    values = {"PCM_GAS": pcm_gas, "PCM_SPEED": pcm_speed}
    return self.packer.make_can_msg_safety("ACC_HUD", 0, values)

  def test_acc_hud_safety_check(self):
    for controls_allowed in [True, False]:
      self.safety.set_controls_allowed(controls_allowed)
      for pcm_gas in range(255):
        for pcm_speed in range(100):
          send = (controls_allowed and pcm_gas <= self.MAX_GAS) or (pcm_gas == 0 and pcm_speed == 0)
          self.assertEqual(send, self._tx(self._send_acc_hud_msg(pcm_gas, pcm_speed)))

  def test_fwd_hook(self):
    # normal operation, not forwarding AEB
    self.FWD_BLACKLISTED_ADDRS[2].append(0x1FA)
    self.safety.set_honda_fwd_brake(False)
    super().test_fwd_hook()

    # forwarding AEB brake signal
    self.FWD_BLACKLISTED_ADDRS = {2: [0xE4, 0x194, 0x33D, 0x30C]}
    self.safety.set_honda_fwd_brake(True)
    super().test_fwd_hook()

  def test_honda_fwd_brake_latching(self):
    # Shouldn't fwd stock Honda requesting brake without AEB
    self.assertTrue(self._rx(self._rx_brake_msg(self.MAX_BRAKE, aeb_req=0)))
    self.assertFalse(self.safety.get_honda_fwd_brake())

    # Now allow controls and request some brake
    openpilot_brake = round(self.MAX_BRAKE / 2.0)
    self.safety.set_controls_allowed(True)
    self.assertTrue(self._tx(self._send_brake_msg(openpilot_brake)))

    # Still shouldn't fwd stock Honda brake until it's more than openpilot's
    for stock_honda_brake in range(self.MAX_BRAKE + 1):
      self.assertTrue(self._rx(self._rx_brake_msg(stock_honda_brake, aeb_req=1)))
      should_fwd_brake = stock_honda_brake >= openpilot_brake
      self.assertEqual(should_fwd_brake, self.safety.get_honda_fwd_brake())

    # Shouldn't stop fwding until AEB event is over
    for stock_honda_brake in range(self.MAX_BRAKE + 1)[::-1]:
      self.assertTrue(self._rx(self._rx_brake_msg(stock_honda_brake, aeb_req=1)))
      self.assertTrue(self.safety.get_honda_fwd_brake())

    self.assertTrue(self._rx(self._rx_brake_msg(0, aeb_req=0)))
    self.assertFalse(self.safety.get_honda_fwd_brake())

  def test_brake_safety_check(self):
    for fwd_brake in [False, True]:
      self.safety.set_honda_fwd_brake(fwd_brake)
      for brake in np.arange(0, self.MAX_BRAKE + 10, 1):
        for controls_allowed in [True, False]:
          self.safety.set_controls_allowed(controls_allowed)
          if fwd_brake:
            send = False  # block openpilot brake msg when fwd'ing stock msg
          elif controls_allowed:
            send = self.MAX_BRAKE >= brake >= 0
          else:
            send = brake == 0
          self.assertEqual(send, self._tx(self._send_brake_msg(brake)))


class TestHondaNidecPcmSafety(HondaPcmEnableBase, TestHondaNidecSafetyBase):
  """
    Covers the Honda Nidec safety mode
  """

  # Nidec doesn't disengage on falling edge of cruise. See comment in safety_honda.h
  def test_disable_control_allowed_from_cruise(self):
    pass


class TestHondaNidecGasInterceptorSafety(GasInterceptorSafetyTest, HondaButtonEnableBase, TestHondaNidecSafetyBase):
  """
    Covers the Honda Nidec safety mode with a gas interceptor, switches to a button-enable car
  """

  TX_MSGS = HONDA_N_COMMON_TX_MSGS + [[0x200, 0]]
  INTERCEPTOR_THRESHOLD = 492

  def setUp(self):
    self.packer = CANPackerSafety("honda_civic_touring_2016_can_generated")
    self.safety = libsafety_py.libsafety
    self.safety.set_current_safety_param_sp(HondaSafetyFlagsSP.GAS_INTERCEPTOR)
    self.safety.set_safety_hooks(CarParams.SafetyModel.hondaNidec, 0)
    self.safety.init_tests()


class TestHondaNidecPcmAltSafety(TestHondaNidecPcmSafety):
  """
    Covers the Honda Nidec safety mode with alt SCM messages
  """
  def setUp(self):
    self.packer = CANPackerSafety("acura_ilx_2016_can_generated")
    self.safety = libsafety_py.libsafety
    self.safety.set_safety_hooks(CarParams.SafetyModel.hondaNidec, HondaSafetyFlags.NIDEC_ALT)
    self.safety.init_tests()

  def _acc_state_msg(self, main_on):
    values = {"MAIN_ON": main_on, "COUNTER": self.cnt_acc_state % 4}
    self.__class__.cnt_acc_state += 1
    return self.packer.make_can_msg_safety("SCM_BUTTONS", self.PT_BUS, values)

  def _button_msg(self, buttons, main_on=False, bus=None):
    bus = self.PT_BUS if bus is None else bus
    values = {"CRUISE_BUTTONS": buttons, "MAIN_ON": main_on, "COUNTER": self.cnt_button % 4}
    self.__class__.cnt_button += 1
    return self.packer.make_can_msg_safety("SCM_BUTTONS", bus, values)


class TestHondaNidecAltGasInterceptorSafety(GasInterceptorSafetyTest, HondaButtonEnableBase, TestHondaNidecSafetyBase):
  """
    Covers the Honda Nidec safety mode with alt SCM messages and gas interceptor, switches to a button-enable car
  """

  TX_MSGS = HONDA_N_COMMON_TX_MSGS + [[0x200, 0]]
  INTERCEPTOR_THRESHOLD = 492

  def setUp(self):
    self.packer = CANPackerSafety("acura_ilx_2016_can_generated")
    self.safety = libsafety_py.libsafety
    self.safety.set_current_safety_param_sp(HondaSafetyFlagsSP.GAS_INTERCEPTOR)
    self.safety.set_safety_hooks(CarParams.SafetyModel.hondaNidec, HondaSafetyFlags.NIDEC_ALT)
    self.safety.init_tests()

  def _acc_state_msg(self, main_on):
    values = {"MAIN_ON": main_on, "COUNTER": self.cnt_acc_state % 4}
    self.__class__.cnt_acc_state += 1
    return self.packer.make_can_msg_safety("SCM_BUTTONS", self.PT_BUS, values)

  def _button_msg(self, buttons, main_on=False, bus=None):
    bus = self.PT_BUS if bus is None else bus
    values = {"CRUISE_BUTTONS": buttons, "MAIN_ON": main_on, "COUNTER": self.cnt_button % 4}
    self.__class__.cnt_button += 1
    return self.packer.make_can_msg_safety("SCM_BUTTONS", bus, values)


class TestHondaElesysScmStanddownSafety(TestHondaNidecPcmAltSafety):
  """
    HONDA_ACCORD_9G_AU (Elesys radar) stock-ACC stand-down (ELESYS_SCM_STANDDOWN): OP re-sends SCM_BUTTONS
    (0x1A6) on bus 2 with MAIN_ON=0 and the stock 0x1A6 is blocked bus 0 -> 2.
    0x33D (4-byte LKAS_HUD) is forwarded from the stock camera, not sent by OP.
  """
  # 0x500 is SP_HUD_STATUS for the LIN-bus gateway, on bus 0 (the module sits on the camera's bus;
  # bus 2 is the Elesys radar branch on this harness)
  TX_MSGS = HONDA_N_COMMON_TX_MSGS + [[0x1A6, 2], [0x500, 0]]
  FWD_BLACKLISTED_ADDRS = {2: [0xE4, 0x194, 0x30C], 0: [0x1A6]}
  RELAY_MALFUNCTION_ADDRS = {0: (0xE4, 0x194, 0x30C)}

  def setUp(self):
    self.packer = CANPackerSafety("acura_ilx_2016_can_generated")
    self.safety = libsafety_py.libsafety
    self.safety.set_safety_hooks(CarParams.SafetyModel.hondaNidec, HondaSafetyFlags.NIDEC_ALT | HondaSafetyFlags.ELESYS_SCM_STANDDOWN)
    self.safety.init_tests()

  def _send_brake_msg(self, brake, aeb_req=0, bus=0):
    # this platform's stock-AEB flag is read from bit 43 (FCW field), not AEB_REQ_1 (bit 29)
    values = {self.BRAKE_SIG: brake, "FCW": aeb_req * 2}
    return self.packer.make_can_msg_safety("BRAKE_COMMAND", bus, values)

  def test_acc_hud_safety_check(self):
    # normal gas rules, except the factory-camera braking pattern (pcm_gas=198, pcm_speed=0) is always allowed
    for controls_allowed in [True, False]:
      self.safety.set_controls_allowed(controls_allowed)
      for pcm_gas in range(255):
        for pcm_speed in range(100):
          send = (controls_allowed and pcm_gas <= self.MAX_GAS) or (pcm_gas == 0 and pcm_speed == 0) or (pcm_gas == 198 and pcm_speed == 0)
          self.assertEqual(send, self._tx(self._send_acc_hud_msg(pcm_gas, pcm_speed)))

  def test_fwd_hook(self):
    # 0x1A6 (SCM_BUTTONS) is blocked bus 0 -> 2 under stand-down; 0x33D is forwarded (stock camera HUD)
    self.FWD_BLACKLISTED_ADDRS = {2: [0xE4, 0x194, 0x30C, 0x1FA], 0: [0x1A6]}
    self.safety.set_honda_fwd_brake(False)
    super(TestHondaNidecSafetyBase, self).test_fwd_hook()

    self.FWD_BLACKLISTED_ADDRS = {2: [0xE4, 0x194, 0x30C], 0: [0x1A6]}
    self.safety.set_honda_fwd_brake(True)
    super(TestHondaNidecSafetyBase, self).test_fwd_hook()

  # FORK(UPSTREAM-FIX): route 114, seg 13, as logged. SCM_BUTTONS (0x1A6) from 804.51 to 805.50, with the LKAS press
  # (CRUISE_SETTING 1) at 805.19-805.25. The panda's heartbeat exit was at ~804.25 and its next 1 Hz tick at ~805.21,
  # before pandad's heartbeat could report MADS engaged; that tick revoked the grant and the car got no steering
  # for 2 s (controlsMismatchLateral at 807.53).
  ROUTE_114_SCM_BUTTONS = """
    0008002a8e800005 0008002a8e800014 0008002a8e800023 0008002a8a800036 0008002a8a800009 0008002d8a800015
    0008002d8a800024 0008002d8a800033 0008002d9a800005 0008002d9a800014 000800239a80002d 000800239a80003c
    000800239a80000f 00080023a1800016 00080023a1800025 0008001ea180003a 0008001ea180000d 0008001ea180001c
    0008001ea7800025 0008001ea7800034 0008001aa780000b 0008001aa780001a 0008001aa7800029 0008001aab800034
    0008001aab800007 00080017ab800019 00080017ab800028 00080017ab800037 00080017a680000f 00080017a680001e
    0008001ba6800029 0008001ba6800038 0008001ba680000b 0008001ba680001a 0008001ba6800029 0008001ba6840034
    0008001ba6840007 0008001ba6840016 0008001ba6800029 0008001ba6800038 0008001ba680000b 0008001ba680001a
    0008001ba6800029 0008001ba6800038 0008001ba680000b 0008001ba680001a 0008001ba6800029 0008001ba6800038
    0008001bad800004 0008001bad800013""".split()

  def test_route_114_lkas_regrant_survives_the_next_heartbeat_tick(self):
    frames = [bytes.fromhex(x) for x in self.ROUTE_114_SCM_BUTTONS]
    self.safety.set_mads_params(True, False, False)   # ALT_EXP_ENABLE_MADS, as logged
    self.safety.set_heartbeat_engaged_mads(True)
    for dat in frames[:20]:                           # no press: MAIN_ON settles at 1
      self.assertTrue(self._rx(libsafety_py.make_CANPacket(0x1A6, 0, dat)))
    self.safety.set_controls_allowed_lateral(True)
    self.safety.mads_heartbeat_engaged_check()
    self.assertTrue(self.safety.get_controls_allowed_lateral())

    # 801.75: openpilot's MADS goes to disabled; the panda's third 1 Hz tick exits lateral
    self.safety.set_heartbeat_engaged_mads(False)
    for _ in range(3):
      self.safety.mads_heartbeat_engaged_check()
    self.assertFalse(self.safety.get_controls_allowed_lateral())

    # 805.19: the LKAS press grants lateral again
    for dat in frames[20:]:
      self.assertTrue(self._rx(libsafety_py.make_CANPacket(0x1A6, 0, dat)))
    self.assertTrue(self.safety.get_controls_allowed_lateral())

    # ~805.21: the next tick, with the heartbeat still saying MADS off. This revoked the grant on the car.
    self.safety.mads_heartbeat_engaged_check()
    self.assertTrue(self.safety.get_controls_allowed_lateral())
    self.safety.set_heartbeat_engaged_mads(True)
    self.safety.mads_heartbeat_engaged_check()
    self.assertTrue(self.safety.get_controls_allowed_lateral())


class TestHondaElesysStanddownGasInterceptorSafety(TestHondaNidecAltGasInterceptorSafety):
  """
    HONDA_ACCORD_9G_AU with comma pedal: ELESYS_SCM_STANDDOWN + gas interceptor.
    OP may send GAS_COMMAND (0x200) and the bus-2 SCM_BUTTONS re-send (0x1A6);
    0x33D (4-byte LKAS_HUD) is forwarded from the stock camera, not sent.
  """
  TX_MSGS = HONDA_N_COMMON_TX_MSGS + [[0x200, 0], [0x1A6, 2], [0x500, 0]]
  FWD_BLACKLISTED_ADDRS = {2: [0xE4, 0x194, 0x30C], 0: [0x1A6]}
  RELAY_MALFUNCTION_ADDRS = {0: (0xE4, 0x194, 0x30C)}

  def setUp(self):
    self.packer = CANPackerSafety("acura_ilx_2016_can_generated")
    self.safety = libsafety_py.libsafety
    self.safety.set_current_safety_param_sp(HondaSafetyFlagsSP.GAS_INTERCEPTOR)
    self.safety.set_safety_hooks(CarParams.SafetyModel.hondaNidec, HondaSafetyFlags.NIDEC_ALT | HondaSafetyFlags.ELESYS_SCM_STANDDOWN)
    self.safety.init_tests()

  def _send_brake_msg(self, brake, aeb_req=0, bus=0):
    # this platform's stock-AEB flag is read from bit 43 (FCW field), not AEB_REQ_1 (bit 29)
    values = {self.BRAKE_SIG: brake, "FCW": aeb_req * 2}
    return self.packer.make_can_msg_safety("BRAKE_COMMAND", bus, values)

  def test_acc_hud_safety_check(self):
    # normal gas rules, except the factory-camera braking pattern (pcm_gas=198, pcm_speed=0) is always allowed
    for controls_allowed in [True, False]:
      self.safety.set_controls_allowed(controls_allowed)
      for pcm_gas in range(255):
        for pcm_speed in range(100):
          send = (controls_allowed and pcm_gas <= self.MAX_GAS) or (pcm_gas == 0 and pcm_speed == 0) or (pcm_gas == 198 and pcm_speed == 0)
          self.assertEqual(send, self._tx(self._send_acc_hud_msg(pcm_gas, pcm_speed)))

  def test_fwd_hook(self):
    # 0x1A6 (SCM_BUTTONS) is blocked bus 0 -> 2 under stand-down; 0x33D is forwarded (stock camera HUD)
    self.FWD_BLACKLISTED_ADDRS = {2: [0xE4, 0x194, 0x30C, 0x1FA], 0: [0x1A6]}
    self.safety.set_honda_fwd_brake(False)
    super(TestHondaNidecSafetyBase, self).test_fwd_hook()

    self.FWD_BLACKLISTED_ADDRS = {2: [0xE4, 0x194, 0x30C], 0: [0x1A6]}
    self.safety.set_honda_fwd_brake(True)
    super(TestHondaNidecSafetyBase, self).test_fwd_hook()


# FORK(HONDA_ACCORD_9G_AU): stock ACC mode (ELESYS_STOCK_ACC). Packed with this car's own DBC.

# Every frame OP sends for longitudinal or the HUD in the other modes, at its real length. None may leave in stock
# ACC mode, whatever the content or the controls state: 0x1FA BRAKE_COMMAND and 0x30C ACC_HUD on both buses,
# 0x200 GAS_COMMAND, 0x1A6 SCM_BUTTONS (the stand-down re-send on bus 2, cancel/resume spam on bus 0),
# 0x33D LKAS_HUD (4 bytes on this car, 5 and 8 on others), 0xE5 BOSCH_SUPPLEMENTAL_1 and 0x296 buttons.
HONDA_ELESYS_LONG_TX = [(0x1FA, 0, 8), (0x1FA, 2, 8), (0x30C, 0, 8), (0x30C, 2, 8), (0x200, 0, 6), (0x1A6, 0, 8), (0x1A6, 2, 8),
                        (0x33D, 0, 4), (0x33D, 0, 5), (0x33D, 0, 8), (0xE5, 0, 8), (0x296, 0, 4), (0x296, 2, 4)]

# What the panda hears on this car. The radar (bus 2) sends only BRAKE_COMMAND and ACC_HUD; the rest is the car's (bus 0).
HONDA_ELESYS_RADAR_FRAMES = (("BRAKE_COMMAND", {"COMPUTER_BRAKE": 0}), ("ACC_HUD", {"PCM_SPEED": 50, "PCM_GAS": 10}))
HONDA_ELESYS_CAR_FRAMES = (("POWERTRAIN_DATA", {}), ("ENGINE_DATA", {"XMISSION_SPEED": 20}), ("SCM_BUTTONS", {"MAIN_ON": 1}),
                           ("LKAS_HUD", {}))


def honda_elesys_wire(test, relay_open: bool, rounds: int = 20):
  """Frames as the firmware sees them (fwd hook, then rx hook). With the harness relay open each frame arrives on its
  own bus; with it closed - a passive comma, routes 0e-82 - bus 0 and bus 2 are one wire and every frame arrives on
  both. Returns the (bus, addr, fwd) decisions and the relay_malfunction state after each frame."""
  seen = []
  for _ in range(rounds):
    for frames, home in ((HONDA_ELESYS_CAR_FRAMES, 0), (HONDA_ELESYS_RADAR_FRAMES, 2)):
      for name, values in frames:
        for bus in ((home,) if relay_open else (home, 2 - home)):
          msg = test.packer.make_can_msg_safety(name, bus, values)
          fwd = test.safety.safety_fwd_hook(bus, msg[0].addr)
          test.safety.safety_rx_hook(msg)
          seen.append((bus, msg[0].addr, fwd, test.safety.get_relay_malfunction()))
  return seen


class TestHondaElesysStockAccSafety(TestHondaNidecPcmAltSafety):
  """
    HONDA_ACCORD_9G_AU stock ACC mode (NIDEC_ALT | ELESYS_STOCK_ACC): the car's own ACC (the Elesys radar on bus 2)
    does gas and brake; OP sends steering and SP_HUD_STATUS (0x500) only and blocks nothing from forwarding: the
    radar's 0x1FA (stock ACC brake and CMBS) and 0x30C reach the car, the real 0x1A6 reaches the radar.
    The pedal is fingerprinted (SP GAS_INTERCEPTOR, as on the car) and must be ignored.
  """
  TX_MSGS = [[0xE4, 0], [0x194, 0], [0x500, 0]]
  FWD_BLACKLISTED_ADDRS = {2: [0xE4, 0x194]}
  # nothing on the car's side sends 0xE4/0x194: the radar's own frames on bus 0 are what show a relay that did not open
  RELAY_MALFUNCTION_ADDRS = {0: (0xE4, 0x194, 0x1FA, 0x30C)}

  def setUp(self):
    self.packer = CANPackerSafety("honda_accord_au_2015_can_generated")
    self.safety = libsafety_py.libsafety
    self.safety.set_current_safety_param_sp(HondaSafetyFlagsSP.GAS_INTERCEPTOR)
    self.safety.set_safety_hooks(CarParams.SafetyModel.hondaNidec, HondaSafetyFlags.NIDEC_ALT | HondaSafetyFlags.ELESYS_STOCK_ACC)
    self.safety.init_tests()

  def test_relay_open_forwards_everything(self):
    seen = honda_elesys_wire(self, relay_open=True)
    self.assertFalse(any(relay for *_, relay in seen))
    for bus, addr, fwd, _ in seen:
      self.assertEqual(2 - bus, fwd, f"{addr=:#x} from {bus=}")

  def test_stuck_relay_is_a_relay_malfunction(self):
    # a harness relay that did not open (undetected harness, loose or flipped cable, failed relay): the radar's
    # first frame seen on bus 0 trips it, and from then on nothing is forwarded back onto the same wire
    seen = honda_elesys_wire(self, relay_open=False)
    first = next(i for i, (*_, relay) in enumerate(seen) if relay)
    self.assertEqual(0, seen[first][0])
    self.assertTrue(seen[first][1] in (0x1FA, 0x30C), hex(seen[first][1]))
    self.assertFalse(any(relay for *_, relay in seen[:first]))
    self.assertTrue(all(relay for *_, relay in seen[first:]))
    self.assertTrue(all(fwd == -1 for _, _, fwd, _ in seen[first + 1:]))
    self.safety.set_controls_allowed(True)
    self.assertFalse(self._tx(self._send_steer_msg(0)))
    self.assertFalse(self._tx(self.packer.make_can_msg_safety("SP_HUD_STATUS", 0, {"LAT_ACTIVE": 1})))

  def test_stuck_relay_respects_the_transition_timeout(self):
    # the first second after the relay switches is not judged, as for every other relay check
    self.safety.set_safety_hooks(CarParams.SafetyModel.hondaNidec, HondaSafetyFlags.NIDEC_ALT | HondaSafetyFlags.ELESYS_STOCK_ACC)
    honda_elesys_wire(self, relay_open=False, rounds=2)
    self.assertFalse(self.safety.get_relay_malfunction())
    self.safety.init_tests()   # safety_mode_cnt = 2: past the timeout (the firmware counts it at 1 Hz)
    honda_elesys_wire(self, relay_open=False, rounds=1)
    self.assertTrue(self.safety.get_relay_malfunction())

  def test_refused_radar_frames_change_nothing(self):
    # 0x1FA/0x30C are relay checks here, never transmits: a refused BRAKE_COMMAND must not become OP's brake level,
    # which would hold the AEB latch below the radar's request
    self.safety.set_controls_allowed(True)
    self.assertFalse(self._tx(self._send_brake_msg(self.MAX_BRAKE)))
    self.assertFalse(self._tx(self._send_acc_hud_msg(0, 0)))
    self.assertTrue(self._rx(self._radar_brake_msg(COMPUTER_BRAKE=1, FCW=2)))
    self.assertTrue(self.safety.get_honda_fwd_brake())

  def _send_brake_msg(self, brake, aeb_req=0, bus=0):
    # this platform's stock-AEB flag is read from bit 43 (FCW field), not AEB_REQ_1 (bit 29)
    values = {self.BRAKE_SIG: brake, "FCW": aeb_req * 2}
    return self.packer.make_can_msg_safety("BRAKE_COMMAND", bus, values)

  def _radar_brake_msg(self, **values):
    return self.packer.make_can_msg_safety("BRAKE_COMMAND", 2, values)

  def test_contract_values(self):
    # fixed contract with the car port and sunnypilot: main safety param 4|64 (pandaStates.safetyParam 68)
    self.assertEqual(64, HondaSafetyFlags.ELESYS_STOCK_ACC)
    self.assertEqual(68, self.safety.get_current_safety_param())
    self.assertTrue(self.safety.get_honda_elesys_stock_acc())

  def test_acc_hud_safety_check(self):
    # OP never sends ACC_HUD in stock ACC mode, whatever the values or the controls state
    for controls_allowed in [True, False]:
      self.safety.set_controls_allowed(controls_allowed)
      for pcm_gas in range(255):
        for pcm_speed in range(100):
          self.assertFalse(self._tx(self._send_acc_hud_msg(pcm_gas, pcm_speed)))

  def test_brake_safety_check(self):
    # nor BRAKE_COMMAND, whatever the AEB latch says
    for fwd_brake in [False, True]:
      self.safety.set_honda_fwd_brake(fwd_brake)
      for brake in np.arange(0, self.MAX_BRAKE + 10, 1):
        for controls_allowed in [True, False]:
          self.safety.set_controls_allowed(controls_allowed)
          self.assertFalse(self._tx(self._send_brake_msg(brake)))

  def test_honda_fwd_brake_latching(self):
    # The latch reads this platform's bit 43 (FCW >= 2), as under the stand-down. It only reports here: the fwd
    # hook forwards 0x1FA whatever it says. OP sends no brake, so any stock brake counts as higher than OP's.
    self.assertTrue(self._rx(self._radar_brake_msg(COMPUTER_BRAKE=self.MAX_BRAKE, AEB_REQ_1=1)))
    self.assertFalse(self.safety.get_honda_fwd_brake())
    for fcw in range(4):
      self.assertTrue(self._rx(self._radar_brake_msg(COMPUTER_BRAKE=self.MAX_BRAKE, FCW=fcw)))
      self.assertEqual(fcw >= 2, self.safety.get_honda_fwd_brake(), f"{fcw=}")
      self.assertEqual(0, self.safety.safety_fwd_hook(2, 0x1FA))

    for stock_brake in range(self.MAX_BRAKE + 1):
      self.assertTrue(self._rx(self._radar_brake_msg(COMPUTER_BRAKE=stock_brake, FCW=2)))
      self.assertTrue(self.safety.get_honda_fwd_brake())

    self.assertTrue(self._rx(self._radar_brake_msg(COMPUTER_BRAKE=0, FCW=0)))
    self.assertFalse(self.safety.get_honda_fwd_brake())

  def test_fwd_hook(self):
    # nothing is blocked but the relay-checked steering frames toward the car (static blocking), latched or not
    for fwd_brake in [False, True]:
      self.safety.set_honda_fwd_brake(fwd_brake)
      super(TestHondaNidecSafetyBase, self).test_fwd_hook()

  def test_stock_acc_frames_forwarded(self):
    # the frames stock ACC and CMBS live on, spelled out: radar -> car 0x1FA (50 Hz) and 0x30C (10 Hz),
    # car -> radar the real 0x1A6 with the driver's MAIN_ON; and the stock camera's 0x33D
    for fwd_brake in [False, True]:
      self.safety.set_honda_fwd_brake(fwd_brake)
      self.assertEqual(0, self.safety.safety_fwd_hook(2, 0x1FA))
      self.assertEqual(0, self.safety.safety_fwd_hook(2, 0x30C))
      self.assertEqual(2, self.safety.safety_fwd_hook(0, 0x1A6))
      self.assertEqual(2, self.safety.safety_fwd_hook(0, 0x33D))

  def test_disable_stock_aeb_does_not_block(self):
    # ALT_EXP_DISABLE_STOCK_AEB has no effect on forwarding here: 0x1FA is the stock ACC's brake too
    self.safety.set_alternative_experience(ALTERNATIVE_EXPERIENCE.DISABLE_STOCK_AEB)
    self.assertTrue(self._rx(self._radar_brake_msg(COMPUTER_BRAKE=self.MAX_BRAKE, FCW=2)))
    self.assertFalse(self.safety.get_honda_fwd_brake())
    self.assertEqual(0, self.safety.safety_fwd_hook(2, 0x1FA))

  def test_long_tx_refused(self):
    packed = [
      self._send_brake_msg(0),
      self._send_brake_msg(0, bus=2),
      self._send_acc_hud_msg(0, 0),
      self.packer.make_can_msg_safety("GAS_COMMAND", 0, {}),
      self.packer.make_can_msg_safety("SCM_BUTTONS", 2, {"MAIN_ON": 0}),
      self.packer.make_can_msg_safety("SCM_BUTTONS", 0, {"CRUISE_BUTTONS": Btn.CANCEL}),
      self.packer.make_can_msg_safety("SCM_BUTTONS", 0, {"CRUISE_BUTTONS": Btn.RESUME}),
      self.packer.make_can_msg_safety("LKAS_HUD", 0, {}),
    ]
    for controls_allowed in [True, False]:
      self.safety.set_controls_allowed(controls_allowed)
      self.safety.set_controls_allowed_lateral(controls_allowed)
      for addr, bus, length in HONDA_ELESYS_LONG_TX:
        self.assertFalse(self._tx(make_msg(bus, addr, length)), f"{addr=:#x} {bus=} {length=}")
      for msg in packed:
        self.assertFalse(self._tx(msg), f"{msg[0].addr=:#x} {msg[0].bus=}")

  def test_steer_and_hud_tx(self):
    # steering and SP_HUD_STATUS go out on the car's bus at their own length, nowhere else
    self.safety.set_controls_allowed(True)
    for addr, length in ((0xE4, 5), (0x194, 4), (0x500, 8)):
      self.assertTrue(self._tx(make_msg(0, addr, length)), f"{addr=:#x}")
      for bus in (1, 2):
        self.assertFalse(self._tx(make_msg(bus, addr, length)), f"{addr=:#x} {bus=}")
      self.assertFalse(self._tx(make_msg(0, addr, 8 if length != 8 else 6)), f"{addr=:#x}")
    self.assertTrue(self._tx(self.packer.make_can_msg_safety("SP_HUD_STATUS", 0, {"LAT_ACTIVE": 1})))

  def test_gas_interceptor_ignored(self):
    # the pedal is fingerprinted, but stock ACC mode forces the interceptor off: GAS_COMMAND is never sent,
    # GAS_SENSOR is not read (no RX check), user gas comes from POWERTRAIN_DATA
    self.safety.set_controls_allowed(True)
    for gas in (0, 0x100, 0x1000, 0xFFFF):
      self.assertFalse(self._tx(self.packer.make_can_msg_safety("GAS_COMMAND", 0, {"GAS_COMMAND": gas, "GAS_COMMAND2": gas, "ENABLE": 1})))
      self._rx(self.packer.make_can_msg_safety("GAS_SENSOR", 0, {"INTERCEPTOR_GAS": gas, "INTERCEPTOR_GAS2": gas}))
      self.assertEqual(0, self.safety.get_gas_interceptor_prev())
      self.assertFalse(self.safety.get_gas_pressed_prev())
      self.assertTrue(self.safety.get_controls_allowed())
    self._rx(self._user_gas_msg(1))
    self.assertTrue(self.safety.get_gas_pressed_prev())

  def test_stock_acc_engagement(self):
    # controls follow the PCM's ACC_STATUS (stock ACC engaged) like a Nidec without the pedal; SET/RES never engage
    self._rx(self._acc_state_msg(True))
    self._rx(self._pcm_status_msg(False))
    self.assertFalse(self.safety.get_controls_allowed())
    for btn in (Btn.SET, Btn.RESUME):
      self._rx(self._button_msg(btn, main_on=True))
      self._rx(self._button_msg(Btn.NONE, main_on=True))
      self.assertFalse(self.safety.get_controls_allowed())

    self._rx(self._pcm_status_msg(True))
    self.assertTrue(self.safety.get_controls_allowed())

    # the stock ACC drops out by itself (~22 km/h): a Nidec keeps controls when the PCM disengages
    self._rx(self._pcm_status_msg(False))
    self.assertTrue(self.safety.get_controls_allowed())

    # the driver's MAIN off ends them
    self._rx(self._acc_state_msg(False))
    self.assertFalse(self.safety.get_controls_allowed())

  def test_bosch_init_clears_stock_acc(self):
    # the flag is module state: a later Bosch init must clear it, as it clears the stand-down
    self.assertTrue(self.safety.get_honda_elesys_stock_acc())
    self.safety.set_safety_hooks(CarParams.SafetyModel.hondaBosch, 0)
    self.assertFalse(self.safety.get_honda_elesys_stock_acc())

    # and so does a Nidec init without the bit
    self.safety.set_safety_hooks(CarParams.SafetyModel.hondaNidec, HondaSafetyFlags.ELESYS_STOCK_ACC)
    self.assertTrue(self.safety.get_honda_elesys_stock_acc())
    self.safety.set_safety_hooks(CarParams.SafetyModel.hondaNidec, HondaSafetyFlags.NIDEC_ALT | HondaSafetyFlags.ELESYS_SCM_STANDDOWN)
    self.assertFalse(self.safety.get_honda_elesys_stock_acc())


class TestHondaElesysStockAccStanddownConflictSafety(common.SafetyTest):
  """
    ELESYS_SCM_STANDDOWN and ELESYS_STOCK_ACC together is not a valid input (the car port writes the param in one
    place, so it should never reach the panda). Neither wins: the panda turns it into a stock car with CMBS intact,
    transmitting nothing and forwarding everything, with no stand-down and no pedal.
  """
  TX_MSGS: list[list[int]] = []
  FWD_BLACKLISTED_ADDRS: dict[int, list[int]] = {}

  def setUp(self):
    self.packer = CANPackerSafety("honda_accord_au_2015_can_generated")
    self.safety = libsafety_py.libsafety
    self.safety.set_current_safety_param_sp(HondaSafetyFlagsSP.GAS_INTERCEPTOR)
    self.safety.set_safety_hooks(CarParams.SafetyModel.hondaNidec,
                                 HondaSafetyFlags.NIDEC_ALT | HondaSafetyFlags.ELESYS_SCM_STANDDOWN | HondaSafetyFlags.ELESYS_STOCK_ACC)
    self.safety.init_tests()

  def test_fwd_hook(self):
    # everything both ways, the radar's 0x1FA and the real 0x1A6 included, latched or not
    for fwd_brake in [False, True]:
      self.safety.set_honda_fwd_brake(fwd_brake)
      super().test_fwd_hook()

  def test_nothing_transmitted(self):
    self.safety.set_controls_allowed(True)
    self.safety.set_controls_allowed_lateral(True)
    for addr, bus, length in HONDA_ELESYS_LONG_TX + [(0xE4, 0, 5), (0x194, 0, 4), (0x500, 0, 8)]:
      self.assertFalse(self._tx(make_msg(bus, addr, length)), f"{addr=:#x} {bus=} {length=}")
    self.assertFalse(self._tx(self.packer.make_can_msg_safety("STEERING_CONTROL", 0, {"STEER_TORQUE": 0})))
    self.assertFalse(self._tx(self.packer.make_can_msg_safety("SCM_BUTTONS", 2, {"MAIN_ON": 0})))

  def test_relay_malfunction_on_the_radars_frames_only(self):
    # nothing is transmitted, but the relay is still checked on the radar's own frames: forwarding everything
    # through a relay that did not open would put every frame back onto the same wire
    for bus in range(3):
      for addr in self.SCANNED_ADDRS:
        self.safety.set_relay_malfunction(False)
        self._rx(make_msg(bus, addr, 8))
        self.assertEqual(bus == 0 and addr in (0x1FA, 0x30C), self.safety.get_relay_malfunction(), (bus, hex(addr)))

  def test_stuck_relay_is_a_relay_malfunction(self):
    self.assertFalse(any(relay for *_, relay in honda_elesys_wire(self, relay_open=True)))
    seen = honda_elesys_wire(self, relay_open=False)
    first = next(i for i, (*_, relay) in enumerate(seen) if relay)
    self.assertEqual(0, seen[first][0])
    self.assertTrue(seen[first][1] in (0x1FA, 0x30C), hex(seen[first][1]))
    self.assertTrue(all(fwd == -1 for _, _, fwd, _ in seen[first + 1:]))

  def test_stock_acc_state(self):
    # stock ACC mode with no stand-down: the radar's AEB latch reads bit 43, the pedal is not read
    self.assertTrue(self.safety.get_honda_elesys_stock_acc())
    self.assertTrue(self._rx(self.packer.make_can_msg_safety("BRAKE_COMMAND", 2, {"COMPUTER_BRAKE": 100, "FCW": 2})))
    self.assertTrue(self.safety.get_honda_fwd_brake())
    self._rx(self.packer.make_can_msg_safety("GAS_SENSOR", 0, {"INTERCEPTOR_GAS": 0x1000, "INTERCEPTOR_GAS2": 0x1000}))
    self.assertEqual(0, self.safety.get_gas_interceptor_prev())


# ********************* Honda Bosch **********************


class TestHondaBoschSafetyBase(HondaBase):
  PT_BUS = 1
  STEER_BUS = 0
  BUTTONS_BUS = 1

  TX_MSGS = [[0xE4, 0], [0xE5, 0], [0x296, 1], [0x33D, 0], [0x33DA, 0], [0x33DB, 0]]
  FWD_BLACKLISTED_ADDRS = {2: [0xE4, 0xE5, 0x33D, 0x33DA, 0x33DB]}
  RELAY_MALFUNCTION_ADDRS = {0: (0xE4, 0xE5, 0x33D, 0x33DA, 0x33DB)}  # STEERING_CONTROL, BOSCH_SUPPLEMENTAL_1

  def setUp(self):
    self.packer = CANPackerSafety("honda_civic_hatchback_ex_2017_can_generated")
    self.safety = libsafety_py.libsafety

  def _alt_brake_msg(self, brake):
    values = {"BRAKE_PRESSED": brake, "COUNTER": self.cnt_brake % 4}
    self.__class__.cnt_brake += 1
    return self.packer.make_can_msg_safety("BRAKE_MODULE", self.PT_BUS, values)

  def _send_brake_msg(self, brake):
    pass

  def test_spam_cancel_safety_check(self):
    self.safety.set_controls_allowed(0)
    self.assertTrue(self._tx(self._button_msg(Btn.CANCEL, bus=self.BUTTONS_BUS)))
    self.assertFalse(self._tx(self._button_msg(Btn.RESUME, bus=self.BUTTONS_BUS)))
    self.assertFalse(self._tx(self._button_msg(Btn.SET, bus=self.BUTTONS_BUS)))
    # do not block resume if we are engaged already
    self.safety.set_controls_allowed(1)
    self.assertTrue(self._tx(self._button_msg(Btn.RESUME, bus=self.BUTTONS_BUS)))


class TestHondaBoschAltBrakeSafetyBase(TestHondaBoschSafetyBase):
  """
    Base Bosch safety test class with an alternate brake message
  """
  def setUp(self):
    super().setUp()
    self.safety.set_safety_hooks(CarParams.SafetyModel.hondaBosch, HondaSafetyFlags.ALT_BRAKE)
    self.safety.init_tests()

  def _user_brake_msg(self, brake):
    return self._alt_brake_msg(brake)

  def test_alt_brake_rx_hook(self):
    self.safety.set_honda_alt_brake_msg(1)
    self.safety.set_controls_allowed(1)
    msg = self._alt_brake_msg(0)
    self.assertTrue(self._rx(msg))
    msg[0].data[2] = msg[0].data[2] & 0xF0  # invalidate checksum
    self.assertFalse(self._rx(msg))
    self.assertFalse(self.safety.get_controls_allowed())

  def test_alt_disengage_on_brake(self):
    self.safety.set_honda_alt_brake_msg(1)
    self.safety.set_controls_allowed(1)
    self._rx(self._alt_brake_msg(1))
    self.assertFalse(self.safety.get_controls_allowed())

    self.safety.set_honda_alt_brake_msg(0)
    self.safety.set_controls_allowed(1)
    self._rx(self._alt_brake_msg(1))
    self.assertTrue(self.safety.get_controls_allowed())


class TestHondaBoschSafety(HondaPcmEnableBase, TestHondaBoschSafetyBase):
  """
    Covers the Honda Bosch safety mode with stock longitudinal
  """
  def setUp(self):
    super().setUp()
    self.safety.set_safety_hooks(CarParams.SafetyModel.hondaBosch, 0)
    self.safety.init_tests()


class TestHondaBoschAltBrakeSafety(HondaPcmEnableBase, TestHondaBoschAltBrakeSafetyBase):
  """
    Covers the Honda Bosch safety mode with stock longitudinal and an alternate brake message
  """


class TestHondaBoschLongSafety(HondaButtonEnableBase, TestHondaBoschSafetyBase):
  """
    Covers the Honda Bosch safety mode with longitudinal control
  """
  NO_GAS = -30000
  MAX_GAS = 2000
  MAX_ACCEL = 2.0  # accel is used for brakes, but openpilot can set positive values
  MIN_ACCEL = -3.5

  STEER_BUS = 1
  TX_MSGS = [[0xE4, 1], [0x1DF, 1], [0x1EF, 1], [0x1FA, 1], [0x30C, 1], [0x33D, 1], [0x33DA, 1], [0x33DB, 1], [0x39F, 1], [0x18DAB0F1, 1]]
  FWD_BLACKLISTED_ADDRS = {}
  # 0x1DF is to test that radar is disabled
  RELAY_MALFUNCTION_ADDRS = {1: (0xE4, 0x1DF, 0x33D, 0x33DA, 0x33DB)}  # STEERING_CONTROL, ACC_CONTROL

  def setUp(self):
    super().setUp()
    self.safety.set_safety_hooks(CarParams.SafetyModel.hondaBosch, HondaSafetyFlags.BOSCH_LONG)
    self.safety.init_tests()

  def _send_gas_brake_msg(self, gas, accel):
    values = {
      "GAS_COMMAND": gas,
      "ACCEL_COMMAND": accel,
      "BRAKE_REQUEST": accel < 0,
    }
    return self.packer.make_can_msg_safety("ACC_CONTROL", self.PT_BUS, values)

  # Longitudinal doesn't need to send buttons
  def test_spam_cancel_safety_check(self):
    pass

  def test_diagnostics(self):
    tester_present = libsafety_py.make_CANPacket(0x18DAB0F1, self.PT_BUS, b"\x02\x3E\x80\x00\x00\x00\x00\x00")
    self.assertTrue(self._tx(tester_present))

    not_tester_present = libsafety_py.make_CANPacket(0x18DAB0F1, self.PT_BUS, b"\x03\xAA\xAA\x00\x00\x00\x00\x00")
    self.assertFalse(self._tx(not_tester_present))

  def test_gas_safety_check(self):
    for controls_allowed in [True, False]:
      for gas in np.arange(self.NO_GAS, self.MAX_GAS + 2000, 100):
        accel = 0 if gas < 0 else gas / 1000
        self.safety.set_controls_allowed(controls_allowed)
        send = (controls_allowed and 0 <= gas <= self.MAX_GAS) or gas == self.NO_GAS
        self.assertEqual(send, self._tx(self._send_gas_brake_msg(gas, accel)), (controls_allowed, gas, accel))

  def test_brake_safety_check(self):
    for controls_allowed in [True, False]:
      for accel in np.arange(self.MIN_ACCEL - 1, self.MAX_ACCEL + 1, 0.01):
        accel = round(accel, 2)  # floats might not hit exact boundary conditions without rounding
        self.safety.set_controls_allowed(controls_allowed)
        send = self.MIN_ACCEL <= accel <= self.MAX_ACCEL if controls_allowed else accel == 0
        self.assertEqual(send, self._tx(self._send_gas_brake_msg(self.NO_GAS, accel)), (controls_allowed, accel))


class TestHondaBoschRadarlessSafetyBase(TestHondaBoschSafetyBase):
  """Base class for radarless Honda Bosch"""
  PT_BUS = 0
  STEER_BUS = 0
  BUTTONS_BUS = 2  # camera controls ACC, need to send buttons on bus 2

  TX_MSGS = [[0xE4, 0], [0x296, 2], [0x33D, 0]]
  FWD_BLACKLISTED_ADDRS = {2: [0xE4, 0x33D]}
  RELAY_MALFUNCTION_ADDRS = {0: (0xE4, 0x33D)}  # STEERING_CONTROL

  def setUp(self):
    self.packer = CANPackerSafety("honda_bosch_radarless_generated")
    self.safety = libsafety_py.libsafety


class TestHondaBoschRadarlessSafety(HondaPcmEnableBase, TestHondaBoschRadarlessSafetyBase):
  """
    Covers the Honda Bosch Radarless safety mode with stock longitudinal
  """

  def setUp(self):
    super().setUp()
    self.safety.set_safety_hooks(CarParams.SafetyModel.hondaBosch, HondaSafetyFlags.RADARLESS)
    self.safety.init_tests()


class TestHondaBoschRadarlessAltBrakeSafety(HondaPcmEnableBase, TestHondaBoschRadarlessSafetyBase, TestHondaBoschAltBrakeSafetyBase):
  """
    Covers the Honda Bosch Radarless safety mode with stock longitudinal and an alternate brake message
  """

  def setUp(self):
    super().setUp()
    self.safety.set_safety_hooks(CarParams.SafetyModel.hondaBosch, HondaSafetyFlags.RADARLESS | HondaSafetyFlags.ALT_BRAKE)
    self.safety.init_tests()


class TestHondaBoschRadarlessLongSafety(common.LongitudinalAccelSafetyTest, HondaButtonEnableBase,
                                        TestHondaBoschRadarlessSafetyBase):
  """
    Covers the Honda Bosch Radarless safety mode with longitudinal control
  """
  TX_MSGS = [[0xE4, 0], [0x33D, 0], [0x1C8, 0], [0x30C, 0]]
  FWD_BLACKLISTED_ADDRS = {2: [0xE4, 0x33D, 0x1C8, 0x30C]}
  RELAY_MALFUNCTION_ADDRS = {0: (0xE4, 0x1C8, 0x30C, 0x33D)}

  def setUp(self):
    super().setUp()
    self.safety.set_safety_hooks(CarParams.SafetyModel.hondaBosch, HondaSafetyFlags.RADARLESS | HondaSafetyFlags.BOSCH_LONG)
    self.safety.init_tests()

  def _accel_msg(self, accel):
    values = {
      "ACCEL_COMMAND": accel,
    }
    return self.packer.make_can_msg_safety("ACC_CONTROL", self.PT_BUS, values)

  # Longitudinal doesn't need to send buttons
  def test_spam_cancel_safety_check(self):
    pass


class TestHondaBoschCANFDSafetyBase(TestHondaBoschSafetyBase):
  """Base class for CANFD Honda Bosch"""
  PT_BUS = 0
  STEER_BUS = 0
  BUTTONS_BUS = 0

  TX_MSGS = [[0xE4, 0], [0x296, 0], [0x33D, 0]]
  FWD_BLACKLISTED_ADDRS = {2: [0xE4, 0x33D]}
  RELAY_MALFUNCTION_ADDRS = {0: (0xE4, 0x33D)}

  def setUp(self):
    self.packer = CANPackerSafety("honda_common_canfd_generated")
    self.safety = libsafety_py.libsafety


class TestHondaBoschCANFDSafety(HondaPcmEnableBase, TestHondaBoschCANFDSafetyBase):
  """
    Covers the Honda Bosch CANFD safety mode with stock longitudinal
  """

  def setUp(self):
    super().setUp()
    self.safety.set_safety_hooks(CarParams.SafetyModel.hondaBosch, HondaSafetyFlags.BOSCH_CANFD)
    self.safety.init_tests()


class TestHondaBoschCANFDAltBrakeSafety(HondaPcmEnableBase, TestHondaBoschCANFDSafetyBase, TestHondaBoschAltBrakeSafetyBase):
  """
    Covers the Honda Bosch CANFD safety mode with stock longitudinal and an alternate brake message
  """

  def setUp(self):
    super().setUp()
    self.safety.set_safety_hooks(CarParams.SafetyModel.hondaBosch, HondaSafetyFlags.BOSCH_CANFD | HondaSafetyFlags.ALT_BRAKE)
    self.safety.init_tests()


class TestHondaNidecHybridSafety(TestHondaNidecPcmSafety):
  """
    Covers the Honda Nidec safety mode with hybrid brake
  """

  BRAKE_SIG = "COMPUTER_BRAKE_HYBRID"

  def setUp(self):
    self.packer = CANPackerSafety("honda_clarity_hybrid_2018_can_generated")
    self.safety = libsafety_py.libsafety
    self.safety.set_current_safety_param_sp(HondaSafetyFlagsSP.NIDEC_HYBRID)
    self.safety.set_safety_hooks(CarParams.SafetyModel.hondaNidec, 0)
    self.safety.init_tests()


if __name__ == "__main__":
  unittest.main()
