"""FORK(BRAKE-SWITCH-MIRROR): the 0x17C BRAKE_SWITCH mirror to the Elesys radar.

openpilot brakes through 0x1FA and the BLR-v1 board lights the brake lamps off it, which leaves the car's own
POWERTRAIN_DATA (0x17C) saying BRAKE_SWITCH = 0 behind lit lamps - the pair the radar objects to. openpilot re-sends
0x17C to the radar on bus 2 with every signal copied and that one bit forced on while the board says it really is
driving the lamps. What this file holds:

  * THE GATE (brake_switch_mirror.brake_lamp_driven): True only while the board reports a fresh, driven, lit output,
    and fail-safe to False for every other input - no board, no out_sp, stale, not driving, lamp dark, coil open, a
    field of the wrong type, an object that raises on attribute access. It never raises;
  * THE FRAME: 0x17C on the bus it is given, every signal the car sent copied through, and BRAKE_SWITCH the only one
    that ever differs - with the gate False the frame is the car's own, signal for signal, BRAKE_SWITCH included,
    and BRAKE_PRESSED is never touched either way (the commented-out line in hondacan.py stays commented);
  * IT IS ON FOR THIS VARIANT IN SOFTWARE, with no toggle, like the stand-down and brake320: honda/interface.py sets
    the panda's safety param bit 256 for HONDA_ELESYS with openpilot longitudinal, stock ACC mode clears it back to
    68, no other Honda carries it, and brake_switch_mirror_on() is honda_nidec_init()'s rule to the bit - the
    CarController and the panda read the one param, so they cannot disagree about who owns the radar's 0x17C;
  * THE CONTROLLER: with the setting on, exactly one 0x17C on the radar's bus every frame (it is a 100 Hz frame and
    the panda blocks the car's own copy, so a gap is a gap at the radar), the forced bit following the board; with it
    off, not one 0x17C on either bus.

The panda's half - the TX allowance and the forward block - is pinned by safety/tests/test_honda.py
(TestHondaElesysBrakeSwitchMirror*), which needs a built libsafety.
"""
import math
import unittest

from opendbc.can import CANPacker, CANParser
from opendbc.car import Bus, gen_empty_fingerprint, structs
from opendbc.car.honda import hondacan
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR, DBC, HONDA_ELESYS, HondaSafetyFlags
from opendbc.sunnypilot.car.honda.brake_switch_mirror import NO_LIGHT_MASK, brake_lamp_driven, brake_switch_mirror_on
from opendbc.sunnypilot.car.interfaces import setup_interfaces

ELESYS = CAR.HONDA_ACCORD_9G_AU
CAR_DBC = DBC[ELESYS][Bus.pt]
STOCK_ACC = "HondaElesysStockAcc"
F = HondaSafetyFlags
RADAR_BUS = 2  # on this harness bus 2 is the Elesys radar branch (hondacan.CanBus.camera)

# A 0x17C with something in every signal, so "every other bit is copied" means something. BRAKE_SWITCH is 0 here:
# that is the case the mirror exists for, openpilot braking with the driver's foot off the pedal.
CAR_FRAME = {
  "PEDAL_GAS": 42,
  "ENGINE_RPM": 2100,
  "GAS_PRESSED": 1,
  "ACC_STATUS": 1,
  "BOH_17C": 5,
  "BRAKE_SWITCH": 0,
  "BOH2_17C": 300,
  "BRAKE_PRESSED": 0,
  "BOH3_17C": 9,
}


def _blr(**kwargs):
  """A CarState stand-in carrying a brakeLampRelay, as CarController sees it (CS.out_sp)."""
  blr = structs.CarStateSP.BrakeLampRelay()
  for k, v in kwargs.items():
    setattr(blr, k, v)
  sp = structs.CarStateSP()
  sp.brakeLampRelay = blr
  return type("CS", (), {"out_sp": sp})()


def _decode(msg):
  """The mirror frame's signals, decoded with the car's own DBC."""
  addr, dat, bus = msg
  cp = CANParser(CAR_DBC, [("POWERTRAIN_DATA", math.nan)], bus)
  cp.update([(10**9, [msg])])
  assert addr == 0x17C, hex(addr)
  assert len(dat) == 8, len(dat)
  return dict(cp.vl["POWERTRAIN_DATA"])


class TestTheGate(unittest.TestCase):
  """Only a board reporting a fresh, driven, lit output may claim the brake switch. Everything else is False."""

  def test_a_driven_lit_board_is_the_one_true_case(self):
    self.assertTrue(brake_lamp_driven(_blr(valid=True, outDrive=True)))

  def test_every_other_board_state_is_false(self):
    for name, cs in (
      ("no board at all", _blr()),
      ("stale status", _blr(valid=False, outDrive=True)),
      ("not driving", _blr(valid=True, outDrive=False)),
      ("driving nothing: the lamp sense is dark", _blr(valid=True, outDrive=True, faults=NO_LIGHT_MASK)),
      ("wanted on but deferred", _blr(valid=True, outDrive=False, wouldOn=True, request=True)),
      ("asked for by openpilot, not delivered", _blr(valid=True, outDrive=False, request=True, cmdFresh=True)),
    ):
      self.assertFalse(brake_lamp_driven(cs), name)

  def test_each_no_light_fault_alone_is_enough(self):
    bit = 1
    while bit <= NO_LIGHT_MASK:
      if bit & NO_LIGHT_MASK:
        self.assertFalse(brake_lamp_driven(_blr(valid=True, outDrive=True, faults=bit)), hex(bit))
      bit <<= 1

  def test_the_other_faults_do_not_hide_a_lit_lamp(self):
    # a trip or a diagnosis that does not mean "dark" leaves the claim alone: the board is still driving the lamps
    others = ~NO_LIGHT_MASK & 0xFF
    self.assertTrue(brake_lamp_driven(_blr(valid=True, outDrive=True, faults=others)))

  def test_it_never_raises(self):
    class Hostile:
      def __getattr__(self, name):
        raise RuntimeError(name)

    nonsense = object()
    for cs in (None, nonsense, Hostile(), type("CS", (), {"out_sp": None})(), type("CS", (), {"out_sp": nonsense})(),
               _blr(valid=True, outDrive=True, faults="not a number")):
      self.assertFalse(brake_lamp_driven(cs), repr(cs))


class TestTheFrame(unittest.TestCase):
  """Copy every bit, change the one."""

  def setUp(self):
    self.packer = CANPacker(CAR_DBC)

  def _mirror(self, on):
    return _decode(hondacan.create_brake_switch_on(self.packer, RADAR_BUS, dict(CAR_FRAME), on))

  def test_the_gate_off_is_the_cars_own_frame(self):
    got = self._mirror(False)
    for sig, want in CAR_FRAME.items():
      self.assertAlmostEqual(got[sig], want, msg=sig)

  def test_the_gate_on_changes_brake_switch_and_nothing_else(self):
    got = self._mirror(True)
    self.assertEqual(got["BRAKE_SWITCH"], 1)
    for sig, want in CAR_FRAME.items():
      if sig == "BRAKE_SWITCH":
        continue
      self.assertAlmostEqual(got[sig], want, msg=sig)

  def test_brake_pressed_is_never_forced(self):
    # the BRAKE_PRESSED line in hondacan.py is commented out on purpose: BRAKE_SWITCH alone first
    for on in (False, True):
      self.assertEqual(self._mirror(on)["BRAKE_PRESSED"], 0, f"{on=}")

  def test_a_pressed_pedal_is_copied_through_whatever_the_board_says(self):
    # the driver's own brake: the bit is already 1 on the wire and the mirror must not disturb the rest
    pressed = dict(CAR_FRAME, BRAKE_SWITCH=1, BRAKE_PRESSED=1)
    for on in (False, True):
      got = _decode(hondacan.create_brake_switch_on(self.packer, RADAR_BUS, dict(pressed), on))
      for sig, want in pressed.items():
        self.assertAlmostEqual(got[sig], want, msg=f"{sig} {on=}")

  def test_the_counter_and_checksum_are_the_packers(self):
    # they are dropped from the copy, so a stale COUNTER from the car's frame can never go out
    stale = dict(CAR_FRAME, COUNTER=3, CHECKSUM=15)
    seen = set()
    for _ in range(8):
      got = _decode(hondacan.create_brake_switch_on(self.packer, RADAR_BUS, dict(stale), True))
      seen.add(got["COUNTER"])
    self.assertEqual(seen, {0, 1, 2, 3}, "the packer's own counter must advance")

  def test_the_address_and_the_bus(self):
    addr, _, bus = hondacan.create_brake_switch_on(self.packer, RADAR_BUS, dict(CAR_FRAME), True)
    self.assertEqual(addr, 0x17C)
    self.assertEqual(bus, RADAR_BUS)


def _params(car=ELESYS, pedal=True, params_list=None, hook=True):
  """car_helpers.get_car's order: get_params, get_params_sp, then the sunnypilot hooks."""
  fp = gen_empty_fingerprint()
  fp[0][0x188] = 8          # GEARBOX_AUTO, as on the car
  if pedal:
    fp[0][0x201] = 6        # GAS_SENSOR: the comma pedal
  CP = CarInterface.get_params(car, fp, [], False, False, False)
  CP_SP = CarInterface.get_params_sp(CP, car, fp, [], False, False, False)
  if hook:
    setup_interfaces(CarInterface, CP, CP_SP, params_list)
  return CP, CP_SP


class TestItIsOnForThisVariant(unittest.TestCase):
  """On in software, with no toggle, like the stand-down and brake320. One source of truth: the safety param the
  panda gets, which the CarController reads back through brake_switch_mirror_on()."""

  def test_this_car_with_openpilot_long_has_it(self):
    for pedal in (True, False):
      CP, _ = _params(pedal=pedal)
      self.assertTrue(CP.openpilotLongitudinalControl)
      self.assertEqual(CP.safetyConfigs[-1].safetyParam,
                       F.NIDEC_ALT | F.ELESYS_SCM_STANDDOWN | F.ELESYS_BRAKE_320 | F.ELESYS_BRAKE_SWITCH_MIRROR)
      self.assertEqual(CP.safetyConfigs[-1].safetyParam, 420, "164 before the mirror, 420 with it")
      self.assertTrue(brake_switch_mirror_on(CP))

  def test_no_param_turns_it_on_or_off(self):
    # there is no key for it: whatever card passes, this car gets the mirror and no other Honda does
    base = _params()[0].to_bytes()
    for params_list in (None, [], [{"HondaElesysBrakeSwitchMirror": False}], [{"HondaElesysPumpC1b": True}]):
      CP, _ = _params(params_list=params_list)
      self.assertTrue(brake_switch_mirror_on(CP), msg=str(params_list))
      if params_list in (None, [], [{"HondaElesysBrakeSwitchMirror": False}]):
        self.assertEqual(CP.to_bytes(), base, msg=str(params_list))

  def test_stock_acc_mode_clears_it(self):
    CP, CP_SP = _params(params_list=[{STOCK_ACC: "1"}])
    self.assertEqual(CP.safetyConfigs[-1].safetyParam, 68, "68 as before the mirror existed")
    self.assertEqual(CP.safetyConfigs[-1].safetyParam & F.ELESYS_BRAKE_SWITCH_MIRROR, 0)
    self.assertFalse(brake_switch_mirror_on(CP))

  def test_the_port_sets_it_with_the_stand_down_and_not_otherwise(self):
    # the port's condition for the two bits is the same one, so they arrive together or not at all
    CP, _ = _params()
    p = CP.safetyConfigs[-1].safetyParam
    self.assertEqual(bool(p & F.ELESYS_BRAKE_SWITCH_MIRROR), bool(p & F.ELESYS_SCM_STANDDOWN))
    # and the reader refuses it without the stand-down, as the panda does
    CP.safetyConfigs[-1].safetyParam = p & ~F.ELESYS_SCM_STANDDOWN.value
    self.assertFalse(brake_switch_mirror_on(CP))

  def test_no_other_honda_carries_it(self):
    for car in CAR:
      if car in HONDA_ELESYS:
        continue
      for pedal in (False, True):
        CP, _ = _params(car=car, pedal=pedal)
        for cfg in CP.safetyConfigs:
          self.assertFalse(cfg.safetyParam & F.ELESYS_BRAKE_SWITCH_MIRROR, (car, pedal))
        self.assertFalse(brake_switch_mirror_on(CP), car)

  def test_the_reader_is_hondas_rule(self):
    # brake_switch_mirror_on() must be honda_nidec_init()'s rule exactly, or the CarController and the panda disagree
    # about who owns the radar's 0x17C
    cfg = structs.CarParams.SafetyConfig.new_message()
    cp = structs.CarParams.new_message()
    for model in (structs.CarParams.SafetyModel.hondaNidec, structs.CarParams.SafetyModel.hondaBosch):
      for param in range(512):
        cfg.safetyModel = model
        cfg.safetyParam = param
        cp.safetyConfigs = [cfg]
        on = (model == structs.CarParams.SafetyModel.hondaNidec and bool(param & 256) and bool(param & 32)
              and not (param & 64))
        self.assertEqual(brake_switch_mirror_on(cp), on, (model, param))
    # two configs: every one of them, or off
    a = structs.CarParams.SafetyConfig.new_message(safetyModel=structs.CarParams.SafetyModel.hondaNidec, safetyParam=420)
    b = structs.CarParams.SafetyConfig.new_message(safetyModel=structs.CarParams.SafetyModel.noOutput)
    cp.safetyConfigs = [b, a]
    self.assertFalse(brake_switch_mirror_on(cp))
    cp.safetyConfigs = [a]
    self.assertTrue(brake_switch_mirror_on(cp))
    for odd in (None, object(), structs.CarParams.new_message()):
      self.assertFalse(brake_switch_mirror_on(odd), repr(odd))


class _CS:
  """What CarController.update() reads from CS (test_elesys's _NowCS, plus the mirror's two)."""

  def __init__(self, lamp_driven=False, seen=True):
    self.powertrain_data_seen = seen
    self.out = structs.CarState.new_message()
    self.out.vEgo = 16.0
    self.out.cruiseState.available = True
    self.out.cruiseState.speed = 16.0
    self.v_cruise_factor = 1.0
    self.stock_brake = {"CHIME": 0, "AEB_REQ_1": 0, "AEB_REQ_2": 0, "AEB_STATUS": 0}
    self.acc_hud = {"FCM_OFF": 0, "FCM_OFF_2": 0, "FCM_PROBLEM": 0, "ICONS": 0}
    self.lkas_hud = {}
    self.scm_buttons = {"CRUISE_BUTTONS": 0, "CRUISE_SETTING": 0, "MAIN_ON": 1}
    self.powertrain_data = dict(CAR_FRAME)
    self.is_metric = True
    self.econ_on = False
    self.out_sp = structs.CarStateSP()
    self.out_sp.brakeLampRelay = structs.CarStateSP.BrakeLampRelay(valid=lamp_driven, outDrive=lamp_driven)


def _run(mirror, frames=10, lamp_driven=False, seen=True):
  """Every frame's sends with the mirror on (this car) or off (the bit cleared, as every other config has it)."""
  CP, CP_SP = _params()
  if not mirror:
    CP.safetyConfigs[-1].safetyParam &= ~F.ELESYS_BRAKE_SWITCH_MIRROR.value
  assert brake_switch_mirror_on(CP) == mirror
  CI = CarInterface(CP, CP_SP)
  cs = _CS(lamp_driven, seen)
  out = []
  for i in range(frames):
    cc = structs.CarControl.new_message()
    cc.enabled = cc.latActive = cc.longActive = True
    cc.actuators.accel = -1.0
    cc.actuators.torque = 0.1
    _, sends = CI.CC.update(cc.as_reader(), structs.CarControlSP(), cs, i * int(1e7))
    out.append(list(sends))
  return CP, out


class TestTheController(unittest.TestCase):

  def _mirrors(self, sends):
    return [m for frame in sends for m in frame if m[0] == 0x17C]

  def test_off_sends_no_17c_at_all(self):
    _, sends = _run(False)
    self.assertEqual(self._mirrors(sends), [], "0x17C must not be sent with the setting off")

  def test_on_sends_one_per_frame_on_the_radars_bus(self):
    CP, sends = _run(True, frames=10)
    self.assertEqual(hondacan.CanBus(CP).camera, RADAR_BUS, "bus 2 is the radar branch on this harness")
    for i, frame in enumerate(sends):
      self.assertEqual(len([m for m in frame if m[0] == 0x17C]), 1, f"frame {i}")
      self.assertEqual([m[2] for m in frame if m[0] == 0x17C], [RADAR_BUS], f"frame {i}")

  def test_nothing_before_the_cars_own_17c_is_decoded(self):
    # cp.vl is zeros until the parser validates a 0x17C, and an all-zero POWERTRAIN_DATA is engine-off to the radar
    _, sends = _run(True, frames=4, seen=False)
    self.assertEqual(self._mirrors(sends), [])

  def test_the_forced_bit_follows_the_board(self):
    for lamp_driven in (False, True):
      _, sends = _run(True, frames=4, lamp_driven=lamp_driven)
      for msg in self._mirrors(sends):
        self.assertEqual(_decode(msg)["BRAKE_SWITCH"], int(lamp_driven), f"{lamp_driven=}")

  def test_nothing_else_changes_on_the_wire(self):
    # the mirror is an addition: every other frame openpilot sends is the same with the setting on and off
    _, off = _run(False, frames=6)
    _, on = _run(True, frames=6)
    for i, (a, b) in enumerate(zip(off, on, strict=True)):
      self.assertEqual([m for m in a if m[0] != 0x17C], [m for m in b if m[0] != 0x17C], f"frame {i}")


if __name__ == "__main__":
  unittest.main()
