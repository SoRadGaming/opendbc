#!/usr/bin/env python3
from opendbc.can import CANParser
from opendbc.car import Bus, structs
from opendbc.car.interfaces import RadarInterfaceBase
from opendbc.car.honda.values import DBC, HONDA_ELESYS

# FORK(HONDA_ACCORD_9G_AU): the Elesys radar's track slots (honda_accord_2015au_radar.dbc; sunnypilot
# docs/fork/CAR-HONDA-ACCORD-9G-AU.md 5.2). Measured on the logged bus 1, 2026-10-24:
#  * a radar cycle is 0x410-0x417, 0x420-0x424, then 0x4FF, 0x300, 0x301 and 0x400, every 0.1 s. 0x424 is the last
#    track frame of every cycle the log holds whole (1.25 M cycles, 126 routes; never before 0x423) and arrives in the
#    CAN batch after 0x423 in 2.5% of them, so the update waits for 0x424 (it fired on 0x423, which left slot 0x424 one
#    cycle stale). 0x400 still comes after it and is read one cycle late, as before. A cycle whose 0x424 never comes
#    (seen only where the radar stopped at the end of a drive) is not published on its own: the next cycle's frames
#    overwrite it and publish on their 0x424;
#  * LONG_DIST raw 0x7FFF (255.9921875 m) is an empty slot, the only empty pattern in 9.9 M frames. Real returns reach
#    255.8 m, so the old test LONG_DIST < 255 dropped returns at 255.0-255.98 m;
#  * NEW_TRACK: the radar has put a new object in the slot. Set on two frames in a row (1,467 times on those 6 routes)
#    the second frame is ANOTHER new object 92% of the time (its range or lateral position 1 m or more off the first's
#    path, 72% by 10 m or more; the frame after it continues the second object), so a new trackId on a rising edge
#    only would join two objects under one id. A held NEW_TRACK on a frame that continues the slot's object (within
#    ELESYS_SAME_*, which 94.5-97.5% of frames without NEW_TRACK meet, and 1.4-5.5% of rising ones) keeps its id. A
#    RISING NEW_TRACK is a new id even when its frame meets ELESYS_SAME_* (2.4% of rising edges on an occupied slot),
#    as upstream's every-NEW_TRACK rule gave it: the check is for held flags only;
#  * COASTING (byte 2 bit 6, FLAG_B22 until 2026-10-24): the radar extrapolated the track this cycle instead of
#    measuring it. RadarPoint's measured flag (deprecated.measured since upstream's #3546, the same field @6) is its
#    inverse, so radarTracks in a route carries it. Nothing reads it: radard, the clutter guard and the UI do not.
ELESYS_TRIGGER = 0x424
ELESYS_LONG_DIST_EMPTY = 0x7FFF / 128  # m, raw 0x7FFF: exact in a float, as the DBC factor is a power of two
ELESYS_DT = 0.1                        # s, the radar's cycle
ELESYS_SAME_D = 1.0  # m: a held NEW_TRACK whose range is within this of the last frame's range moved on at its vRel,
ELESYS_SAME_Y = 1.0  # m: whose lateral position is within this of the last frame's,
ELESYS_SAME_V = 2.0  # m/s: and whose vRel is within this of the last frame's, is the same object


def elesys_same_object(pt, cpt) -> bool:
  """FORK(HONDA_ACCORD_9G_AU): does this frame (cpt) continue the object the slot's point (pt) holds?"""
  return (abs(cpt['LONG_DIST'] - (pt.dRel + pt.vRel * ELESYS_DT)) < ELESYS_SAME_D and
          abs(-cpt['LAT_DIST'] - pt.yRel) < ELESYS_SAME_Y and abs(cpt['REL_SPEED'] - pt.vRel) < ELESYS_SAME_V)


def _create_nidec_can_parser(car_fingerprint):
  if car_fingerprint in HONDA_ELESYS:
    radar_messages = [0x400] + list(range(0x410, 0x418)) + list(range(0x420, 0x425))
    messages = [(m, 10) for m in radar_messages]
  else:
    radar_messages = [0x400] + list(range(0x430, 0x43A)) + list(range(0x440, 0x446))
    messages = [(m, 20) for m in radar_messages]
  return CANParser(DBC[car_fingerprint][Bus.radar], messages, 1)


class RadarInterface(RadarInterfaceBase):
  def __init__(self, CP, CP_SP):
    super().__init__(CP, CP_SP)
    self.radar_fault = False
    self.radar_wrong_config = False
    self.radar_off_can = CP.radarUnavailable
    self.radar_type = 'Elesys' if CP.carFingerprint in HONDA_ELESYS else 'Nidec'

    # Nidec
    if self.radar_off_can:
      self.rcp = None
    else:
      self.rcp = _create_nidec_can_parser(CP.carFingerprint)
    self.trigger_msg = ELESYS_TRIGGER if (self.radar_type == 'Elesys') else 0x445  # FORK(HONDA_ACCORD_9G_AU): was 0x423
    self.updated_messages = set()
    self.new_track_last: dict[int, bool] = {}  # FORK(HONDA_ACCORD_9G_AU): each Elesys slot's NEW_TRACK on its last frame

  def update(self, can_strings):
    # in Bosch radar and we are only steering for now, so sleep 0.05s to keep
    # radard at 20Hz and return no points
    if self.radar_off_can:
      return super().update(None)

    vls = self.rcp.update(can_strings)
    self.updated_messages.update(vls)

    if self.trigger_msg not in self.updated_messages:
      return None

    rr = self._update(self.updated_messages)
    self.updated_messages.clear()
    return rr

  def _update(self, updated_messages):
    ret = structs.RadarData()

    for ii in sorted(updated_messages):
      cpt = self.rcp.vl[ii]
      if ii == 0x400:
        # check for radar faults
        if self.radar_type == 'Elesys':
          self.radar_fault = cpt['RADAR_STATE'] not in (104, 111, 125)
        else:
          self.radar_fault = cpt['RADAR_STATE'] != 0x79
        self.radar_wrong_config = cpt['RADAR_STATE'] == 0x69
      elif self.radar_type == 'Elesys':  # FORK(HONDA_ACCORD_9G_AU)
        self._update_elesys_track(ii, cpt)
      elif cpt['LONG_DIST'] < 255:
        if ii not in self.pts or cpt['NEW_TRACK']:
          self.pts[ii] = structs.RadarData.RadarPoint()
          self.pts[ii].trackId = self.track_id
          self.track_id += 1
        self.pts[ii].dRel = cpt['LONG_DIST']  # from front of car
        self.pts[ii].yRel = -cpt['LAT_DIST']  # in car frame's y axis, left is positive
        self.pts[ii].vRel = cpt['REL_SPEED']
      else:
        if ii in self.pts:
          del self.pts[ii]

    if not self.rcp.can_valid:
      ret.errors.canError = True
    if self.radar_fault:
      ret.errors.radarFault = True
    if self.radar_wrong_config:
      ret.errors.wrongConfig = True

    ret.points = list(self.pts.values())

    return ret

  def _update_elesys_track(self, ii, cpt):
    """FORK(HONDA_ACCORD_9G_AU): one Elesys track slot (see the notes at the top). Empty only at raw 0x7FFF. A new
    trackId when the slot was empty, when NEW_TRACK rises, or when NEW_TRACK stays set on a frame that does not
    continue the slot's object."""
    if cpt['LONG_DIST'] == ELESYS_LONG_DIST_EMPTY:
      self.pts.pop(ii, None)
      self.new_track_last.pop(ii, None)
      return
    new_track = bool(cpt['NEW_TRACK'])
    pt = self.pts.get(ii)
    if pt is None or (new_track and not (self.new_track_last.get(ii, False) and elesys_same_object(pt, cpt))):
      pt = self.pts[ii] = structs.RadarData.RadarPoint()
      pt.trackId = self.track_id
      self.track_id += 1
    self.new_track_last[ii] = new_track
    pt.dRel = cpt['LONG_DIST']  # from front of car
    pt.yRel = -cpt['LAT_DIST']  # in car frame's y axis, left is positive
    pt.vRel = cpt['REL_SPEED']
    pt.deprecated.measured = not cpt['COASTING']
