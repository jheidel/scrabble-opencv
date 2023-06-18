import cv2
import numpy as np

import configs
from .util import *


class Corners:
  """
  Maintains an averaged position of the 4 corners of the scrabble board.
  """

  def __init__(self):
    self.reset()

  def reset(self):
    self._corner_history = []

  def observe(self, pts):
    if len(pts) != 4:
      if configs.DEBUG_CORNERS:
        print("Expected 4 corners, saw %d: %s" % (len(pts), pts))
      return

    # Sort the corners into the appropriate order
    xy = list(sorted((p[0]+p[1], i) for i, p in enumerate(pts)))
    tl_i = xy[0][1]
    u1_i = xy[1][1]
    u2_i = xy[2][1]
    br_i = xy[3][1]
    tr_i = u1_i if pts[u1_i][0] > pts[u2_i][0] else u2_i
    bl_i = u1_i if tr_i != u1_i else u2_i
    tl = pts[tl_i]
    tr = pts[tr_i]
    br = pts[br_i]
    bl = pts[bl_i]

    #Check lengths to reject invalid corner layouts
    top_len = distance(tl, tr)
    left_len = distance(tl, bl)
    bottom_len = distance(bl, br)
    right_len = distance(tr, br)
    sides = np.array([top_len, left_len, bottom_len, right_len])
    side_dev = float(sides.std()) / sides.mean()
    if side_dev > configs.CORNER_SIDE_DEV_THRESH:
      if configs.DEBUG_CORNERS:
        print("Invalid board corners detected! (std of %.2f)" % side_dev)
      return

    new_corners = [tl, tr, br, bl]

    # Reject when corners move too far from established position
    if len(self._corner_history) > configs.CORNER_HISTORY_COUNT / 2:
      current_corners = self.get()
      distances = [distance(p1, p2) for p1, p2 in zip(new_corners, current_corners)]
      if max(distances) > configs.CORNER_MOVE_REJECT_THRESH:
        if configs.DEBUG_CORNERS:
          print("Rejecting corners due to excessive movement: %s" % distances)
        return

    self._corner_history.insert(0, new_corners)
    if len(self._corner_history) > configs.CORNER_HISTORY_COUNT:
      self._corner_history.pop()


  def get(self):
    """
    Get returns corners in the order [tl, tr, br, bl]
    """
    if not self._corner_history:
      return None

    corners = []
    for i in range(4):
      x = sum(p[i][0] for p in self._corner_history)
      y = sum(p[i][1] for p in self._corner_history)
      c = len(self._corner_history)
      corners.append((int(float(x)/c), int(float(y)/c)))

    return corners


class CornerFinder:
  """
  CornerFinder handles detection of the corners of the scrabble board.
  """
  def __init__(self):
    self.corners = Corners()

    self._erode_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (configs.CORNER_ERODE_RAD,configs.CORNER_ERODE_RAD))
    self._dilate_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (configs.CORNER_DILATE_RAD,configs.CORNER_DILATE_RAD))

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    self._aruco = cv2.aruco.ArucoDetector(aruco_dict)

  def reset(self):
    self.corners.reset()

  def process(self, frame):
    """
    Processes a given frame, returning the detected corners
    """

    if configs.DEBUG_CORNERS and configs.DEBUG_VERBOSE:
      POST("RAW", frame)

    draw = frame.copy()

    # TODO: clean up this method, refactor the detection methods (markers,
    # corners, border) into separate helpers.
    if configs.BOARD_MODE_MARKERS:

      (corners, ids, rejected) = self._aruco.detectMarkers(frame)
      cv2.aruco.drawDetectedMarkers(draw, corners, ids)

      tl, tr, bl, br = (None, None, None, None)
      for cid, c in zip(ids, corners):
        if cid == 10:
          tl = c[0][2]
        elif cid == 20:
          tr = c[0][3]
        elif cid == 30:
          bl = c[0][1]
        elif cid == 40:
          br = c[0][0]

      if all(x is not None for x in [tr, tl, br, bl]):
        corners = [tl, tr, br, bl]
        self.corners.observe(corners)

    else:
      luv = cv2.split(cv2.cvtColor(frame, cv2.COLOR_RGB2LUV))

      v_chan = luv[2]

      if configs.DEBUG_CORNERS and configs.DEBUG_VERBOSE:
        POST("V", v_chan)

      blur = cv2.GaussianBlur(v_chan, (configs.CORNER_BLUR_RAD,configs.CORNER_BLUR_RAD), 0)

      if configs.DEBUG_CORNERS and configs.DEBUG_VERBOSE:
        POST("blur", blur)
      thresh = cv2.adaptiveThreshold(blur, 255, 0, 1, configs.CORNER_THRESH_PARAM, configs.CORNER_BLOCK_SIZE)

      if configs.DEBUG_CORNERS:
        POST("thresh", thresh)

      erode = cv2.erode(thresh, self._erode_element)
      erode = cv2.dilate(erode, self._dilate_element)
      
      if configs.DEBUG_CORNERS:
        POST("erode", erode)
      
      contours, hierarchy = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      cv2.drawContours(draw, contours, -1, (255, 0, 0), 1)

      # Find a board identified by four red circles near the corners. Stickers
      # can be used to label the corners.
      if configs.BOARD_MODE_RED_CIRCLES:
        possible_corners = []

        #Find circle markers on board
        for cnt in contours:
          sz = cv2.contourArea(cnt)
          if sz>75 and sz < 1000:
            ellipse = cv2.fitEllipse(cnt)
            ((x,y), (w,h), r) = ellipse
            ar = w / h if w > h else h / w
            if ar > 1.8:
              continue
            pf = (w * h * 0.75) / sz
            if pf > 1.5:
              continue
            cv2.ellipse(draw,ellipse,(0,255,255),2)
            possible_corners.append((x,y))

        h, w, _ = frame.shape 
        c1 = min(possible_corners, key=lambda c: distance(c, [0,0]))
        c2 = min(possible_corners, key=lambda c: distance(c, [w,0]))
        c3 = min(possible_corners, key=lambda c: distance(c, [w,h]))
        c4 = min(possible_corners, key=lambda c: distance(c, [0,h]))

        corners = [c1, c2, c3, c4]
        for cr in corners:
          cv2.circle(draw, (int(cr[0]), int(cr[1])), 7, (0, 255, 0), thickness=1)

        cv2.ellipse(draw,ellipse,(0,255,255),2)
        self.corners.observe(corners)

      # Find a board identified by a red outline (newer scrabble boards have a
      # red border)
      elif configs.BOARD_MODE_RED_BORDER:
        cdebug = draw.copy()
        cv2.drawContours(cdebug, contours, -1, (0, 255, 0), 1)

        for cnt in contours:
          hull = cv2.convexHull(cnt)
          area = cv2.contourArea(hull)
          # Expect the board to take up at least 25% of the image.
          if area < configs.CAPTURE_WIDTH * configs.CAPTURE_HEIGHT * 0.25:
            continue

          cv2.drawContours(cdebug, [hull], -1, (255, 0, 0), 1)

          perimeter = cv2.arcLength(hull, True)
          approx = cv2.approxPolyDP(hull, perimeter * 0.02, True)

          cv2.drawContours(cdebug, [approx], -1, (0, 0, 255), 1)

          corners = [p[0] for p in approx]
          if corners:
            for cr in corners:
              cv2.circle(draw, (int(cr[0]), int(cr[1])), 7, (0, 255, 0), thickness=1)

          self.corners.observe(corners)

        if configs.DEBUG_CORNERS:
          POST("red_border_debug", cdebug)

      else:
        raise Exception("Must specify a board detection mode")

    corners = self.corners.get()
    if corners:
      for cr in corners:
        cv2.circle(draw, (int(cr[0]), int(cr[1])), 15, (0, 0, 255), thickness=3)

    POST("Corner Detection (R=average, G=instant)", draw)

    return corners
