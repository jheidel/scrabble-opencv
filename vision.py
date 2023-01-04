import cv2
import numpy as np
from math import *
import sys
import gc
from scipy.interpolate import griddata
import configs
from threading import Thread, Lock
from board import Board, AveragedBoard
import traceback
import importlib
import time

def POST(name, img):
  cv2.namedWindow(name)
  cv2.imshow(name, img)


def distance(p1, p2):
  (x1,y1) = p1
  (x2,y2) = p2
  return sqrt((x1-x2)**2 + (y1-y2)**2)


def get_center(x, y):
  return (int(configs.LETTER_SIZE * configs.LETTER_PAD_FRAC) + int(configs.LETTER_SIZE/2) + configs.LETTER_SIZE * x, int(configs.LETTER_SIZE * configs.LETTER_PAD_FRAC) + int(configs.LETTER_SIZE/2) + configs.LETTER_SIZE * y)


def get_bounding_rect(x, y):
  start = (int(configs.LETTER_SIZE * configs.LETTER_PAD_FRAC) + configs.LETTER_SIZE * x, int(configs.LETTER_SIZE * configs.LETTER_PAD_FRAC) + configs.LETTER_SIZE * y)
  end = (start[0] + configs.LETTER_SIZE, start[1] + configs.LETTER_SIZE)
  return [start, end]


def get_centroid_rect(c, frac):
  w = int(configs.LETTER_SIZE * frac)
  start = (c[0] - int(w/2), c[1] - int(w/2))
  end = (start[0] + w, start[1] + w)
  return [start, end]


def get_board_size():
  return int(configs.LETTER_SIZE*configs.BOARD_SIZE + configs.LETTER_SIZE * (2 * configs.LETTER_PAD_FRAC))


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

  def reset(self):
    self.corners.reset()

  def process(self, frame):
    """
    Processes a given frame, returning the detected corners
    """

    if configs.DEBUG_CORNERS and configs.DEBUG_VERBOSE:
      POST("RAW", frame)

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

    draw = frame.copy()
    
    contours, hierarchy = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(draw, contours, -1, (255, 0, 0), 1)

    # Find a board identified by four red circles near the corners. Stickers
    # can be used to label the corners.
    if configs.BOARD_MODE_RED_CIRCLES:
      possible_corners = []

      #Find circle markers on board
      for cnt in contours:
        sz = cv2.contourArea(cnt)
        if sz>75 and sz < 650:
          ellipse = cv2.fitEllipse(cnt)
          ((x,y), (w,h), r) = ellipse
          ar = w / h if w > h else h / w
          if ar > 1.8:
            continue
          pf = (w * h * 0.75) / sz
          if pf > 1.5:
            continue
          cv2.ellipse(draw,ellipse,(0,255,0),2)
          possible_corners.append((sz, (x,y)))

      possible_corners.sort(reverse=True)
      self.corners.observe([p[1] for p in possible_corners[:4]])

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


class LetterModel:
  def __init__(self):
    self._responses = None
    self._samples = None
    self._model = None

  def load(self):

    # TODO: we should save the actual binary training images to make it clearer
    # what's in the dataset and to maybe experiment with other models.

    if configs.TRAIN:
      print("Training mode!")
      if not configs.TRAIN_APPEND:
        self._responses = []
        self._samples = np.empty((0,configs.LETTER_TRAIN_SIZE**2))
      else:
        self._samples = np.loadtxt('generalsamples.data',np.float32)
        self._responses = np.loadtxt('generalresponses.data',np.float32)
        self._responses = self._responses.reshape((self._responses.size,1))
        self._responses = list(map(lambda x: x[0], list(self._responses)))
    else:
      print("Loading trained data")
      
      self._samples = np.loadtxt('generalsamples.data',np.float32)
      self._responses = np.loadtxt('generalresponses.data',np.float32)
      self._responses = self._responses.reshape((self._responses.size,1))

      print("Training model")

      self._model = cv2.ml.KNearest_create()
      self._model.train(self._samples, cv2.ml.ROW_SAMPLE, self._responses)

      print("Model trained")

  def classify_all(self, finder):
    letters = Board()
    for j in range(0,configs.BOARD_SIZE):
      for i in range(0,configs.BOARD_SIZE):
        letter_thresh_image = finder.get_thresh(i, j)
        r = self.classify_letter(letter_thresh_image)
        if r is not None:
          letters.set(i, j, r)
    return letters

  def classify_letter(self, thresh_image):
    if thresh_image is None:
      return None

    shrunk = cv2.resize(thresh_image, (configs.LETTER_TRAIN_SIZE, configs.LETTER_TRAIN_SIZE))
    
    sample = shrunk.reshape((1,configs.LETTER_TRAIN_SIZE**2))

    if configs.TRAIN:
      print("Which letter is this? (use '0' for star, enter to stop, esc to skip)")
      POST("Which letter is this? (TRAINING)", shrunk)
      o = cv2.waitKey(0)
      if o == 27:
        print("Skipping, here's another...")
      elif chr(o).lower() >= 'a' and chr(o).lower() <= 'z' or chr(o).lower() == '0':
        x = chr(o).lower()
        print("You said it's a %s" % str(x))
        self._responses.append(ord(x)-96)
        self._samples = np.append(self._samples, sample, 0)
        print("Added to sample set")
      else:
        self._responses = np.array(self._responses,np.float32)
        self._responses = self._responses.reshape((self._responses.size,1))
        print("training complete")

        np.savetxt('generalsamples.data', self._samples)
        np.savetxt('generalresponses.data', self._responses)
        sys.exit(0)
    else:

      #classify!
      sample = np.float32(sample)
      retval, results, neigh_resp, dists = self._model.findNearest(sample, k = 1)
      retchar = chr(int((results[0][0])) + 96)
      if retchar == '0':
        #Star character!
        return None
      return retchar


class LetterFinder:
  """
  LetterFinder locates letter shapes on the board.
  """

  def __init__(self):
    self.image = None
    self.thresh = None
    self.board_centroids = [[None for i in range(configs.BOARD_SIZE)] for j in range(configs.BOARD_SIZE)]

  def get_thresh(self, x, y):
    """
    Gets a binary threshold image for a specific board coordinate

    Returns None if no letter is detected at that location.
    """
    centroid = self.board_centroids[x][y]
    if not centroid:
      return None

    w = int(configs.LETTER_SIZE * configs.LETTER_TRAIN_SUBPIX_FRAC)
    return cv2.getRectSubPix(self.thresh, [w, w], centroid)


  def process(self, image):
    """
    Processes the provided image, identifying the locations for all letters on the board.

    Once processed, the binary thresholded image for a specific letter can be
    retrieved using `get_thresh`.
    """
    h, w, _ = image.shape
    sz = get_board_size()
    assert w == sz and h == sz

    self.image = image
    draw = self.image.copy()

    split = cv2.split(cv2.cvtColor(self.image, configs.LETTER_COLORSPACE))

    channel = split[configs.LETTER_CHANNEL]

    if configs.DEBUG_LETTERS and configs.DEBUG_VERBOSE:
      POST("letters channel", channel)

    if configs.LETTER_BLUR:
      channel = cv2.GaussianBlur(channel, (configs.LETTER_BLUR,configs.LETTER_BLUR), 0)

    self.thresh = cv2.adaptiveThreshold(channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, configs.LETTER_BLOCK, configs.LETTER_THRESH)
   
    # TODO: add dilation to filter chain?
    #element = cv2.getStructuringElement(cv2.MORPH_CROSS, (7,7))
    #self.thresh = cv2.dilate(self.thresh, element)

    if configs.DEBUG_LETTERS:
      POST("letters thresh", self.thresh)

    contours, hierarchy = cv2.findContours(self.thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(draw, contours, -1, (255, 0, 0), 1)

    hulls = []
    areas = []
    possible_contour_indexes = []

    for i, c in enumerate(contours):
      hull = cv2.convexHull(c)
      hulls.append(hull)

      a = cv2.contourArea(hull)
      areas.append(a)

      # Check whether contour area is reasonable for a letter.
      if a < int(configs.LETTER_SIZE**2 * configs.LETTER_CONTOUR_MIN_FRAC):
        continue
      if a > int(configs.LETTER_SIZE**2 * configs.LETTER_CONTOUR_MAX_FRAC):
        continue

      [x,y,w,h] = cv2.boundingRect(c)
      
      if w > h*configs.LETTER_TEXT_RATIO:
        # Bad ratio, reject these.
        continue

      if w*h >= configs.LETTER_SIZE**2 * configs.LETTER_MAX_FILL:
        # Too much fill, reject these.
        continue


      possible_contour_indexes.append(i)

    # Remove valid children that are inside valid parents so we only have one valid contour.
    possible_contour_indexes = [p for p in possible_contour_indexes if not hierarchy[0][p][3] in possible_contour_indexes]

    contour_centroids = []

    for i in possible_contour_indexes:
      hull = hulls[i]
      a = areas[i]

      moments = cv2.moments(hull)
      hull_centroid = (int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00']))

      contour_centroids.append((hull_centroid, i))

      cv2.drawContours(draw, [hull], -1, (0, 0, 255), 2)
      cv2.circle(draw, hull_centroid, 2, (0,0,255), thickness=3)

      cv2.putText(draw, "A%d" % (a) , hull_centroid, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255))

    if not contour_centroids:
      return

    for i in range(0, configs.BOARD_SIZE):
      for j in range(0, configs.BOARD_SIZE):
        p = get_center(i, j)
        cv2.circle(draw, p, 2, (0,255,0), thickness=3)

        r = get_bounding_rect(i, j)
        cv2.rectangle(draw, r[0], r[1], (0, 255, 0), 1)

        (centroid, contour_i) = min(contour_centroids, key=lambda k: distance(k[0], p))
        d = distance(centroid, p)

        if d > configs.LETTER_SIZE * configs.LETTER_MAX_SHIFT_FRAC:
          continue

        # Letter detection
        cv2.line(draw, p, centroid, (0, 255, 255), 1)
        cv2.putText(draw, "D%d" % (d) , p, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,255))

        rc = get_centroid_rect(centroid, configs.LETTER_TRAIN_SUBPIX_FRAC)
        cv2.rectangle(draw, rc[0], rc[1], (0, 255, 255), 1)

        self.board_centroids[i][j] = centroid

    if configs.DEBUG_LETTERS:
      # TODO: it's pretty hard to read this. Scale this up so it's easier to
      # read areas and distances. Also woudl be useful to add aspect ratios for
      # easier contour filtering.
      POST("letters detect", draw)


class BlankFinder:
  def __init__(self):
    pass

  def process(self, image, letters):
    h, w, _ = image.shape
    sz = get_board_size()
    assert w == sz and h == sz

    channel = cv2.cvtColor(image, configs.BLANK_COLORSPACE)
    if configs.BLANK_CHANNEL is not None:
      channel = cv2.split(channel)[configs.BLANK_CHANNEL]

    if configs.DEBUG_BLANKS and configs.DEBUG_VERBOSE:
      POST("blank channel", channel)

    channel = cv2.GaussianBlur(channel, (3,3), 0)
    draw = cv2.cvtColor(channel, cv2.COLOR_GRAY2RGB)

    possible_blanks = Board()

    # STAGE 1: we identify all blank squares on the board (including empty
    # background squares)
    for i in range(configs.BOARD_SIZE):
      for j in range(configs.BOARD_SIZE):

        if letters.get(i, j) is not None:
          # Ignore any position that already has a letter classification.
          continue

        center = get_center(i, j)
        rect = get_centroid_rect(center, configs.BLANK_PATCH_FRAC)
        cv2.rectangle(draw, rect[0], rect[1], (0, 255, 255), 1)

        w = int(configs.LETTER_SIZE * configs.BLANK_PATCH_FRAC)
        patch = cv2.getRectSubPix(channel, [w,w], center)

        mean, stddev = cv2.meanStdDev(patch)
        cv = stddev / mean * 100

        cv2.putText(draw, "%.1f" % mean , [center[0]-5, center[1]], cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255))
        cv2.putText(draw, "%.1f" % stddev , [center[0]-5, center[1]+12], cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0))

        if cv > configs.BLANK_COEF_VAR_MAX:
          continue

        cv2.putText(draw, "%.1f" % cv , [center[0]-5, center[1]+24], cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0))

        # Save the mean as a possible blank.
        possible_blanks.set(i, j, mean)

    if configs.DEBUG_BLANKS:
      POST("blank draw stage 1 (G=CV, R=mean, B=stddev)", draw)

    draw = cv2.cvtColor(channel, cv2.COLOR_GRAY2RGB)

    # STAGE 2: we look for blank squares that are outliers among their
    # neighbors. These are going to be the blank letter tiles.
    final_blanks = Board()
    for i in range(configs.BOARD_SIZE):
      for j in range(configs.BOARD_SIZE):
        r = possible_blanks.get(i, j)
        if not r:
          continue

        nearest = np.array(possible_blanks.get_nearest_not_none(i, j, configs.BLANK_NEIGHBORS))

        mean = np.mean(nearest)
        std = np.std(nearest)
        z = abs(r - mean) / std

        color = (0, 0, 255)
        if z > configs.BLANK_Z_THRESH:
          final_blanks.set(i, j, True)
          color = (0, 255, 0)

        center = get_center(i, j)
        cv2.putText(draw, "%.1f" % z , [center[0]-5, center[1]], cv2.FONT_HERSHEY_SIMPLEX, 0.4, color)

    if configs.DEBUG_BLANKS:
      POST("blank draw stage 2 (vs neighbor Z score)", draw)

    return final_blanks


class IterSkip(Exception): #Using exceptions for loop control... so hacky...
  def __init__(self):
    pass


class ScrabbleVision(Thread):
  def __init__(self, source):
    Thread.__init__(self)
    self.daemon = True
    self.source = source

    self.l = Lock()
    self.started = False
    self.killed = False

    self.board = Board()
    self.overrides = Board()
    self.letter_model = LetterModel()
    self.averaged_board = AveragedBoard()
    self.corner_finder = CornerFinder()

    self.intervals = []

  def reset(self):
    self.corner_finder.reset()
    self.averaged_board.reset()

  def get_current_board(self):
    with self.l:
      return self.board.copy()

  def set_override(self, i, j, v):
    with self.l:
      self.overrides.set(i, j, v)

  def kill(self):
    self.killed = True

  def run(self):

      self.letter_model.load()
      self.source.start()

      # TODO: simplify this massive loop.
      while True:
        start = time.time()

        frame_raw = self.source.read()
        if frame_raw is None:
          print('No frame received; terminating.')
          return

        if self.killed:
          print("Vision terminating")
          return

        # Reload configs.py so that it can be changed on the fly in order to
        # tune vision processing.
        importlib.reload(configs)

        try:

          if configs.CAPTURE_VFLIP: 
            frame = cv2.flip(frame_raw, flipCode=-1)
          else:
            frame = frame_raw

          corners_sorted = self.corner_finder.process(frame)

          if not corners_sorted:
            if configs.DEBUG:
              print("Missing corners")
            raise IterSkip()

          #sort corners top left, top right, bottom right, bottom left
          src = np.array(corners_sorted, np.float32)
          d_dp = np.array([[0,0],[1000,0],[1000,1000],[0,1000]], np.float32)
          m1 = cv2.getPerspectiveTransform(src, d_dp)

          # Crop in relative to 1000x1000 using configured settings
          d_adj = np.array([
            [configs.TL_X, configs.TL_Y],
            [1000 - configs.TR_X, configs.TR_Y],
            [1000 - configs.BR_X, 1000 - configs.BR_Y],
            [configs.BL_X, 1000 - configs.BL_Y],
          ], np.float32)

          m2 = cv2.getPerspectiveTransform(d_adj, d_dp)

          # Pad the output by a fraction of a letter to give us a margin
          pad = int(1000. / configs.BOARD_SIZE  * configs.LETTER_PAD_FRAC)
          d_pad = np.array([[pad,pad],[1000-pad,pad],[1000-pad,1000-pad],[pad,1000-pad]], np.float32)

          m3 = cv2.getPerspectiveTransform(d_dp, d_pad)

          sz = get_board_size()
          step = int(configs.LETTER_SIZE * configs.LETTER_PAD_FRAC)

          d_final = np.array([[0,0],[sz,0],[sz,sz],[0,sz]], np.float32)
          m4 = cv2.getPerspectiveTransform(d_dp, d_final)

          normalized_board_image = cv2.warpPerspective(frame, m4.dot(m3.dot(m2.dot(m1))), (sz,sz))

          # Find and classify letters.
          letter_finder = LetterFinder()
          letter_finder.process(normalized_board_image)
          letters = self.letter_model.classify_all(letter_finder)

          # Find blanks.
          blank_finder = BlankFinder()
          blanks = blank_finder.process(normalized_board_image, letters)

          # Update the averaged board with above observations.
          for j in range(0,configs.BOARD_SIZE):
            for i in range(0,configs.BOARD_SIZE):
              is_blank = blanks.get(i, j)
              if is_blank:
                letters.set(i, j, '-')

              r = letters.get(i, j)
              if r:
                self.averaged_board.observe(i,j,r)
              else:
                self.averaged_board.observe(i,j,None)  # empty square

          # Update the published board
          with self.l:
            for i in range(0,configs.BOARD_SIZE):
              for j in range(0,configs.BOARD_SIZE):
                v = self.averaged_board.get(i,j)
                o = self.overrides.get(i, j)
                if o:
                  v = o
                self.board.set(i,j,v)

          # Draw the current letter classification.

          # TODO: would be cleaner with simple rectangles
          letter_draw = normalized_board_image.copy()
          line_color = (0,0,255)
          # Draw bounding lines
          cv2.line(letter_draw, (step,0), (step,sz), line_color)
          cv2.line(letter_draw, (sz-step,0), (sz-step,sz), line_color)
          cv2.line(letter_draw, (0,step), (sz,step), line_color)
          cv2.line(letter_draw, (0,sz-step), (sz,sz-step), line_color)
          # Draw gridlines on the board
          x = step
          for i in range(0,14):
            x += float(sz-2*step) / configs.BOARD_SIZE
            cv2.line(letter_draw, (int(x),step), (int(x),sz-step), line_color)
          y = step
          for i in range(0,14):
            y += float(sz-2*step) / configs.BOARD_SIZE
            cv2.line(letter_draw, (step,int(y)), (sz-step,int(y)), line_color)

          y = step
          for j in range(0,configs.BOARD_SIZE):
            x = step
            for i in range(0,configs.BOARD_SIZE):

              r = letters.get(i,j)
              if r is not None:
                cv2.putText(letter_draw, str(r.upper()), (int(x)+configs.LETTER_SIZE-12,int(y)+configs.LETTER_SIZE-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))

              r_out = self.board.get(i,j) 
              if r_out is not None:
                cv2.putText(letter_draw, str(r_out.upper()), (int(x)+12,int(y)+32), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255))

              x += float(sz-2*step) / configs.BOARD_SIZE
            y += float(sz-2*step) / configs.BOARD_SIZE

          POST("Letter Classification (W=filtered, G=instant)", letter_draw)

        except IterSkip as e:
          pass
        except Exception as e:
          print("Exception occured: %s" % str(e))
          print("--------")
          print(traceback.format_exc())
          print("--------")

        self.intervals.insert(0, time.time() - start)
        if len(self.intervals) > 10:
          self.intervals.pop()
        fps = 1.0 / (sum(self.intervals) / len(self.intervals)) if self.intervals else 0

        if configs.DEBUG_VERBOSE:
          print("Processing at %.2f FPS" % fps)

        self.started = True

        # TODO: replace with capped FPS
        key = cv2.waitKey(333)

      print("Terminating...")

