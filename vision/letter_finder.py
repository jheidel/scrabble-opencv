import cv2
import configs

from .util import *


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
