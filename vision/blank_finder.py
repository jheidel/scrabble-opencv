import cv2
import numpy as np
import configs
from board import Board

from .util import *


class BlankFinder:
  """
  BlankFinder handles detection of blank letter tiles in an image.
  """
  def __init__(self):
    pass

  def process(self, image, letters):
    """
    process the provided `image`, ignoring previously classified letters in the provided board `letters`
    """
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
