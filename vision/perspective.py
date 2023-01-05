import cv2
import numpy as np
import configs

from .util import *


def crop_board(frame, corners):
  src = np.array(corners, np.float32)
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

  d_final = np.array([[0,0],[sz,0],[sz,sz],[0,sz]], np.float32)
  m4 = cv2.getPerspectiveTransform(d_dp, d_final)

  return cv2.warpPerspective(frame, m4.dot(m3.dot(m2.dot(m1))), (sz,sz))
