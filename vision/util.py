import math
import configs
import cv2


def POST(name, img):
  cv2.namedWindow(name)
  cv2.imshow(name, img)


def distance(p1, p2):
  (x1,y1) = p1
  (x2,y2) = p2
  return math.sqrt((x1-x2)**2 + (y1-y2)**2)


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
