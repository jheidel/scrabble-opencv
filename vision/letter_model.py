import cv2
import numpy as np
import configs
from board import Board
import os


from .util import *

DATASET_PATH = "vision/dataset/"
MODEL_PATH = "vision/dataset/model.yml"


def to_sample(img):
  shrunk = cv2.resize(img, (configs.LETTER_TRAIN_SIZE, configs.LETTER_TRAIN_SIZE))
  return shrunk.reshape((1,configs.LETTER_TRAIN_SIZE**2))


class LetterModelTrainer:
  def __init__(self):
    self._responses = []
    self._samples = np.empty((0,configs.LETTER_TRAIN_SIZE**2))

  def load_dataset_samples(self):
    for letter in os.listdir(DATASET_PATH):
      if len(letter) != 1:
        continue
      letter_path = os.path.join(DATASET_PATH, letter)
      for png in os.listdir(letter_path):
        png_path = os.path.join(letter_path, png)
        img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        self.append_model(img, letter)

  def save(self):
    samples = np.array(self._samples, np.float32)
    responses = np.array(self._responses, np.float32)
    responses = responses.reshape((responses.size,1))

    print("Training model")
    model = cv2.ml.KNearest_create()
    model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    print("Saving model to %s" % MODEL_PATH)
    model.save(MODEL_PATH)

  def train_all(self, finder):
    for j in range(0,configs.BOARD_SIZE):
      for i in range(0,configs.BOARD_SIZE):
        img = finder.get_thresh(i, j)
        if img is not None:
          self.train_letter(img)

  def train_letter(self, thresh_image):
    print("Which letter is this? (use '0' for star, enter to stop, esc to skip)")
    POST("Which letter is this? (TRAINING)", thresh_image)
    o = cv2.waitKey(0)
    if chr(o).lower() >= 'a' and chr(o).lower() <= 'z' or chr(o).lower() == '0':
      x = chr(o).lower()
      print("You said it's a %s" % str(x))
      self.append_model(thresh_image, x)

      # Save raw image to the dataset directory
      path = os.path.join(DATASET_PATH, x)
      if not os.path.exists(path):
        os.makedirs(path)
      cv2.imwrite(os.path.join(path, '%04d.png' % len(os.listdir(path))), thresh_image)

    elif o == 27:
      print("Skipping, here's another...")
    else:
      raise StopIteration

  def append_model(self, img, value):
    sample = to_sample(img)
    self._samples = np.append(self._samples, sample, 0)
    self._responses.append(ord(value.lower()))


class LetterModelClassifier:
  def __init__(self):
    self._model = None

  def load(self):
    self._model = cv2.ml.KNearest_create().load(MODEL_PATH)

  def classify_all(self, finder):
    letters = Board()
    for j in range(0,configs.BOARD_SIZE):
      for i in range(0,configs.BOARD_SIZE):
        img = finder.get_thresh(i, j)
        if img is None:
          continue
        r = self.classify_letter(img)
        if r is not None:
          letters.set(i, j, r)
    return letters

  def classify_letter(self, thresh_image):
    sample = to_sample(thresh_image)
    sample = np.float32(sample)
    retval, results, neigh_resp, dists = self._model.findNearest(sample, k = 1)
    retchar = chr(int((results[0][0])))
    if retchar == '0':
      #Star character!
      return None
    return retchar
