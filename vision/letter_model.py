import cv2
import numpy as np
import configs
from board import Board

from .util import *


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
