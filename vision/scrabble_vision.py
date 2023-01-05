import cv2
import numpy as np
import configs
from threading import Thread, Lock, Event
from board import Board, AveragedBoard
import traceback
import importlib
import time

from .corner_finder import CornerFinder
from .letter_finder import LetterFinder
from .letter_model import LetterModelClassifier
from .blank_finder import BlankFinder
from .perspective import crop_board
from .util import *


class CameraThread(Thread):
  def __init__(self, source):
    Thread.__init__(self)

    self.l = Lock()
    self.killed = Event()

    self.source = source
    self.source.start()
    self.frame = self.source.read()

  def get(self):
    with self.l:
      return self.frame

  def kill(self):
    self.killed.set()

  def run(self):
    while not self.killed.is_set():
      frame = self.source.read()
      with self.l:
        self.frame = frame


class IterSkip(Exception): #Using exceptions for loop control... so hacky...
  def __init__(self):
    pass


class ScrabbleVision(Thread):
  def __init__(self, source):
    Thread.__init__(self)
    self.daemon = True
    self.camera = CameraThread(source)

    self.l = Lock()
    self.started = False
    self.killed = Event()

    self.board = Board()
    self.overrides = Board()
    self.letter_model = LetterModelClassifier()
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
    self.killed.set()

  def run(self):
      self.letter_model.load()
      self.camera.start()

      # TODO: simplify this massive loop.
      while True:

        frame = self.camera.get()
        if frame is None:
          print('No frame received; terminating.')
          return

        if self.killed.is_set():
          print("Vision terminating")
          return

        start = time.time()


        # Reload configs.py so that it can be changed on the fly in order to
        # tune vision processing.
        importlib.reload(configs)

        try:
          corners = self.corner_finder.process(frame)

          if not corners:
            if configs.DEBUG_VERBOSE:
              print("Missing corners")
            raise IterSkip()

          normalized_board_image = crop_board(frame, corners)

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
          step = int(configs.LETTER_SIZE * configs.LETTER_PAD_FRAC)
          sz = get_board_size()
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

        interval = time.time() - start

        self.intervals.insert(0, interval)
        if len(self.intervals) > 10:
          self.intervals.pop()
        fps = 1.0 / (sum(self.intervals) / len(self.intervals)) if self.intervals else 0

        if configs.DEBUG_VERBOSE:
          print("Processing at %.2f FPS" % fps)

        self.started = True

        key = cv2.waitKey(max(10, int(1000. / configs.MAX_FPS - 1000 * interval)))

      print("Terminating...")
