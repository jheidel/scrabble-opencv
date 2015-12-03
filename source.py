import cv2
import ipcam


class FileSource(object):

  def start(self):
    pass

  def read(self):
    im = cv2.imread('scrabble_images/IMG_20141025_164044.jpg')
    # TODO
    return small


class IPSource(object):
  def start(self):
    self.ip = ipcam.OpenCvIPCamera('http://192.168.1.122:8080/video')
    self.ip.start()

  def read(self):
    im = self.ip.read()
    small = cv2.resize(im, (0,0), fx=0.25, fy=0.25) 
    return small


class CvSource(object):
  def start(self):
    self.vc = cv2.VideoCapture(0)

  def read(self):
    rval, frame_raw = vc.read()
    if rval:
      return frame_raw
    else:
      return None

