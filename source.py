import cv2
import ipcam
import configs
import re
import os


class FileSource(object):

  def start(self):
    pass

  def read(self):
    img = cv2.imread('scrabble_images/IMG_20141025_164052.jpg')
    small = cv2.resize(img, (720,960))
    return small


class IPSource(object):
  def start(self):
    self.ip = ipcam.OpenCvIPCamera('http://192.168.1.122:8080/video')
    self.ip.start()

  def read(self):
    im = self.ip.read()
    small = cv2.resize(im, (0,0), fx=0.25, fy=0.25) 
    return small


def GetWebcamID():
    DEFAULT_CAMERA_NAME = '/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_1142B73F-video-index0'
    if not os.path.exists(DEFAULT_CAMERA_NAME):
        return None
    device_path = os.path.realpath(DEFAULT_CAMERA_NAME)
    device_re = re.compile("\/dev\/video(\d+)")
    info = device_re.match(device_path)
    if not info:
        return None
    return int(info.group(1))


class CvSource(object):
  def start(self):
    webcam_id = GetWebcamID()
    if webcam_id is None:
        print('Target webcam not found!')
        webcam_id = 0
    self.vc = cv2.VideoCapture(webcam_id)

    self.vc.set(cv2.CAP_PROP_FRAME_WIDTH, configs.CAPTURE_WIDTH)
    self.vc.set(cv2.CAP_PROP_FRAME_HEIGHT, configs.CAPTURE_HEIGHT)

  def read(self):
    rval, frame_raw = self.vc.read()
    if not rval:
      return None
    # TODO: make vflip work for all sources using base class
    if configs.CAPTURE_VFLIP: 
      frame_raw = cv2.flip(frame_raw, flipCode=-1)
    return frame_raw

