"""IP camera interface with python OpenCV.

Branched from https://gist.github.com/thearn/5562029.
"""

import cv2
import numpy as np
import time
import threading
import requests
import Queue as queue


class OpenCvIPCamera():
  """Uses an IP webcam target as an OpenCV image source."""

  # Size of chunks to read in from the IP camera stream.
  CHUNK_SIZE = 4096

  def __init__(self, url, lossy=False):
    """Initializer.

    Args:
        url: The URL of the mjpeg stream to stream images from.
        lossy: If enabled, incoming images will be discarded if images are not
        fetched quickly enough. This can be useful if the consumer reads images
        at a much slower rate than produced by the IP camera.
    """
    self.url = url
    self.lossy = lossy

    self.stream = requests.get(self.url, stream=True)
    self.image_queue = queue.Queue(maxsize=1)

    self.thread_cancelled = False
    self.thread = threading.Thread(target=self._run)
    self.thread.daemon = True
    
  def start(self):
    """Start capturing images from the IP camera."""
    self.thread.start()

    # Grab a test image. If that fails, dump out the buffer for debugging.
    if self.read(timeout=5) is None:
      print 'BUFFER DEBUG: %s' % self.imgbuf
      print '** WARNING: failed to capture a test image from %s. **' % self.url

  def shut_down(self):
    """Shuts down image capture thread."""
    self.thread_cancelled = True
    return self.thread.join(timeout=5)

  def read(self, timeout=None):
    """Gets an image from the IP camera, blocking until one is available."""
    try:
      return self.image_queue.get(block=True, timeout=timeout)
    except queue.Empty:
      return None
   
  def _run(self):
    # TODO(jheidel): Optimize buffering & marker finding
    self.imgbuf = ''
    while not self.thread_cancelled:
      try:
        self.imgbuf += self.stream.raw.read(self.CHUNK_SIZE)
        a = self.imgbuf.find('\xff\xd8')  # JPEG start marker
        b = self.imgbuf.find('\xff\xd9')  # JPEG end marker
        if a != -1 and b != -1:
          jpg_bytes = self.imgbuf[a:b+2]
          self.imgbuf = self.imgbuf[b+2:]
          img = cv2.imdecode(np.fromstring(jpg_bytes, dtype=np.uint8),
                             cv2.IMREAD_COLOR)
          try:
            # TODO(jheidel): Freshness of image on the queue? If there's
            # something already on the Queue we probably want to replace it
            # with the latest image so read will pick up the latest image.
            self.image_queue.put(img, block=not self.lossy, timeout=0.001)
          except queue.Full:
            pass  # Discard image

      except ThreadError:
        self.thread_cancelled = True

