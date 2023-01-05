import cv2
import source
import configs
import importlib

from vision.corner_finder import CornerFinder
from vision.letter_finder import LetterFinder
from vision.perspective import crop_board
from vision.letter_model import LetterModelTrainer


def main():
  print('Training mode!')

  vs = source.CvSource()
  vs.start()

  model = LetterModelTrainer()
  print('Training from existing dataset samples')
  model.load_dataset_samples()

  while True:
    try:
      importlib.reload(configs)

      print('Grabbing frame')
      frame = vs.read()

      corners = CornerFinder().process(frame)
      if not corners:
        print('Missing corners!')
        continue

      board_image = crop_board(frame, corners)

      letter_finder = LetterFinder()
      letter_finder.process(board_image)
      model.train_all(letter_finder)
    except StopIteration:
      break

  model.save()
  print('Training complete')


if __name__ == '__main__':
  main()
