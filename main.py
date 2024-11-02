from ultralytics import YOLO
import tensorflow as tf
from IPython.display import display, Image
import yaml

model = YOLO("yolov8n.pt")

data = { 'train' : 'train_path',
         'val' : 'val_path',
         'test' : 'test_path',
         'names' : ['fire','smoke'],
         'nc' : 2}

with open('data.yaml_path', 'w') as f:
  yaml.dump(data, f)

with open('data.yaml_path', 'r') as f:
  fire_detect = yaml.safe_load(f)
  display(fire_detect)

model.train(data='data.yaml_path', epochs=100, imgsz=640)

results = model.predict(source='output_path', save=True)

import locale
def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding
