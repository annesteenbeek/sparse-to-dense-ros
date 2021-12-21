import os
import shutil
import numpy as np
import threading 
from PIL import Image as PILImage


class FrameSaver(object):
    # Used to collect frames for 3D reconstruction in post processing

    def __init__(self, prefix, enabled=True):
        self.enabled = enabled
        if not self.enabled:
            return

        self.foldername = prefix
        self.labels = open("%s.txt" % prefix, "w+")
        self.label_lock = threading.Lock()

        self.labels.write('# cnn depth estimation imagesn\n# timestamp filename \n')

        if os.path.exists(self.foldername):
            shutil.rmtree(self.foldername) # remove old folder if it exists
        os.makedirs(self.foldername)

    def save_image(self, img_np, timestamp):
        if not self.enabled:
            return
        save_thread = threading.Thread(target=self._save_image, args=(img_np, timestamp))
        save_thread.start()

    def _save_image(self, img_np, timestamp):
        time_str = "%.6f" % timestamp.to_sec()
        img_name = "{}/{}.png".format(self.foldername, time_str)

        with self.label_lock:
            self.labels.write("{} {}\n".format(time_str, img_name))

        if img_np.ndim == 3: # rgb
            im = PILImage.fromarray((img_np*255).astype(np.uint8)) 
        else: # depth
            im = PILImage.fromarray((img_np*5000).astype(np.uint16)) # depth images are scaled by 5000 e.g. pixel value of 5000 is 1m
        im.save(img_name)

    def close(self):
        if not self.enabled:
            return

        self.labels.close()
        print("Saved frames in {}".format(self.foldername))

