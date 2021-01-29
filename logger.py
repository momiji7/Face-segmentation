from pathlib import Path
import importlib, warnings
import os, sys, time, numpy as np
import scipy.misc 
from io import BytesIO as BIO
import cv2

class Logger(object):
  
    def __init__(self, log_dir, logstr):
        """Create a summary writer logging to log_dir."""
        self.log_dir = Path(log_dir)
        self.model_dir = Path(log_dir) / 'checkpoint'
        self.meta_dir = Path(log_dir) / 'metas'
        self.image_dir = Path(log_dir) / 'image'
        self.log_dir.mkdir(mode=0o775, parents=True, exist_ok=True)
        self.model_dir.mkdir(mode=0o775, parents=True, exist_ok=True)
        self.meta_dir.mkdir(mode=0o775, parents=True, exist_ok=True)
        self.image_dir.mkdir(mode=0o775, parents=True, exist_ok=True)

        self.logger_path = self.log_dir / '{:}.log'.format(logstr)
        self.logger_file = open(self.logger_path, 'w')


    def __repr__(self):
        return ('{name}(dir={log_dir})'.format(name=self.__class__.__name__, **self.__dict__))

    def path(self, mode):
        if mode == 'meta'   : return self.meta_dir
        elif mode == 'model': return self.model_dir
        elif mode == 'log'  : return self.log_dir
        else: raise TypeError('Unknow mode = {:}'.format(mode))

    def last_info(self):
        return self.log_dir / 'last-info.pth'

    def extract_log(self):
        return self.logger_file

    def close(self):
        self.logger_file.close()

    def log(self, string, save=True):
        print (string)
        if save:
            self.logger_file.write('{:}\n'.format(string))
            self.logger_file.flush()

    def save_images(self, image, save_name):
        cv2.imwrite('{}/{}'.format( self.image_dir, save_name), image)