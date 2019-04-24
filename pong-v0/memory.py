import numpy as np
import gym
import time
import tensorflow as tf
from collections import deque # Ordered collection with ends
from model import DQNetwork
import cv2
import utils
import os
import _pickle as pickle



# max_load_size = 3000
# memory_save_size = 1000
# max_load_file = max_load_size // memory_save_size
# save_path = "./data/"
# prefix_data_name = "data_"


class Memory():
    _defaults = {
        "prefix_data_name": "data_",
        "save_path": "./data/",
        "max_size": 100,
        "max_load_size": 300
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides

        self.buffer = deque(maxlen = self.max_size)
        self.max_load_file = self.max_load_size // self.max_size
        self.tmp_max_load_size = 0
        self.to_save_prefix = self.save_path + self.prefix_data_name
        self.get_data_file_names()

    def get_end_of_name(self, i = -1):
        return len(self.data_file_names) - i - 1

    def get_data_file_names(self):
        if os.path.isdir(self.save_path):
            self.data_file_names = os.listdir(self.save_path)
        else:
            os.mkdir(self.save_path)
            self.data_file_names = []

    def save_file(self, data):
        end_of_name = self.get_end_of_name()
        data_file_name = utils.gen_data_file_name(self.to_save_prefix, end_of_name)
        with open(data_file_name, 'wb') as handle:
            pickle.dump(data, handle)

    def add(self, experience):
        """
        add and save when buffer reached max size
        """
        self.buffer.append(experience)
        if len(self.buffer) == self.max_size:
            print("save file")
            self.save_file(self.buffer)
            self.get_data_file_names()
            self.tmp_max_load_size = min(self.tmp_max_load_size + self.max_size, self.max_load_size)
            self.buffer = deque(maxlen = self.max_size)

    def get_random_list(self, batch_size):
        random_list = np.random.randint(0, self.tmp_max_load_size + len(self.buffer),
                                        size = batch_size) 
        return np.array(random_list)

    def find_files_to_load(self, random_list):
        """
        it map a random_list number into exact data_file_name
        return a dictionary, with keys of data_file_names and ...
        also 
        """
        files_to_load = {}
        for i in range(min(self.max_load_file, len(self.data_file_names))):
            end_of_name = self.get_end_of_name(i)
            data_file_name = utils.gen_data_file_name(self.to_save_prefix, end_of_name)
            mask = ((random_list>= self.max_size *i) & (random_list < self.max_size *(i+1)))
            if sum(mask) > 0:
                files_to_load[data_file_name] = random_list[mask] % self.max_size
        return files_to_load

    def sample(self, batch_size):
        random_list = self.get_random_list(batch_size)
        files_to_load = self.find_files_to_load(random_list)
        batch_data = self.load_sample_from_files(files_to_load)
        load_in_buffer = random_list[random_list > self.tmp_max_load_size] %self.max_size
        for data in load_in_buffer:
            batch_data.append(self.buffer[data])
        return batch_data

    def load_sample_from_files(self, files_to_load):
        """
        of course I can produce a files_to_load just by calling find_files_to_load
        function. But I do think it is not good. Dont know why.
        """
        batch = []
        for data_file_name in files_to_load:
            with open(data_file_name, 'rb') as handle:
                data_file = pickle.load(handle)
                for ind in files_to_load[data_file_name]:
                    batch.append(data_file[ind])
        return batch