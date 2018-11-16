import numpy as np
import cv2
from PIL import Image


top_margin = 34
image_width = 160
state_frame_size = (84,84)

def rgb2gray(rgb):
    # why I not use cv2 function? Because I dont want to mess up
    # with bullshit rgb bgr things
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def preprocess_image(image):
    new_image = cv2.resize(image[top_margin: top_margin+image_width,:,:], state_frame_size)
    gray_image = rgb2gray(new_image)
    return gray_image

def stack_4_frames(image_list):
    # stack 4 frames into a 4 channels image
    # change type to float16 to reduce memory usage
    stacked = np.stack(image_list, axis=-1).astype(np.float16)
    return stacked

def stack_frames(stacked_frames, new_frame, is_new_episode):
    new_frame = preprocess_image(new_frame)
    if is_new_episode:
        stacked_frames.append(new_frame)
        stacked_frames.append(new_frame)
        stacked_frames.append(new_frame)
        stacked_frames.append(new_frame)
    else:
        stacked_frames.append(new_frame)
    stacked_state = stack_4_frames(stacked_frames)
    return stacked_frames, stacked_state


def encode_number_to_onehot(number, size):
    onehot = np.zeros(size)
    onehot[number] = 1
    return onehot

def encode_list_to_onehot(_list, action_size):
    return np.array([encode_number_to_onehot(i, action_size) for i in _list])