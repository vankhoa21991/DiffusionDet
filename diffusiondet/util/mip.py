import numpy as np
import matplotlib.pyplot as plt
import os
import SimpleITK as sitk


def interval_mapping(image, from_min, from_max, to_min, to_max):
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)

def createMIP_transverse(np_img, slices_num):
    ''' create the mip image from original image, slice_num is the number of slices for maximum intensity projection'''
    img_shape = np.shape(np_img)
    np_mip = np.zeros_like(np_img)
    for i in range(img_shape[2]-slices_num):
        np_mip[:, :, i] = np.amax(np_img[:, :, i:(i + slices_num)], axis=2)
    return np_mip

def createMIP_transverse_nifti(np_img, slices_num):
    ''' create the mip image from original image, slice_num is the number of slices for maximum intensity projection'''
    img_shape = np.shape(np_img)
    np_mip = np.zeros_like(np_img)
    for i in range(img_shape[0]-slices_num):
        np_mip[i, :, :] = np.amax(np_img[i:(i + slices_num), :, :], axis=0)
    return np_mip

