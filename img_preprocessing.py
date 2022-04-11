import numpy as np
import skimage.io
import skimage.color
import skimage.filters

import skimage.io
import skimage.color
import skimage.filters

from PIL import Image, ImageOps
import matplotlib.pyplot as plt


def crop_img(df_loc_im):

    pic_name = df_loc_im["name"]

    x1 = df_loc_im["x1"]
    x2 = df_loc_im["x2"]
    y1 = df_loc_im["y1"]
    y2 = df_loc_im["y2"]

    im = Image.open(f"../dataset/Medical mask/images/{pic_name}")
    im_crop = im.crop((x1, x2, y1, y2))

    return im_crop


def resize_img(im):
    im_resized = im.resize((50, 50))

    return im_resized


def gs(im_resized):
    im_gray = ImageOps.grayscale(im_resized)
    return im_gray


def blurr_gs(gs_im):
    im_gs_arr = np.asarray(gs_im)
    blurred_image_gs = skimage.filters.gaussian(im_gs_arr, sigma=1.0)
    return blurred_image_gs


def split_rgb(im_resized):
    im_r, im_g, im_b = im_resized.split()

    rgb_chan = [im_r, im_g, im_b]
    return rgb_chan


def blurr_img(rgb_chan):
    im_r_arr, im_g_arr, im_b_arr = [np.asarray(rgb_chan[i]) for i in range(len(rgb_chan))]

    blurred_image_r = skimage.filters.gaussian(im_r_arr, sigma=1.0)
    blurred_image_g = skimage.filters.gaussian(im_g_arr, sigma=1.0)
    blurred_image_b = skimage.filters.gaussian(im_b_arr, sigma=1.0)

    blurred_rgb_chan = (blurred_image_r, blurred_image_g, blurred_image_b)

    return blurred_rgb_chan


def gs_hist(blurred_gs_im):
    histogram_gs, bin_edges = np.histogram(blurred_gs_im, bins=300, range=(0, 1))
    return histogram_gs, bin_edges


def rgb_hist(blurred_rgb_chan):
    histogram_r, bin_edges_r = np.histogram(blurred_rgb_chan[0], bins=256, range=(0, 1))
    histogram_g, bin_edges_g = np.histogram(blurred_rgb_chan[1], bins=256, range=(0, 1))
    histogram_b, bin_edges_b = np.histogram(blurred_rgb_chan[2], bins=256, range=(0, 1))

    rgb_chan_hist = [(histogram_r, bin_edges_r),
                     (histogram_g, bin_edges_g),
                     (histogram_b, bin_edges_b)]
    # plt.show()
    plt.close("all")
    return rgb_chan_hist


def search_tresh_gs(gs_chan_hist):

    gs_tresh_offset = 0
    max_bin_index = np.argmax(gs_chan_hist[0])
    if max_bin_index < 248:
        gs_tresh_offset = 50
    max_gsval_pxcount = gs_chan_hist[1][max_bin_index + gs_tresh_offset]
    return max_gsval_pxcount


def search_tresh(rgb_chan_hist):
    red_tresh_offset = 0
    max_bin_index = np.argmax(rgb_chan_hist[0][0])
    if max_bin_index < 248:
        red_tresh_offset = 9
    max_red_pxcount = rgb_chan_hist[0][1][max_bin_index + red_tresh_offset]

    green_tresh_offset = 0
    max_bin_index = np.argmax(rgb_chan_hist[1][0])
    if max_bin_index < 248:
        green_tresh_offset = 9
    max_green_pxcount = rgb_chan_hist[1][1][max_bin_index + green_tresh_offset]

    blue_tresh_offset = 0
    max_bin_index = np.argmax(rgb_chan_hist[2][0])
    if max_bin_index < 248:
        blue_tresh_offset = 9
    max_blue_pxcount = rgb_chan_hist[2][1][max_bin_index + blue_tresh_offset]

    max_pxcount_rgb = (max_red_pxcount, max_green_pxcount, max_blue_pxcount)

    return max_pxcount_rgb


def gs_treshold(max_gsval_pxcount, blurred_image_gs):
    gray_treshold = max_gsval_pxcount
    binary_mask_gs = blurred_image_gs < gray_treshold

    return binary_mask_gs


def treshold(max_pxcount_rgb, blurred_img_rgb):
    red_treshold = max_pxcount_rgb[0]
    green_treshold = max_pxcount_rgb[1]
    blue_treshold = max_pxcount_rgb[2]

    binary_mask_r = blurred_img_rgb[0] < red_treshold
    binary_mask_g = blurred_img_rgb[1] < green_treshold
    binary_mask_b = blurred_img_rgb[2] < blue_treshold

    binary_mask_rgb = (binary_mask_r, binary_mask_g, binary_mask_b)

    return binary_mask_rgb


def px_rate_gs(binary_mask_gs):
    px_0 = (binary_mask_gs == False).sum()
    px_1 = (binary_mask_gs == True).sum()

    return px_0, px_1


def px_rate_rgb(binary_mask_rgb):
    px_0_r = (binary_mask_rgb[0] == False).sum()
    px_1_r = (binary_mask_rgb[0] == True).sum()
    px_rate_r = px_0_r / px_1_r

    px_0_g = (binary_mask_rgb[1] == False).sum()
    px_1_g = (binary_mask_rgb[1] == True).sum()
    px_rate_g = px_0_g / px_1_g

    px_0_b = (binary_mask_rgb[2] == False).sum()
    px_1_b = (binary_mask_rgb[2] == True).sum()
    px_rate_b = px_0_b / px_1_b

    px_rate_rgb_data = (px_rate_r, px_rate_g, px_rate_b)

    return px_rate_rgb_data
