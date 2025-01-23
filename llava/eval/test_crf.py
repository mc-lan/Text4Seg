import sys
import numpy as np
import cv2
from cv2 import imread, imwrite

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels


def crf_refine(image, labels, n_labels=2, gt_prob=0.9):
    """
    Refine a coarse mask using DenseCRF.
    
    :param image: Input image (HxWx3)
    :param labels: Coarse mask (HxW) where 0=background, 1=first class, 2=second class, ...
    :return: Refined mask
    """
    # Create the DenseCRF model
    # image = image.transpose((1,2,0))
    h, w = image.shape[:2]
    image = image.astype(np.uint8)
    # print(h, w)
    # print(image.shape)
    # print(labels.shape)
    labels = labels.flatten()
    d = dcrf.DenseCRF2D(w, h, n_labels)

    # Get unary potentials (negative log probabilities)
    unary = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=False)
    d.setUnaryEnergy(unary)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    d.addPairwiseBilateral(sxy=(50, 50), srgb=(13, 13, 13), rgbim=image,
                           compat=10,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)

    # Inference
    refined_mask = d.inference(5)  # Number of iterations
    refined_mask = np.argmax(refined_mask, axis=0)
    refined_mask = refined_mask.reshape(h, w)

    return refined_mask

if __name__ == '__main__':
    fn_im = sys.argv[1]
    fn_anno = sys.argv[2]
    fn_output = sys.argv[3]
    gt = sys.argv[4]

    ##################################
    ### Read images and annotation ###
    ##################################
    img = imread(fn_im)
    gt_img = imread(gt)
    labels = imread(fn_anno, cv2.IMREAD_GRAYSCALE) 

    refine_out = crf_refine(img, (labels / 255).astype(np.int32), n_labels=2)
    color_out = refine_out.reshape(img.shape[:2]) * 255

    color_out = np.hstack([img, np.repeat(labels[..., np.newaxis], 3, 2), np.repeat(color_out[..., np.newaxis], 3, 2), gt_img]) 
    imwrite(fn_output, color_out)
