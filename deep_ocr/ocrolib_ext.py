# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 12:11:05 2017

@author: ocropus
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.ndimage import filters,interpolation,morphology,measurements
from scipy.ndimage.filters import gaussian_filter,uniform_filter,maximum_filter
from deep_ocr.ocrolib import psegutils, morph, sl
import deep_ocr.ocrolib as ocrolib

def debug_show(image, title):
    if type(image)==list:
        assert len(image)==3
        image = np.transpose(np.array(image),[1,2,0])
    plt.clf()
    plt.title(title)
    plt.imshow(image)
    raw_input("PRESS ANY KEY TO CONTINUE.")

def normalize_raw_image(raw):
    ''' perform image normalization '''
    image = raw - np.amin(raw)
    if np.amax(image)==np.amin(image):
        print("# image is empty ")
        return None
    image /= np.amax(image)
    return image

def estimate_local_whitelevel(image, zoom=0.5, perc=80, range=20, debug=0):
    '''flatten it by estimating the local whitelevel
    zoom for page background estimation, smaller=faster, default: %(default)s
    percentage for filters, default: %(default)s
    range for filters, default: %(default)s
    '''
    m = interpolation.zoom(image, zoom)
    m = filters.percentile_filter(m, perc, size=(range, 2))
    m = filters.percentile_filter(m, perc, size=(2, range))
    m = interpolation.zoom(m, 1.0/zoom)
    if debug > 0:
        plt.clf()
        plt.title("m after remove noise")
        plt.imshow(m, vmin=0, vmax=1)
        raw_input("PRESS ANY KEY TO CONTINUE.")
    w, h = np.minimum(np.array(image.shape), np.array(m.shape))
    flat = np.clip(image[:w,:h]-m[:w,:h]+1,0,1)
    if debug > 0:
        plt.clf()
        plt.title("flat after clip")
        plt.imshow(flat,vmin=0,vmax=1)
        raw_input("PRESS ANY KEY TO CONTINUE.")
    return flat

def estimate_skew_angle(image, angles, debug):
    estimates = []
    for a in angles:
        v = np.mean(interpolation.rotate(image, a, order=0, mode='constant'),
                    axis=1)
        v = np.var(v)
        estimates.append((v,a))
    if debug>0:
        plt.clf()
        plt.title("estimate_skew_angle")
        plt.plot([y for x,y in estimates],[x for x,y in estimates])
        raw_input("PRESS ANY KEY TO CONTINUE.")
    _, a = max(estimates)
    return a

def estimate_skew(flat, bignore=0.1, maxskew=2, skewsteps=8, debug=0):
    ''' estimate skew angle and rotate'''
    d0,d1 = flat.shape
    o0,o1 = int(bignore*d0),int(bignore*d1) # border ignore
    flat = np.amax(flat)-flat
    flat -= np.amin(flat)
    est = flat[o0:d0-o0,o1:d1-o1]
    ma = maxskew
    ms = int(2*maxskew*skewsteps)
    # print(linspace(-ma,ma,ms+1))
    angle = estimate_skew_angle(est,
                                np.linspace(-ma,ma,ms+1),
                                debug=debug)
    flat = interpolation.rotate(flat, angle, mode='constant', reshape=0)
    flat = np.amax(flat)-flat
    return flat, angle


def estimate_thresholds(flat, bignore=0.1, escale=1.0, lo=5, hi=90, debug=0):
    '''# estimate low and high thresholds
    ignore this much of the border for threshold estimation, default: %(default)s
    scale for estimating a mask over the text region, default: %(default)s
    lo percentile for black estimation, default: %(default)s
    hi percentile for white estimation, default: %(default)s
    '''
    d0,d1 = flat.shape
    o0,o1 = int(bignore*d0), int(bignore*d1)
    est = flat[o0:d0-o0,o1:d1-o1]
    if escale>0:
        # by default, we use only regions that contain
        # significant variance; this makes the percentile
        # based low and high estimates more reliable
        e = escale
        v = est - filters.gaussian_filter(est, e*20.0)
        if debug:
            plt.clf()
            plt.title("first gaussian_filter")
            plt.imshow(v)
            raw_input("PRESS ANY KEY TO CONTINUE.")
        v = filters.gaussian_filter(v**2, e*20.0)**0.5
        if debug:
            plt.clf()
            plt.title("second gaussian_filter")
            plt.imshow(v)
            raw_input("PRESS ANY KEY TO CONTINUE.")
        v = (v > 0.3 * np.amax(v))
        if debug:
            plt.clf()
            plt.title("binarization")
            plt.imshow(v)
            raw_input("PRESS ANY KEY TO CONTINUE.")
        v = morphology.binary_dilation(v, structure=np.ones((int(e*50), 1)))
        v = morphology.binary_dilation(v, structure=np.ones((1, int(e*50))))
        if debug:
            plt.clf()
            plt.title("morphology dilation")
            plt.imshow(v)
            raw_input("PRESS ANY KEY TO CONTINUE.")
        est = est[v]
    lo = stats.scoreatpercentile(est.ravel(),lo)
    hi = stats.scoreatpercentile(est.ravel(),hi)
    return lo, hi

def compute_colseps_conv(binary, csminheight, maxcolseps, scale=1.0, debug=False):
    """Find column separators by convoluation and
    thresholding."""
    h,w = binary.shape
    # find vertical whitespace by thresholding
    smoothed = gaussian_filter(1.0 * binary, (scale, scale*0.5))
    smoothed = uniform_filter(smoothed, (5.0*scale,1))
    thresh = (smoothed<np.amax(smoothed)*0.1)
    if debug:
        debug_show(thresh, "compute_colseps_conv thresh")
    # find column edges by filtering
    grad = gaussian_filter(1.0*binary, (scale, scale*0.5), order=(0,1))
    grad = uniform_filter(grad, (10.0*scale,1))
    # grad = abs(grad) # use this for finding both edges
    grad = (grad>0.5*np.amax(grad))
    if debug:
        debug_show(grad, "compute_colseps_conv grad")
    # combine edges and whitespace
    seps = np.minimum(thresh,maximum_filter(grad, (int(scale), int(5*scale))))
    seps = maximum_filter(seps,(int(2*scale),1))
    if debug:
        debug_show(seps, "compute_colseps_conv seps")
    # select only the biggest column separators
    seps = morph.select_regions(seps,sl.dim0,
                                min=csminheight*scale,
                                nbest=maxcolseps)
    if debug:
        debug_show(seps, "compute_colseps_conv 4seps")
    return seps

def compute_separators_morph(binary, scale, sepwiden, maxseps):
    """Finds vertical black lines corresponding to column separators."""
    d0 = int(max(5,scale/4))
    d1 = int(max(5,scale))+sepwiden
    thick = morph.r_dilation(binary,(d0,d1))
    vert = morph.rb_opening(thick,(10*scale,1))
    vert = morph.r_erosion(vert,(d0//2, sepwiden))
    vert = morph.select_regions(vert,sl.dim1,min=3,nbest=2*maxseps)
    vert = morph.select_regions(vert,sl.dim0,min=20*scale,nbest=maxseps)
    return vert


def compute_colseps(binary, scale, csminheight, maxcolseps,
                    blackseps, maxseps, sepwiden, debug=False):
    """Computes column separators either from vertical black lines or whitespace."""
    print("considering at most %g whitespace column separators" % maxcolseps)
    colseps = compute_colseps_conv(binary=binary,
                                   csminheight=csminheight,
                                   maxcolseps=maxcolseps,
                                   scale=scale,
                                   debug=debug)
    if debug:
        debug_show(0.7*colseps + 0.3*binary,
                   "compute_colseps colwsseps")
    if blackseps and maxseps == 0:
        # simulate old behaviour of blackseps when the default value
        # for maxseps was 2, but only when the maxseps-value is still zero
        # and not set manually to a non-zero value
        maxseps = 2

    if maxseps > 0:
        print("considering at most %g black column separators" % maxseps)
        seps = compute_separators_morph(binary, scale, sepwiden, maxseps)
        if debug:
            debug_show(0.7*colseps + 0.3*binary,
                       "compute_colseps colseps")
        #colseps = compute_colseps_morph(binary,scale)
        colseps = np.maximum(colseps, seps)
        binary = np.minimum(binary, 1-seps)
    return colseps,binary


def remove_hlines(binary, scale, maxsize=10):
    labels, _ = morph.label(binary)
    objects = morph.find_objects(labels)
    for i,b in enumerate(objects):
        if sl.width(b)>maxsize*scale:
            labels[b][labels[b]==i+1] = 0
    return np.array(labels!=0,'B')


def compute_gradmaps(binary, scale, usegauss, vscale, hscale, debug=False):
    # use gradient filtering to find baselines
    boxmap = psegutils.compute_boxmap(binary,scale)
    cleaned = boxmap*binary
    if debug:
        debug_show(cleaned, "cleaned")
    if usegauss:
        # this uses Gaussians
        grad = gaussian_filter(1.0*cleaned,(vscale*0.3*scale,
                                            hscale*6*scale),
                                            order=(1,0))
    else:
        # this uses non-Gaussian oriented filters
        grad = gaussian_filter(1.0*cleaned, (max(4, vscale*0.3*scale),
                                            hscale*scale ), order=(1,0))
        grad = uniform_filter(grad, (vscale, hscale*6*scale))
    if debug:
        debug_show(grad, "compute_gradmaps grad")
    bottom = ocrolib.norm_max((grad<0)*(-grad))
    top = ocrolib.norm_max((grad>0)*grad)
    if debug:
        debug_show(bottom, "compute_gradmaps bottom")
        debug_show(top, "compute_gradmaps top")
    return bottom, top, boxmap


def compute_line_seeds(binary, bottom, top, colseps, threshold, vscale, scale, debug=False):
    """Base on gradient maps, computes candidates for baselines
    and xheights.  Then, it marks the regions between the two
    as a line seed."""
    t = threshold
    vrange = int(vscale*scale)
    bmarked = maximum_filter(bottom==maximum_filter(bottom, (vrange, 0)),(2,2))
    bmarked = bmarked*(bottom>t*np.amax(bottom)*t)*(1-colseps)
    tmarked = maximum_filter(top==maximum_filter(top,(vrange,0)),(2,2))
    tmarked = tmarked*(top>t*np.amax(top)*t/2)*(1-colseps)
    tmarked = maximum_filter(tmarked,(1,20))
    seeds = np.zeros(binary.shape, 'i')
    delta = max(3,int(scale/2))
    for x in range(bmarked.shape[1]):
        transitions = sorted([(y, 1) for y in np.where(bmarked[:,x])[0]]+[(y,0) for y in np.where(tmarked[:,x][0])])[::-1]
        transitions += [(0,0)]
        for l in range(len(transitions)-1):
            y0,s0 = transitions[l]
            if s0==0: continue
            seeds[y0-delta:y0,x] = 1
            y1,s1 = transitions[l+1]
            if s1==0 and (y0-y1)<5*scale: seeds[y1:y0,x] = 1
    seeds = maximum_filter(seeds,(1,int(1+scale)))
    seeds = seeds*(1-colseps)
    if debug:
        debug_show([seeds,0.3*tmarked+0.7*bmarked,binary], "lineseeds")
    seeds,_ = morph.label(seeds)
    return seeds

def compute_segmentation(binary, scale,
                         csminheight, maxcolseps, blackseps,
                         maxseps, sepwiden, usegauss, hscale, vscale, threshold,
                         debug=False,
                         verbose=False):
    """Given a binary image, compute a complete segmentation into
    lines, computing both columns and text lines."""
    binary = np.array(binary, 'B')

    # start by removing horizontal black lines, which only
    # interfere with the rest of the page segmentation
    binary = remove_hlines(binary, scale)

    # do the column finding
    if verbose:
        print("computing column separators")

    colseps, binary = compute_colseps(binary=binary, scale=scale,
                                      csminheight=csminheight,
                                      maxcolseps=maxcolseps, blackseps=blackseps,
                                      maxseps=maxseps, sepwiden=sepwiden, debug=debug)

    # now compute the text line seeds
    if verbose: print("computing lines")
    bottom,top,boxmap = compute_gradmaps(binary=binary,
                                         scale=scale,
                                         usegauss=usegauss,
                                         vscale=vscale,
                                         hscale=hscale,
                                         debug=debug)
    seeds = compute_line_seeds(binary=binary,
                               bottom=bottom,
                               top=top,
                               colseps=colseps,
                               threshold=threshold,
                               vscale=vscale,
                               scale=scale,
                               debug=debug)
    if debug:
        debug_show([bottom, top, boxmap], "seeds")

    # spread the text line seeds to all the remaining
    # components
    if verbose: print("propagating labels")
    llabels = morph.propagate_labels(boxmap, seeds, conflict=0)
    if verbose: print("spreading labels")
    spread = morph.spread_labels(seeds,maxdist=scale)
    llabels = np.where(llabels>0,llabels,spread*binary)
    segmentation = llabels*binary
    return segmentation
