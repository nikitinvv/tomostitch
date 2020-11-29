import numpy as np
import dxchange
import h5py
import argparse
import tomopy
import scipy.signal
import concurrent.futures as cf
import threading
from scipy import ndimage
from functools import partial
import cupy as cp
import cv2

def apply_shift(psi, p):
    """Apply shift for all projections."""
    [ntheta,nz,n] = psi.shape
    tmp = cp.zeros([ntheta, 2*nz, 2*n], dtype='float32')
    tmp[:, nz//2:3*nz//2, n//2:3*n//2] = psi
    [x, y] = cp.meshgrid(cp.fft.rfftfreq(2*n),
                            cp.fft.fftfreq(2*nz))
    shift = cp.exp(-2*cp.pi*1j *
                    (x*p[:, 1, None, None]+y*p[:, 0, None, None]))
    res0 = cp.fft.irfft2(shift*cp.fft.rfft2(tmp))
    res = res0[:, nz//2:3*nz//2, n//2:3*n//2]
    return res

def apply_shift_batch(u, p):
    [ntheta,nz,n] = u.shape
    ptheta = 1
    res = np.zeros([ntheta, nz, n], dtype='float32')
    for k in range(0, ntheta//ptheta):
        ids = np.arange(k*ptheta, (k+1)*ptheta)
        # copy data part to gpu
        u_gpu = cp.array(u[ids])
        p_gpu = cp.array(p[ids])
        # Radon transform
        res_gpu = apply_shift(u_gpu, p_gpu)
        # copy result to cpu
        res[ids] = res_gpu.get()
    return res

def find_min_max(data):
    """Find min and max values according to histogram"""
    
    mmin = np.zeros(data.shape[0],dtype='float32')
    mmax = np.zeros(data.shape[0],dtype='float32')
    
    for k in range(data.shape[0]):
        h, e = np.histogram(data[k][:],1000)
        stend = np.where(h>np.max(h)*0.005)
        st = stend[0][0]
        end = stend[0][-1]        
        mmin[k] = e[st]
        mmax[k] = e[end+1]
     
    return mmin,mmax

def register_shift_sift(datap1,datap2):
    """Find shifts via SIFT detecting features"""

    mmin,mmax = find_min_max(datap1)
    sift = cv2.xfeatures2d.SIFT_create()
    shifts = np.zeros([datap1.shape[0],2],dtype='float32')
    for id in range(datap1.shape[0]):
        tmp1 = ((datap2[id]-mmin[id]) /
                    (mmax[id]-mmin[id])*255)
        tmp1[tmp1 > 255] = 255
        tmp1[tmp1 < 0] = 0
        tmp2 = ((datap1[id]-mmin[id]) /
                (mmax[id]-mmin[id])*255)
        tmp2[tmp2 > 255] = 255
        tmp2[tmp2 < 0] = 0
        # find key points
        tmp1 = tmp1.astype('uint8')
        tmp2 = tmp2.astype('uint8')
        
        kp1, des1 = sift.detectAndCompute(tmp1,None)
        kp2, des2 = sift.detectAndCompute(tmp2,None)
        # cv2.imwrite('original_image_right_keypoints.png',cv2.drawKeypoints(tmp1,kp1,None))
        # cv2.imwrite('original_image_left_keypoints.png',cv2.drawKeypoints(tmp2,kp2,None))
        match = cv2.BFMatcher()
        matches = match.knnMatch(des1,des2,k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.5*n.distance:
                good.append(m)
        draw_params = dict(matchColor=(0,255,0),
                            singlePointColor=None,
                            flags=2)
        tmp3 = cv2.drawMatches(tmp1,kp1,tmp2,kp2,good,None,**draw_params)
        # cv2.imwrite("original_image_drawMatches.jpg", tmp3)
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        shift = (src_pts-dst_pts)[:,0,:]
        shifts[id] = np.mean(shift,axis=0)[::-1]
    print(shifts)
    return shifts

def merge(datap1,datap2,shift,ntheta):
    """Merge projections"""
    datanew = np.zeros([ntheta,nz,2*n],dtype='float32')
    datanew[:,:,n:] = apply_shift_batch(datap1,shift/2)
    datanew[:,:,:n] = apply_shift_batch(datap2,-shift/2)    
    datanew[:,:,:np.int(-shift[0,1]/2+8)] = datanew[:,:,np.int(-shift[0,1]/2+8),None]
    datanew[:,:,-np.int(-shift[0,1]/2+8):] = datanew[:,:,-np.int(-shift[0,1]/2+8),None]
    return datanew    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fname", help="Directory containing an input file name: /data/sample.h5")
    parser.add_argument(
        "foutname", help="Directory containing an output file name: /data/sample.h5")
           
    args = parser.parse_args()
    
    # Read (0,90,180,270) degree angles to find shifts
    proj, flat, dark, theta = dxchange.read_aps_32id(
        args.fname, sino=(0, 1000),proj = (0,3000,750))        
    print(theta[np.arange(0,3000,750)])
    # width of the  projection part used for registration
    w = 400
    data = tomopy.normalize(proj, flat, dark)                
    [ntheta,nz,n] = data.shape    
    datap1 = data[:ntheta//2,:,:w]    
    datap2 = data[ntheta//2:,:,::-1][:,:,-w:]    
    shifts = register_shift_sift(datap1,datap2)
    shifts[:,1] = shifts[:,1]-w
    shifts = np.mean(shifts,axis=0)

    print(shifts)
    
    fid = h5py.File(args.foutname,'w')
    fid.create_dataset('/exchange/data', (3000//2,1000,n*2), chunks=(3000//2,1,2*n),dtype='float32')
    fid.create_dataset('/exchange/data_white', (flat.shape[0],1000,n*2), chunks=(flat.shape[0],1,2*n),dtype='float32')
    fid.create_dataset('/exchange/data_dark', (dark.shape[0],1000,n*2), chunks=(dark.shape[0],1,2*n),dtype='float32')   

    
    for k in range(0,10):
        print(k)
        proj, flat, dark, theta = dxchange.read_aps_32id(
            args.fname, sino=(k*100, (k+1)*100))                
        [ntheta,nz,n] = proj.shape        
        nflat = flat.shape[0]                
        ndark = dark.shape[0]            
        projp1 = proj[:ntheta//2]    
        projp2 = proj[ntheta//2:,:,::-1]    
        flatp1 = flat    
        flatp2 = flat[:,:,::-1]    
        darkp1 = dark    
        darkp2 = dark[:,:,::-1]           
        shiftnew = np.zeros([ntheta,2],dtype='float32') 
        shiftnew[:,1] = shifts[1]
        projnew = merge(projp1,projp2,shiftnew,ntheta//2)
        darknew = merge(darkp1,darkp2,shiftnew,ndark)
        flatnew = merge(flatp1,flatp2,shiftnew,nflat)        
        fid['/exchange/data'][:,k*100:(k+1)*100] = projnew
        fid['/exchange/data_white'][:,k*100:(k+1)*100] = flatnew
        fid['/exchange/data_dark'][:,k*100:(k+1)*100] = darknew        
        