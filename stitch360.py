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

def _upsampled_dft(data, ups,
                    upsample_factor=1, axis_offsets=None):

    im2pi = 1j * 2 * np.pi
    tdata = data.copy()
    kernel = (cp.tile(cp.arange(ups), (data.shape[0], 1))-axis_offsets[:, 1:2])[
        :, :, None]*cp.fft.fftfreq(data.shape[2], upsample_factor)
    kernel = cp.exp(-im2pi * kernel)
    tdata = cp.einsum('ijk,ipk->ijp', kernel, tdata)
    kernel = (cp.tile(cp.arange(ups), (data.shape[0], 1))-axis_offsets[:, 0:1])[
        :, :, None]*cp.fft.fftfreq(data.shape[1], upsample_factor)
    kernel = cp.exp(-im2pi * kernel)
    rec = cp.einsum('ijk,ipk->ijp', kernel, tdata)

    return rec

def registration_shift(src_image, target_image, upsample_factor=1, space="real"):

    # assume complex data is already in Fourier space
    if space.lower() == 'fourier':
        src_freq = src_image
        target_freq = target_image
    # real data needs to be fft'd.
    elif space.lower() == 'real':
        src_freq = cp.fft.fft2(src_image)
        target_freq = cp.fft.fft2(target_image)

    # Whole-pixel shift - Compute cross-correlation by an IFFT
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    cross_correlation = cp.fft.ifft2(image_product)
    A = cp.abs(cross_correlation)
    maxima = A.reshape(A.shape[0], -1).argmax(1)
    maxima = cp.column_stack(cp.unravel_index(maxima, A[0, :, :].shape))

    midpoints = np.array([cp.fix(axis_size / 2)
                            for axis_size in shape[1:]])

    shifts = cp.array(maxima, dtype=cp.float64)
    ids = cp.where(shifts[:, 0] > midpoints[0])
    shifts[ids[0], 0] -= shape[1]
    ids = cp.where(shifts[:, 1] > midpoints[1])
    shifts[ids[0], 1] -= shape[2]
    
    if upsample_factor > 1:
        # Initial shift estimate in upsampled grid
        shifts = np.round(shifts * upsample_factor) / upsample_factor
        upsampled_region_size = np.ceil(upsample_factor * 1.5)
        # Center of output array at dftshift + 1
        dftshift = np.fix(upsampled_region_size / 2.0)

        normalization = (src_freq[0].size * upsample_factor ** 2)
        # Matrix multiply DFT around the current shift estimate

        sample_region_offset = dftshift - shifts*upsample_factor
        cross_correlation = _upsampled_dft(image_product.conj(),
                                                upsampled_region_size,
                                                upsample_factor,
                                                sample_region_offset).conj()
        cross_correlation /= normalization
        # Locate maximum and map back to original pixel grid
        A = cp.abs(cross_correlation)
        maxima = A.reshape(A.shape[0], -1).argmax(1)
        maxima = cp.column_stack(
            cp.unravel_index(maxima, A[0, :, :].shape))

        maxima = cp.array(maxima, dtype=cp.float64) - dftshift

        shifts = shifts + maxima / upsample_factor
            
    return shifts

def registration_shift_batch(u, w, upsample_factor=1, space="real"):
    ntheta = u.shape[0]
    ptheta = 1
    res = np.zeros([ntheta, 2], dtype='float32')
    for k in range(0, ntheta//ptheta):
        ids = np.arange(k*ptheta, (k+1)*ptheta)
        # copy data part to gpu
        u_gpu = cp.array(u[ids])
        w_gpu = cp.array(w[ids])
        # Radon transform
        res_gpu = registration_shift(
            u_gpu, w_gpu, upsample_factor, space)
        # copy result to cpu
        res[ids] = res_gpu.get()
    return res

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
    parser.add_argument("--axis", nargs='?', type=str, default="100", help="Approximate rotation axis location (pixel): 10.0 (default 10 image horizontal size)")
           
    args = parser.parse_args()
    apr_center = np.int(args.axis)

    # Read data
    proj, flat, dark, theta = dxchange.read_aps_32id(
        args.fname, sino=(0, 1000),proj = (0,3000,750))        
    print(theta[np.arange(0,3000,750)])
    # filter data        
    data = tomopy.normalize(proj, flat, dark)                
    # stitched data            
    w = apr_center*2
    [ntheta,nz,n] = data.shape    
    datap1 = data[:ntheta//2]    
    datap2 = data[ntheta//2:,:,::-1]    
    # dxchange.write_tiff(datap1,'t1.tiff',overwrite=True)
    # dxchange.write_tiff(datap2,'t2.tiff',overwrite=True)
    # dxchange.write_tiff(datap1[:,:,0:w]-np.mean(datap1[:,:,0:w]),'tc1.tiff',overwrite=True)
    # dxchange.write_tiff(datap2[:,:,-w:]-np.mean(datap2[:,:,-w:]),'tc2.tiff',overwrite=True)
    shift = registration_shift_batch(datap1[:,:,0:w],datap2[:,:,-w:],upsample_factor=4)
    shift[:] = np.mean(shift,axis=0)
    print('Average shift', shift)    
    shift[:,1] = (shift[:,1]-w/2)      
    # resulting data
    datanew = merge(datap1,datap2,shift,ntheta//2)
    # dxchange.write_tiff_stack(datanew,'t/t.tiff',overwrite=True)    
    
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
        shiftnew[:,1] = shift[0,1]
        projnew = merge(projp1,projp2,shiftnew,ntheta//2)
        darknew = merge(darkp1,darkp2,shiftnew,ndark)
        flatnew = merge(flatp1,flatp2,shiftnew,nflat)        
        fid['/exchange/data'][:,k*100:(k+1)*100] = projnew
        fid['/exchange/data_white'][:,k*100:(k+1)*100] = flatnew
        fid['/exchange/data_dark'][:,k*100:(k+1)*100] = darknew        
        