import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import time

#FFT algo
def align_fft(b, gr):
    #using Gaussian filter to highlight edges here
    # b_filtered = cv2.filter2D(b, -1, cv2.getGaussianKernel(ksize=5, sigma=1))
    # gr_filtered = cv2.filter2D(gr, -1, cv2.getGaussianKernel(ksize=5, sigma=1))
    b_filtered=b
    gr_filtered=gr
    #apply FFT
    fft_b = np.fft.fft2(b_filtered)
    fft_gr = np.fft.fft2(gr_filtered)
    
    #apply fft shift
    fft_b_shift = np.fft.fftshift(fft_b)
    fft_gr_shift = np.fft.fftshift(fft_gr)
    
    #conjugate    
    fft_gr_conj = np.conjugate(fft_gr_shift)
    product=fft_b_shift*fft_gr_conj

    #apply inverse fft - to get peak location
    inverse_fft=np.fft.ifft2(product/(np.abs(product) + 1e-8))
    inverse_fft_shift=np.fft.ifftshift(inverse_fft)
    corr=np.abs(inverse_fft_shift)

    #displacement obtained from peak of correlation
    max_loc=np.unravel_index(np.argmax(corr),corr.shape)
    dx,dy=max_loc[0] - b.shape[0] // 2, max_loc[1] - b.shape[1] // 2

    return dx,dy,corr

def colorize_img_fft(img):
    #Split the image into B,G,R
    h_each=img.shape[0] // 3
    b=img[:h_each, :]
    g=img[h_each:2*h_each, :]
    r=img[2*h_each:, :]

    ########resize###########   in case total height isn't a multiple of 3
    if not (b.shape[0] == g.shape[0] == r.shape[0]):
        r = cv2.resize(r, (b.shape[1], b.shape[0]))

    #align green-blue
    ti=time.time()
    dx_g,dy_g,correlation_g=align_fft(b, g)
    g_aligned=np.roll(g,shift=(dx_g,dy_g),axis=(0,1))   #applying the disp found    

    #align red-blue
    dx_r,dy_r,correlation_r=align_fft(b, r)
    tf=time.time()
    print(tf-ti)
    r_aligned=np.roll(r, shift=(dx_r, dy_r), axis=(0, 1))   #applying the disp found

    #merging
    final_img=cv2.merge([b,g_aligned,r_aligned])

    return final_img,(dx_g,dy_g),(dx_r,dy_r),correlation_g,correlation_r

def viz_correlation(correlation, title):
    plt.imshow(correlation,cmap='hot')
    plt.title(title)
    plt.colorbar()
    plt.show()

def main():
    filepath="your path"
    img=cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    
    #FFT-based colorizing
    final_img,disp_g,disp_r,correlation_g,correlation_r=colorize_img_fft(img)

    #save
    file_name = os.path.splitext(os.path.basename(filepath))[0]
    savefilename="yourpath" + file_name + ".jpg"
    cv2.imwrite(savefilename, final_img)

    #viz
    viz_correlation(correlation_g,"Green Channel Correlation")
    viz_correlation(correlation_r,"Red Channel Correlation")

    cv2.imshow("Final Aligned Image",final_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #print best displacements for G,R channels against B
    print("Best displacement against Blue:  Green={}  |  Red={}".format(disp_g,disp_r))

if __name__ == "__main__":
    main()
