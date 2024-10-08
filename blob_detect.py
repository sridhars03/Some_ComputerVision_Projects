# Libraries you will find useful
import numpy as np
import scipy 
import skimage
import matplotlib.pyplot as plt
import cv2 
from scipy.ndimage import gaussian_laplace, rank_filter, generic_filter
import time
from matplotlib.patches import Circle

def applyLOG(img,sigma):
    imglog=gaussian_laplace(img,sigma)
    return imglog


def show_all_circles(image, cx, cy, rad, color='r'):
    """
    image: numpy array, representing the grayscsale image
    cx, cy: numpy arrays or lists, centers of the detected blobs
    rad: numpy array or list, radius of the detected blobs
    """
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(image, cmap='gray')
    for x, y, r in zip(cx, cy,rad):
        circ = Circle((x, y), r, color=color, fill=False)
        ax.add_patch(circ)

    plt.title('%i circles' % len(cx))
    plt.show()

def viz_detection_scale(n_circles):

    #this is a plot: Scale vs Num-of-Detected points
    plt.figure(figsize=(8,5))
    plt.plot(list(n_circles.keys()), list(n_circles.values()), marker='o', linestyle='-', color='b')
    plt.title("Scale vs No of Detected Points")
    plt.xlabel("Scale")
    plt.ylabel("Number of Detected Points")
    plt.show()


def blob_detect(img,sigmalog):
    
    '''
    Algo idea:
    1.Generate a Laplacian of Gaussian filter
    2.Build a Laplacian scale space, starting with some initial scale and going for n iterations:
        Filter image with scale-normalized Laplacian at current scale.
        Save square of Laplacian response for current level of scale space.
        Increase scale by a factor k.
    3.Perform nonmaximum suppression in scale space.
    4.Display resulting circles at their characteristic scales
    '''

    ######## Method 1: using different scales ########
    n=10       #10 levels in scale pyramid
    scale_space = np.empty((img.shape[0],img.shape[1],n))

    detected_points = []
    rad=[]
    for i in range(n):
        scale_space[:,:,i] = (((sigmalog*(1.1**i))**2) * (applyLOG(img,(sigmalog*(1.1**i)))))**2         #this is squared LOG response
        
        ####### threshold here to choose detected regions ##########
        filtered_image=rank_filter(scale_space[:,:,i],rank=-1,size=9)
        
        nms_result=np.where(scale_space[:,:,i] == filtered_image, scale_space[:,:,i], 0)  #this ll have 0s and non 0s(detected points). 

        #thresholding
        row,col=np.where(nms_result>0.01)
        radius_from_sigma=np.round((sigmalog*(1.1**i))*np.sqrt(2) ,4)
        
        #radius for detected points and storing
        rad.append(radius_from_sigma*np.ones(len(col)))
        detected_points.append((col,row))
    
    n_circles,count={},0
    #draw circles for detected regions - each pic=each scale
    for det_points,listofradius in zip(detected_points,rad):
        cols,rows=det_points
        n_circles[np.round(sigmalog*(1.1**count) , 4)]=len(cols)
        show_all_circles(img,cols,rows,listofradius)
        count+=1
    
    print(n_circles)
    viz_detection_scale(n_circles)


def main(): 
    filepath='G://CS 543-Computer Vision//mp2//mp2//images/butterfly.jpg'
    img=cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    img=np.round((img-np.min(img))/(np.max(img)-np.min(img)),5)
    sigmalog=2      #starting sigma
    
    blob_detect(img,sigmalog)
    


    # cv2.imshow("og",img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ =="__main__":
    main()