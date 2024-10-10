#imports
import numpy as np
import scipy 
import skimage
from skimage.transform import resize
import matplotlib.pyplot as plt
import cv2 
from scipy.ndimage import gaussian_laplace, rank_filter,maximum_filter
import time
from matplotlib.patches import Circle

#######################################
def applyLOG(img,sigma):
    imglog=gaussian_laplace(img,sigma)
    return imglog

def applyDOG(img,sigmas):
    """
    DOG = Difference of Gaussian at each scale
    """
    num_scales=len(sigmas)
    DOG_pyramid,gaussian_pyramid=[],[]
    
    #this will create gaussian pyramid
    for sigma in sigmas:
        gaussian_blur = cv2.GaussianBlur(img, (0, 0), sigma)
        gaussian_pyramid.append(gaussian_blur)
    
    #DOG pyramid
    for i in range(1, num_scales):
        DOG=gaussian_pyramid[i] - gaussian_pyramid[i-1]
        DOG_pyramid.append(DOG)
    
    return DOG_pyramid,sigmas
########################################

def show_all_circles_harris(image, cx, cy, rad, color='r'):
    """
    image: numpy array, representing the grayscsale image
    cx, cy: numpy arrays or lists, centers of the detected blobs
    rad: numpy array or list, radius of the detected blobs
    """
    _, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(image, cmap='gray')
    for x, y, r in zip(cx, cy,rad):
        circ = Circle((x, y), r, color=color, fill=False)
        ax.add_patch(circ)

    plt.title('%i circles | After Harris Detection' % len(cx))
    plt.show()


######### use after Harris
def show_all_circles(image, cx, cy, rad, color='r'):
    """
    image: numpy array, representing the grayscsale image
    cx, cy: numpy arrays or lists, centers of the detected blobs
    rad: numpy array or list, radius of the detected blobs
    """
    _, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(image, cmap='gray')
    for x, y, r in zip(cx, cy,rad):
        circ = Circle((x, y), r, color=color, fill=False)
        ax.add_patch(circ)

    plt.title('%i circles' % len(cx))
    plt.show()

def viz_detection_scale(n_circles):
    #this is a plot for: Scale vs Num-of-Detected points
    plt.figure(figsize=(8,5))
    plt.plot(list(n_circles.keys()), list(n_circles.values()), marker='o', linestyle='-', color='b')
    plt.title("Scale vs No of Detected Points")
    plt.xlabel("Scale")
    plt.ylabel("Number of Detected Points")
    plt.show()

########### Unique circles ############
def get_unique_circles(detected_points, radii, min_dist_threshold=10, radius_threshold=1):
    """
    This funcion filters out duplicate circles based on proximity in position and radius similarity.
    """
    uniquecx,uniquecy,uniquerad=[],[],[]

    cx = np.concatenate([det[0] for det in detected_points])
    cy = np.concatenate([det[1] for det in detected_points])
    radii = np.concatenate(radii)

    all_points=np.stack((cx, cy, radii), axis=-1)
    
    #thi will only keep unique ones
    for i, (x, y, r) in enumerate(all_points):
        is_unique=True
        for j in range(len(uniquecx)):
            dist=(uniquecx[j]-x)**2 + (uniquecy[j]-y)**2
            radius_diff=np.abs(uniquerad[j]-r)
            
            #if current point is close to any previous point
            if dist < min_dist_threshold and radius_diff < radius_threshold:
                is_unique = False   #its not unique
                break
        
        if is_unique:
            uniquecx.append(x)
            uniquecy.append(y)
            uniquerad.append(r)
    
    return np.array(uniquecx), np.array(uniquecy), np.array(uniquerad)

################# harris filter ###################
def harris_detect(img, detected_points, rad, harris_threshold=0.0001):
    """
    Harris corner detection to reject edge blobs. Blob detection is prone to detect edges also!
    """
    harris_img = cv2.dilate(cv2.cornerHarris(np.float32(img),blockSize=2,ksize=3,k=0.04), None)

    #Harris threshold -significant Harris response
    filtered_cx,filtered_cy,filtered_radii=[],[],[]
    for _, (cx, cy, r) in enumerate(zip(detected_points[0], detected_points[1], rad)):
        if harris_img[int(np.round(cy)), int(np.round(cx))] > harris_threshold:  
            filtered_cx.append(cx)
            filtered_cy.append(cy)
            filtered_radii.append(r)
    return np.array(filtered_cx), np.array(filtered_cy), np.array(filtered_radii)
##############################################

def blob_detect(img,sigmalog,method="scale"):
    
    '''
    Algo 1: Scaling up the kernel, keeping the image size constant
    1.Generate a Laplacian of Gaussian filter
    2.Build a Laplacian scale space, starting with some initial scale and going for n iterations:
        Filter image with scale-normalized Laplacian at current scale.
        Save square of Laplacian response for current level of scale space.
        Increase scale by a factor k.
    3.Perform nonmaximum suppression in scale space.
    4.Display resulting circles at their characteristic scales
    '''

    '''
    Algo 2: Downsampling the image, keeping the scale constant - generally proven to be faster(smaller image operation)
    '''

    rad,detected_points = [],[]
    n=10       #10 levels in scale pyramid
    k=1.35

    ######## Method 1: using different scales ########
    if method == "scale":
        print("Chosen method: Increase filter size-Same img size")
        
        t1 = time.time()
        scale_space = np.empty((img.shape[0],img.shape[1],n))
        
        for i in range(n):
            sigma=sigmalog*(k**i)
            scale_space[:,:,i] = ((sigma**2) * (applyLOG(img,sigma)))**2         #this is squared LOG response
            
            ####### threshold here to choose detected regions ##########
            filtered_image = rank_filter(scale_space[:,:,i],rank=-1,size=9)   
            
            nms_result = np.where(scale_space[:,:,i] == filtered_image, scale_space[:,:,i], 0)  #this ll have 0s and non 0s(detected points). 

            #thresholding
            #increasing threshold reduces no of detections
            row,col=np.where(nms_result>0.028)
            
            radius_from_sigma=np.round((sigma)*np.sqrt(2) ,4)
            
            #radius for detected points and storing
            rad.append(radius_from_sigma * np.ones(len(col)))
            detected_points.append((col, row))                

        #getting unique circles
        uniquecx, uniquecy, uniquerad = get_unique_circles(detected_points, rad)
        
        print("Method 1 runtime:",np.round(time.time()-t1,5),"seconds")


    else:
        print("Chosen method: Downsample img size, Same filter size")
        
        t1=time.time()
        scale_space = np.empty(n, dtype=object)
        for i in range(n):
            downsampled_img = resize(img, (int(img.shape[0] / (1.2**i)), int(img.shape[1] / (1.2**i))))
            
            #resizing to img.shape after downsampling and applying LOG filter
            scale_space[i] = resize(((sigmalog**2) * (applyLOG(downsampled_img,sigmalog)))**2 , img.shape, order=1)  # Bilinear interpolation  + keepign the scale constant
            
            filtered_image = rank_filter(scale_space[i],rank=-1,size=9)   
            
            nms_result = np.where(scale_space[i] == filtered_image, scale_space[i], 0)  #this ll have 0s and non 0s(detected points). 
            
            #thresholding
            #increasing threshold reduces no of detections
            row,col=np.where(nms_result>0.035)
            
            #constant radius all the loops - see what shd be changed
            radius_from_sigma=np.round((sigmalog*(k**i))*np.sqrt(2) ,4)
            
            #radius for detected points and storing
            rad.append(radius_from_sigma * np.ones(len(col)))
            detected_points.append((col,row))
        
        #getting unique circles
        uniquecx,uniquecy,uniquerad=get_unique_circles(detected_points,rad)
        
        print("Method 2 runtime:",np.round(time.time()-t1,5),"seconds")

    n_circles,count={},0
    for det_points,listofradius in zip(detected_points,rad):
        cols,rows = det_points
        n_circles[np.round(sigmalog*(k**count) , 4)] = len(cols)
        count+=1
    

    show_all_circles(img, uniquecx, uniquecy, uniquerad)
    viz_detection_scale(n_circles)

    harris_cx, harris_cy, harris_rad = harris_detect(img, (uniquecx, uniquecy), uniquerad, harris_threshold=0.00001)
    print("After Harris detection")
    show_all_circles_harris(img, harris_cx, harris_cy, harris_rad)
###################################################################


###################### DOG method ############################
def blob_detect_DoG(img, sigmas, threshold=0.035):
    DOG_pyramid,sigmas=applyDOG(img,sigmas)
    cx,cy  = [],[] 
    radii=[]
    for i in range(1,len(DOG_pyramid)-1):
        curr_img = DOG_pyramid[i]
    
        #applying NMS here
        maxima=maximum_filter(curr_img,size=5)
        maxima=np.logical_and(maxima==curr_img,curr_img>threshold)
        
        for r,c in zip(*np.where(maxima)):
            cx.append(c)
            cy.append(r)
            radii.append(sigmas[i]*np.sqrt(2))
    
    return np.array(cx), np.array(cy), np.array(radii)

def main():
    filepath='G://CS 543-Computer Vision//mp2//mp2//images/butterfly.jpg'
    img=cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    img=np.round((img-np.min(img))/(np.max(img)-np.min(img)),5)
    
    sigmas=[1.0, 1.5, 2.0, 3.0,3.5,4.0,4.5,5.0]  #Scales for DOG method
    sigmalog=2      #starting sigma
    
    blob_detect(img,sigmalog,method="scale")             #method 1 = img size same, filter size increase
    blob_detect(img,sigmalog,method="downsample")         #method 2 = downsample img size, filter size same

    ######## DOG calling function ###########   DOG = Difference of Gaussian method
    t1 = time.time()
    cxDOG, cyDOG,radii_fromDoG=blob_detect_DoG(img,sigmas)
    t_DOG=time.time()-t1
    print("Runtime of DOG method=",np.round(t_DOG,5), " seconds")
    show_all_circles(img, cxDOG, cyDOG, radii_fromDoG)
    

if __name__ =="__main__":
    main()