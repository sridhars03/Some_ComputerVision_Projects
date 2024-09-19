#imports
import cv2
import numpy as np

#to calc ssd
def calc_ssd(b, gr):
    #Sum of Squared Differences(SSD)
    return np.sum((b - gr) ** 2)

#to calc ncc
def calc_ncc(b, gr):
    #Normalized Cross-Correlation(NCC)
    b=(b-np.mean(b)) / (np.std(b)+1e-7)     #1e-7  ---- incase std is 0.
    gr=(gr-np.mean(gr)) / (np.std(gr)+1e-7)
    return np.sum(b * gr)  #dot product between two imgs

def align_simple(b,gr,metric,disp_size=15):
    """
    ##### Algo decription #########

    This algo uses NCC or SSD method as metric.. 
    moves the indie color channel over B
    Scores it, finds the best score + displacement
    returns the displacement for each color channel
    """
    
    best_disp = (0, 0)
    best_score_ncc=float("-inf")
    best_score_ssd=float("inf")
    
    disp = np.arange(-disp_size,disp_size+1)    #-15 to 16 in this code

    for dx in disp:
        for dy in disp:
            #this shifts G, R imgs
            shift_gr=np.roll(np.roll(gr, dx, axis=1),dy,axis=0)
            
            #cropping to eliminate boundary artifacts
            cropped_gr = shift_gr[disp_size+15:-(disp_size+15), disp_size+15:-(disp_size+15)]
            cropped_b = b[disp_size+15:-(disp_size+15), disp_size+15:-(disp_size+15)]
            
            if metric=='SSD':
                currscore = calc_ssd(cropped_b, cropped_gr)
                #lower SSD score is better
                if currscore < best_score_ssd:  
                    best_score_ssd = currscore
                    best_disp=(dx, dy)
            elif metric == 'NCC':
                currscore = calc_ncc(cropped_b, cropped_gr)
                #higher NCC score is better
                if currscore > best_score_ncc:  
                    best_score_ncc = currscore
                    best_disp=(dx, dy)
    
    return best_disp

def colorize_img(img, metric='SSD', disp_size=15):
    #Split the image into B,G,R
    h_each=img.shape[0] // 3
    b = img[:h_each, :]
    g = img[h_each:2*h_each, :]
    r = img[2*h_each:, :]  

    ########resize###########   in case total height isn't a multiple of 3
    if b.shape[0]==g.shape[0]==r.shape[0]:
        pass
    else:
        r=cv2.resize(r,(b.shape[1],b.shape[0]))

    print(b.shape,g.shape,r.shape)
    ############calc scores and align G, R #################
    best_disp_g = align_simple(b, g, metric, disp_size)
    best_disp_r = align_simple(b, r, metric, disp_size)

    #use the disp to align G,R finally
    #so this shifts the pixels with corresponding vectors first along columns, then rows
    shift_gr_g=np.roll(np.roll(g, best_disp_g[0], axis=1), best_disp_g[1], axis=0)
    shift_gr_r=np.roll(np.roll(r, best_disp_r[0], axis=1), best_disp_r[1], axis=0)

    #merging to get color pic
    final_img=cv2.merge([b,shift_gr_g,shift_gr_r])
    
    save_name="G://CS 543-Computer Vision//Some_ComputerVision_Projects//Results//final_img.jpg"
    cv2.imwrite(save_name, final_img)

    return final_img,best_disp_g,best_disp_r

def viz(img):
    cv2.imshow("final_img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    img=cv2.imread("G://CS 543-Computer Vision//Assignment 1//data-basic alignment//01112v.jpg",cv2.IMREAD_GRAYSCALE)
    metric='NCC'
    disp_size=15    #u can change it for different imgs
    final_image,disp_g,disp_r=colorize_img(img, metric=metric, disp_size=disp_size)

    viz(final_image)
    print("Best displacement against Blue:  Green={}  |  Red={}".format(disp_g,disp_r))

if __name__ == "__main__":
    main()
