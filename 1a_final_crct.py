import numpy as np
import cv2

###########IMAGE READ####################
im=cv2.imread("G://CS 543-Computer Vision//Assignment 1//data-basic alignment//00153v.jpg",cv2.IMREAD_GRAYSCALE)
b=im[:im.shape[0]//3,:]
g=im[im.shape[0]//3:2*(im.shape[0]//3),:]
r=im[2*(im.shape[0]//3):,:]

########resize###########
if b.shape[0]==g.shape[0]==r.shape[0]:
    pass
else:
    r=cv2.resize(r,(b.shape[1],b.shape[0]))


#neglect borders (preprocess)
b = b[45:-45, 45:-45]
g = g[45:-45, 45:-45]
r = r[45:-45, 45:-45]
########################################################

'''
########### ALGO ##################
1.warp affine g,r in two separate loops
2.choose ROI for each r,g- ROI is inside the displacement border
3.do NCC & store their individual best disp vectors
4.compare which disp is largest and crop all three in that direction
#########################################
'''

disp=np.arange(-15,16)
best_disp_r,best_disp_g=(),()
gNCC_score_prev=float("-inf")
rNCC_score_prev=float("-inf")
zero_count=0
#####GREEN SHIFT######
for dx in disp:
    for dy in disp:
        
        #displacement
        translation_matrix=np.float32([[1,0,dx],[0,1,dy]])
        green_shifted = cv2.warpAffine(g, translation_matrix, (g.shape[1], g.shape[0]))
        #####ROI selection########
        # if dx>0:
        #     gcopy = green_shifted[:, dx:]
        #     b_g = b[:, dx:]
        # elif dx<0:
        #     gcopy=green_shifted[:,:dx]
        #     b_g = b[:, :dx]
        # else:   
        #     pass    #no movement along height

        # if dy>0:
        #     gcopy = gcopy[dy:, :]
        #     b_g = b_g[dy:, :]
        # elif dy<0:
        #     gcopy=gcopy[:dy, :]
        #     b_g = b_g[:dy, :]
        # else:   
        #     pass    #no movement along width

        if dx > 0 and dy > 0:
            gcopy = green_shifted[dy:, dx:]
            b_g = b[dy:, dx:]
        elif dx > 0 and dy < 0:
            gcopy = green_shifted[:dy, dx:]
            b_g = b[:dy, dx:]
        elif dx < 0 and dy > 0:
            gcopy = green_shifted[dy:, :dx]
            b_g = b[dy:, :dx]
        elif dx < 0 and dy < 0:
            gcopy = green_shifted[:dy, :dx]
            b_g = b[:dy, :dx]
        else:  # No shift in either direction
            gcopy = green_shifted
            b_g = b
        
        ##############NCC Scoring############
        gflat = gcopy.flatten().astype(np.float32)
        bflat = b_g.flatten().astype(np.float32)
        gNCC_numerator = np.sum((gflat - np.mean(gflat)) * (bflat - np.mean(bflat)))
        gNCC_denominator = np.sqrt(np.sum((gflat - np.mean(gflat))**2) * np.sum((bflat - np.mean(bflat))**2))
        gNCC_score_curr =gNCC_numerator / gNCC_denominator if gNCC_denominator != 0 else 0

        #condition
        if gNCC_score_curr > gNCC_score_prev:
            best_disp_g=(dx,dy)
            gNCC_score_prev=gNCC_score_curr


#####RED SHIFT######
for dx in disp:
    for dy in disp:
        
        #displacement
        translation_matrix=np.float32([[1,0,dx],[0,1,dy]])
        red_shifted = cv2.warpAffine(r, translation_matrix, (r.shape[1], r.shape[0]))

        # #####ROI selection########
        # if dx>0:
        #     rcopy = red_shifted[:, dx:]
        #     b_r = b[:, dx:]
        # elif dx<0:
        #     rcopy=red_shifted[:,:dx]
        #     b_r = b[:, :dx]
        # else:
        #     pass

        # if dy>0:
        #     rcopy = rcopy[dy:, :]
        #     b_r = b_r[dy:, :]
        # elif dy<0:
        #     rcopy=rcopy[:dy, :]
        #     b_r = b_r[:dy, :]
        # else:
        #     pass

        if dx > 0 and dy > 0:
            rcopy = red_shifted[dy:, dx:]
            b_r = b[dy:, dx:]
        elif dx > 0 and dy < 0:
            rcopy = red_shifted[:dy, dx:]
            b_r = b[:dy, dx:]
        elif dx < 0 and dy > 0:
            rcopy = red_shifted[dy:, :dx]
            b_r = b[dy:, :dx]
        elif dx < 0 and dy < 0:
            rcopy = red_shifted[:dy, :dx]
            b_r = b[:dy, :dx]
        else:  # No shift in either direction
            rcopy = red_shifted
            b_r = b

        ##############NCC Scoring############
        rflat = rcopy.flatten().astype(np.float32)
        bflat = b_r.flatten().astype(np.float32)
        rNCC_numerator = np.sum((rflat - np.mean(rflat)) * (bflat - np.mean(bflat)))
        rNCC_denominator = np.sqrt(np.sum((rflat - np.mean(rflat))**2) * np.sum((bflat - np.mean(bflat))**2))
        rNCC_score_curr =rNCC_numerator / rNCC_denominator if rNCC_denominator != 0 else 0

        #condition
        if rNCC_score_curr > rNCC_score_prev:
            best_disp_r=(dx,dy)
            rNCC_score_prev=rNCC_score_curr

#########Crop in direction of largest displacement###########
#crop along Width
if abs(best_disp_r[0]) >= abs(best_disp_g[0]):
    print("r has smaller width")
    if best_disp_r[0]>0:
        b=b[:, best_disp_r[0]:]
        g=g[:, best_disp_r[0]:]
        r=r[:, best_disp_r[0]:]
    elif best_disp_r[0]<0:
        b=b[:, :best_disp_r[0]]
        g=g[:, :best_disp_r[0]]
        r=r[:, :best_disp_r[0]]
    else:
        pass

else:
    print("g has smaller width")
    if best_disp_g[0]>0:
        b=b[:, best_disp_g[0]:]
        r=r[:, best_disp_g[0]:]
        g=g[:, best_disp_g[0]:]
    elif best_disp_g[0]<0:
        b=b[:, :best_disp_g[0]]
        r=r[:, :best_disp_g[0]]
        g=g[:, :best_disp_g[0]]
    else:
        pass

#crop along Height
if abs(best_disp_r[1]) >= abs(best_disp_g[1]):
    print("r has smaller height")
    if best_disp_r[1]>0:
        b=b[best_disp_r[1]:, :]
        g=g[best_disp_r[1]:, :]
        r=r[best_disp_r[1]:, :]
    elif best_disp_r[1]<0:
        b=b[:best_disp_r[1], :]
        g=g[:best_disp_r[1], :]
        r=r[:best_disp_r[1], :]
    else:
        pass
else:
    print("g has smaller height")
    if best_disp_g[1]>0:
        b=b[best_disp_g[1]:, :]
        r=r[best_disp_g[1]:, :]
        g=g[best_disp_g[1]:, :]
    elif best_disp_g[1]<0:
        b=b[:best_disp_g[1], :]
        r=r[:best_disp_g[1], :]
        g=g[:best_disp_g[1], :]
    else:
        pass

#merge to get final image
final_img=cv2.merge([b,g,r])

############IMAGE SHOW###################
# cv2.imshow("OG",im)
# cv2.imshow("b",b)
# cv2.imshow("g",g)
# cv2.imshow("r",r)
print("Best Disp.Vector Red(dx,dy)={},Disp.Vector Green(dx,dy)={}".format(best_disp_r,best_disp_g))
cv2.imshow("final img",final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
