import numpy as np
import cv2

###########IMAGE READ####################
im=cv2.imread("G://CS 543-Computer Vision//Assignment 1//data-basic alignment//00125v.jpg",cv2.IMREAD_GRAYSCALE)
print(im.shape)
b=im[:im.shape[0]//3,:]
g=im[im.shape[0]//3:2*(im.shape[0]//3),:]
r=im[2*(im.shape[0]//3):,:]

########resize###########
if b.shape[0]==g.shape[0]==r.shape[0]:
    pass
else:
    r=cv2.resize(r,(b.shape[1],b.shape[0]))

#neglect borders (make it automatic)
b = b[20:b.shape[0]-20, 20:b.shape[1]-20]
g = g[20:g.shape[0]-20, 20:g.shape[1]-20]
r = r[20:r.shape[0]-20, 20:r.shape[1]-20]

##########PIXEL SHIFT for R,G ##############
disp=np.arange(-15,16)
count=0
gr_ncc_score_min,rg_ncc_score_min=float("-inf"), float("-inf")
disp_vector_bgr,disp_vector_rgb=(0,0),(0,0)

#displace G/R, merge, take ncc with B, merge the max NCC imgs
for dx in disp:
    for dy in disp:
        translation_matrix=np.float32([[1,0,dx],[0,1,dy]])
        # Apply the translation to the Green and Red channels
        green_shifted = cv2.warpAffine(g, translation_matrix, (g.shape[1], g.shape[0]))
        red_shifted = cv2.warpAffine(r, translation_matrix, (r.shape[1], r.shape[0]))

        #merge G/R
        gr=cv2.merge([green_shifted,red_shifted])
        rg=cv2.merge([red_shifted,green_shifted])

        grayscale_image = 0.587 * green_shifted + 0.299 * red_shifted
        grayscale_image = np.clip(grayscale_image, 0, 255).astype(np.uint8)
        
        print(gr.shape, b.shape)

        ###########NCC calc#############
        grflat = gr.flatten().astype(np.float32)
        rgflat = rg.flatten().astype(np.float32)
        bflat = b.flatten().astype(np.float32)
        
#         gr_NCC_numerator = np.sum((grflat - np.mean(grflat)) * (bflat - np.mean(bflat)))
#         gr_NCC_denominator = np.sqrt(np.sum(grflat - np.mean(grflat)**2) * np.sum(bflat - np.mean(bflat)**2))
#         gr_NCC_score = gr_NCC_numerator / gr_NCC_denominator if gr_NCC_denominator != 0 else 0

#         rg_NCC_numerator = np.sum((rgflat - np.mean(rgflat)) * (bflat - np.mean(bflat)))
#         rg_NCC_denominator = np.sqrt(np.sum(rgflat - np.mean(rgflat)**2) * np.sum(bflat - np.mean(bflat)**2))
#         rg_NCC_score = rg_NCC_numerator / rg_NCC_denominator if rg_NCC_denominator != 0 else 0

#         if gr_NCC_score > gr_ncc_score_min:
#             bgr=cv2.merge([b,green_shifted,red_shifted])
#             disp_vector_bgr=(dx,dy)
#             gr_ncc_score_min = gr_NCC_score

#         if rg_NCC_score > rg_ncc_score_min:
#             rgb=cv2.merge([red_shifted,green_shifted,b])
#             disp_vector_rgb=(dx,dy)
#             rg_ncc_score_min = rg_NCC_score


# ############IMAGE SHOW###################
# # cv2.imshow("OG",im)
# # cv2.imshow("b",b)
# # cv2.imshow("g",g)
# # cv2.imshow("r",r)
# #finally bgr vs rgb
# if gr_NCC_score > rg_NCC_score:
#     cv2.imshow("bgr",bgr)
# else:
#     cv2.imshow("rgb",rgb)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
