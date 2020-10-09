import numpy as np
import matplotlib.pyplot as plt
import cv2, matplotlib
from matplotlib.patches import Ellipse
from pylab import ginput, show, axis

# READ RGB IMAGE IN OPEN CV
img_cv = cv2.imread('images/IMG_20200904_180629(!).jpg')


# CONVERT TO GRAYSCALE IMAGE
gray_img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

gray_img_plt = cv2.cvtColor(gray_img_cv, cv2.COLOR_GRAY2RGB) 
plt.figure(), plt.imshow(gray_img_plt)

# threshold for grayscale image 
th = int(np.mean(gray_img_plt))
_, th_img_cv = cv2.threshold(gray_img_cv, th, 255, cv2.THRESH_BINARY) 
#cv2.imshow('image',th_img_cv)

th_img_plt = cv2.cvtColor(th_img_cv, cv2.COLOR_GRAY2RGB) 
plt.figure(8), plt.imshow(th_img_plt)
#cv2.imshow('image',th_img_plt)

"""human click on the drop let""" 
clicks=2
print("Please click two(2) times")
fig_inp = ginput(clicks) # it will wait for two clicks

for iclick in range(clicks): 
	plt.plot(fig_inp[iclick][0],fig_inp[iclick][1],'bo')

xdata = np.array([p[0] for p in fig_inp]) 
ydata = np.array([p[1] for p in fig_inp])

yleft=np.array(ydata[0]) 
yright=np.array(ydata[1])

xleft=np.array(xdata[0]) 
xright=np.array(xdata[1])

"""calculating the contact angle"""

m=(yleft-yright)/(xleft-xright) 
rot_angle=int(np.arctan(m)*180/np.pi)

if rot_angle<0 :
	rot_angle+=180

print("the contact angle is",rot_angle) 