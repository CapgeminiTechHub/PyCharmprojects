import numpy as np
import cv2 as cv

# Read in colour image
img = cv.imread("images/connect4 image 1.jpg")
#img = cv.imread("images/single circle.jpg")
print(img.shape)
print(img.size)
print(img.dtype)


# Change size and resolution of image.
img = cv.resize(img, (360, 640))
print(img.shape)
print(img.size)
print(img.dtype)

# Blur the image
connect4_blurred = cv.GaussianBlur(img,(5,5),2,2)
# Write blurred image
cv.imwrite("images/Connect4_blurred.jpg", connect4_blurred)

# Convert the image to gray
connect4_gray = cv.cvtColor(connect4_blurred, cv.COLOR_BGR2GRAY)

# Do edge detection using Canny method
connect4_edges = cv.Canny(connect4_gray, threshold1=90, threshold2=180)
# Write the image showing edge detection
cv.imwrite("images/Connect4_edges.jpg", connect4_edges)

# Show edge detection image
#cv.imshow('detected circles',connect4_edges)
#cv.waitKey(0)

# Detect circles using Hough Circle Transform
rows = img.shape[0]
print(rows)


circles = cv.HoughCircles(connect4_gray, cv.HOUGH_GRADIENT,1.0,minDist=20, param1=50, param2=20,minRadius=15,maxRadius=21)

#circles = cv.HoughCircles(connect4_edges, cv.HOUGH_GRADIENT,dp=1.5, minDist=80, minRadius=20, maxRadius=80)

##circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT,1,200,param1=50, param2=10,minRadius=0,maxRadius=0)

# Get count of the number of circles detected

print("Number of circles detected: %d" % len(circles[0]))

print(circles)
circles2 =
#circles = np.uint8(np.around(circles))
#for i in circles[0,:]:
    # draw the outer circle
#    cv.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
#    cv.circle(img,(i[0],i[1]),2,(0,0,255),3)

# ensure at least some circles were found
for cir in circleLog:
    # convert the (x, y) coordinates and radius of the circles to integers
    output = np.copy(orig_image)

    if (len(cir) > 1):
        print("FAIL before")
        exit()

    print(cir[0, :])

    cir = np.round(cir[0, :]).astype("int")

    for (x, y, r) in cir:
        cv2.circle(output, (x, y), r, (0, 0, 255), 2)
        cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)


cv.imshow('detected circles',img)
cv.waitKey(0)
cv.destroyAllWindows()

