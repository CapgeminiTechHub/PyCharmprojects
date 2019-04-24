import numpy as np
import argparse
import cv2
import signal

from functools import wraps
import errno
import os
import copy
from sklearn.cluster import KMeans


# Read in colour image
connect4 = cv2.imread("images/connect4 image 1.jpg")
#img = cv2.imread("images/single circle.jpg")
#print(connect4.shape)
#print(connect4.size)
#print(connect4.dtype)


# Change size and resolution of image.
connect4 = cv2.resize(connect4, (360, 640))
#print(connect4.shape)
#print(connect4.size)
#print(connect4.dtype)

# Blur the image
connect4_blurred = cv2.GaussianBlur(connect4,(5,5),2,2)
# Write blurred image
cv2.imwrite("images/Connect4_blurred.jpg", connect4_blurred)

orig_image = np.copy(connect4)
output = connect4.copy()
gray = cv2.cvtColor(connect4_blurred, cv2.COLOR_BGR2GRAY)

circles = None

minimum_circle_size = 13      #this is the range of possible circle in pixels you want to find
maximum_circle_size = 18  #maximum possible circle size you're willing to find in pixels

guess_dp = 1.0

number_of_circles_expected = 42          #we expect to find just one circle
breakout = False

max_guess_accumulator_array_threshold = 100     #minimum of 1, no maximum, (max 300?) the quantity of votes
                                                #needed to qualify for a circle to be found.
circleLog = []

guess_accumulator_array_threshold = max_guess_accumulator_array_threshold

while guess_accumulator_array_threshold > 1 and breakout == False:
    #start out with smallest resolution possible, to find the most precise circle, then creep bigger if none found
    guess_dp = 1.0
    # print("resetting guess_dp:" + str(guess_dp))
    while guess_dp < 9 and breakout == False:
        guess_radius = maximum_circle_size
        # print("setting guess_radius: " + str(guess_radius))
        # print(circles is None)
        while True:

            #HoughCircles algorithm isn't strong enough to stand on its own if you don't
            #know EXACTLY what radius the circle in the image is, (accurate to within 3 pixels)
            #If you don't know radius, you need lots of guess and check and lots of post-processing
            #verification.  Luckily HoughCircles is pretty quick so we can brute force.

            # print("guessing radius: " + str(guess_radius) +
            #        " and dp: " + str(guess_dp) + " vote threshold: " +
            #        str(guess_accumulator_array_threshold))

            circles = cv2.HoughCircles(gray,
                cv2.HOUGH_GRADIENT,
                dp=guess_dp,               #resolution of accumulator array.
                minDist=20,                #chnaged from 100 to 20 number of pixels center of circles should be from each other, hardcode
                param1=50,
                param2=guess_accumulator_array_threshold,
                minRadius=(guess_radius-3),    #HoughCircles will look for circles at minimum this size
                maxRadius=(guess_radius+3)     #HoughCircles will look for circles at maximum this size
                )

            if circles is not None:
                if len(circles[0]) == number_of_circles_expected:
                    #print("**********************************************************************************len of circles: " + str(len(circles[0])))
                    #print("Circles: ", circles)
                    # The value returned by the Hough Circles function is a 1,42,3 numpy array .i.e. an outer array containing 42 points each point consisting of 3 values ( x,y and radious )
                    # To order the circles in x corrdinate order we need to create a ndarray with the right structure . Sort the output from the Hough's Circles function
                    # and assign it to the new variable at the right point in the structure
                    circles2 = np.ndarray(shape=(1,42,3))
                    circles2[0] = sorted(circles[0], key=lambda x: (x[0], x[1]))
                    #print("Circles sorted: ", circles2)
                    circleLog.append(copy.copy(circles2))
                    #print("CircleLog: ", circleLog)
                    #print("dp = %f" % guess_dp)
                    #print("param2 : %f" % guess_accumulator_array_threshold)
                    #print("min radius :  %f" % (guess_radius-3))
                    #print("max radius : %f" % (guess_radius+3))
                    #print("k1")
                    breakout = True
                break

                circles = None
            guess_radius -= 1
            if guess_radius < 5:
                break;

        guess_dp += 1.5

    guess_accumulator_array_threshold -= 2

#Return the circleLog with the highest accumulator threshold

#lets try and reshape the circleLog[0] into a 6 x 7 x 3 numpy array
circleLog[0] = np.reshape(circleLog[0],(7,6,3))
print("circleLog: ", circleLog)
print("length of circleLog[0]: ", len(circleLog[0]))
print("length of cicleLog: ", len(circleLog))

# ensure at least some circles were found
for cir in circleLog:
    # convert the (x, y) coordinates and radius of the circles to integers
    output = np.copy(orig_image)

    if (len(cir) != 7):
        print(len(cir))
        print("Should be 7 columns")
        exit()

    print(cir[0, :])

    cir = np.round(cir[0, :]).astype("int")

    circle_count=1
    for (x, y, r) in cir:
        color = connect4[y,x]
        print(color)
        cv2.circle(output, (x, y), r, (0, 0, 255), 2)
        circle_count_string="%s" % circle_count
        cv2.putText(output, circle_count_string, (x-5,y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (0,0,0), 2)
        #cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        circle_count=circle_count+1



    cv2.imwrite("images/output.jpg",output)
    cv2.imshow("output", np.hstack([orig_image, output]))
    cv2.waitKey(0)