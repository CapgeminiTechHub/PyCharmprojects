import cv2
import numpy as np

def ColorNameFromHSV(image,y,x):
    # Create a table of HSV colour values and associated name as a lookup table
    ColorTable = [[14,23, "Orange"],[25,35, "Yellow"],[55,65,"Green"], [115,125,"Blue"], [175,5,"Red"]    ]

    # get the HSV color
    colorHSV = image[y,x]
    hue=colorHSV[0]
    saturation=colorHSV[1]

    # Check that we have a strong coliur i.e. satuartion is great than 25
    if saturation < 25:
        return"Unknown"

    for color in ColorTable:
        print("Hue: %d ColorTableLower %d ColourTableUpper %d " % (hue, color[0], color[1]))
        if hue >= color[0] and hue <= color[1] :
            return color[2]
            break
     # Red is a special condition >175 but < than 5
    if hue >= 175 or hue <= 5:
        return("Red")

# Read in colour image in BGR colour space
connect4BGR = cv2.imread("images/connect4 image 5.jpg")
#print(connect4BGR.shape)
#print(connect4BGR.size)
#print(connect4BGR.dtype)


# Change size and resolution of image. Don't need a high resolution image for shape detection
connect4BGR = cv2.resize(connect4BGR, (360, 640))
#print(connect4BGR.shape)
#print(connect4BGR.size)
#print(connect4BGR.dtype)

# Create a HSV version of the image. HSV is better for colour comparison
connect4HSV = cv2.cvtColor(connect4BGR,cv2.COLOR_BGR2HSV)

# Blur the image. This has the effect of hiding some small features such as the circle shapes on the counters which we don't want to detect
connect4_blurred = cv2.GaussianBlur(connect4BGR,(5,5),2,2)
# Write blurred image
cv2.imwrite("images/Connect4_blurred.jpg", connect4_blurred)

# Take a copy of the original image to compare with the processed image
orig_image = np.copy(connect4BGR)
output = connect4BGR.copy()

# Hough's Circles works better with a gray scale image so convert to grayscale
gray = cv2.cvtColor(connect4_blurred, cv2.COLOR_BGR2GRAY)

# Initialise list of circles
circles = None

# Don't like the way you have to hard code these values.
# Makes the program very dependant on the size of the image taken

minimum_circle_size = 13   #this is the range of possible circle in pixels you want to find
maximum_circle_size = 18    #maximum possible circle size you're willing to find in pixels

guess_dp = 1.0

number_of_circles_expected = 42          #we expect to find 42 circles on the Connect4 board
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
                    #print("Circles: ", circles)
                    # The value returned by the Hough Circles function is a 1,42,3 numpy array .i.e. an outer array containing 42 points each point consisting of 3 values ( x,y and radious )
                    # To order the circles in x corrdinate order we need to create a ndarray with the right structure . Sort the output from the Hough's Circles function
                    # and assign it to the new variable at the right point in the structure
                    circles2 = np.ndarray(shape=(1,42,3))
                    circles2[0] = sorted(circles[0], key=lambda x: (x[0], x[1]))
                    print("Circles sorted: ", circles2)
                    circleLog.append(np.copy(circles2))
                    #print("CircleLog: ", circleLog)
                    #print("dp = %f" % guess_dp)
                    #print("param2 : %f" % guess_accumulator_array_threshold)
                    #print("min radius :  %f" % (guess_radius-3))
                    #print("max radius : %f" % (guess_radius+3))
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
#print("circleLog: ", circleLog)
#print("length of circleLog[0]: ", len(circleLog[0]))
#print("length of cicleLog: ", len(circleLog))

#Having reshaped the array into a 7 x 6 array to match the board we have to sort each circle in the each column by the y coordinate to get
# them in the right order

#print(connect4BGR.shape)

# ensure at least some circles were found
for board in circleLog:
    # convert the (x, y) coordinates and radius of the circles to integers
    #output = np.copy(orig_image)

    if (len(board) != 7):
        print(len(board))
        print("Should be 7 columns on the board")
        exit()

    circle_count = 1
    for column in range(7):
        # Round the positions of the circles to integer values to map to x and y coordinates
        board = np.round(board[:]).astype("int")
        print("Column number:", column)
        #print("Column contents: ", board[column])

        # Order the 6 circles in this column by their summing their x and y coordinate. The x-coordinate with only
        # vary slightly because they are all in the same column.
        column2 = np.ndarray(shape=(6, 3))
        column2 = sorted(board[column], key=lambda x: (x[0]+x[1]))
        #print("Column 2 : ", column2)
        board[column] = np.copy(column2)
        #print("column coordinates : ", board[column])

        for (x, y, r) in board[column]:
            # Get BGR and HSV color value using row and column pixel value
            colorBGR = connect4BGR[y,x]
            colorHSV = connect4HSV[y,x]
            print("Circle %d ColourBGR : %s ColorHSV : %s" % (circle_count, colorBGR, colorHSV))
            colorName=ColorNameFromHSV(connect4HSV,y,x)
            print("Colour Name : %s" % colorName)
            # Draw a circle around each detected circle based on it's detected centre coordinates and radius
            # Then label each circle with a number
            cv2.circle(output, (x, y), r, (0, 0, 255), 2)
            circle_count_string="%s" % circle_count
            cv2.putText(output, circle_count_string, (x-10,y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (0,0,0), 2)
            circle_count=circle_count+1


    cv2.imwrite("images/output.jpg",output)
    cv2.imshow("output", np.hstack([orig_image, output]))


# Having used Houghs Circles to detect positions on the board lets try a different approach
# Let's try and detect areas of the image that are yellow or red like the counters
# To remove some of the variability with lighting conditions suggest we use HSV ( Hue, Saturation and Brightest instead of BGR values
# Note. OpenCV uses BGR ( Blue, Green, Red ) when most people use RGB





    cv2.waitKey(0)