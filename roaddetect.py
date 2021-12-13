#!/usr/bin/env python3
#!/usr/bin/env python2
import cv2
import numpy as np
import imutils

###############################################################################
#                                                                             #
#         Roadsigns amd Driving Lanes Detection                               #
#                                                                             #
###############################################################################

# open video0
cap = cv2.VideoCapture(0)

if __name__ == '__main__':
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if ret == None:
            print("No frame")
            break

        # Change to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny edge detection
        canny = cv2.Canny(blur, 50, 150)

        # Find contours
        contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)

        #create a copy of the original image to draw on
        drawing = np.copy(frame)

        # Create a mask for lane detection
        mask = np.zeros(frame.shape, dtype=np.uint8)

        # Draw a stencil for lane detection
        cv2.rectangle(mask, (0, 0), (frame.shape[1], frame.shape[0]), (255, 255, 255), -1)

        # apply the stencil to the original image
        masked = cv2.bitwise_and(mask, frame)

        

        # # Find two converging sloped lines
        # //lines = cv2.HoughLinesP(masked, 1, np.pi / 180, 100, np.array([]), minLineLength=100, maxLineGap=10)

        # # Draw lines on the original image
        # if lines is not None:
        #     for line in lines:
        #         x1, y1, x2, y2 = line[0]
        #         cv2.line(drawing, (x1, y1), (x2, y2), (0, 255, 0), 2)


        # Find contours with area > 1000 and check if yhey are a Circle
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 5000:
                approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
                
                if len(approx) == 8:      
                    #cut the image
                    x, y, w, h = cv2.boundingRect(cnt)
                    cropped = frame[y:y+h, x:x+w]              
                    #drwaw contours
                    cv2.drawContours(frame, [cnt], 0, (0, 0, 255), 2)
                    #edge detection
                    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                    blur = cv2.GaussianBlur(cropped, (5, 5), 0)
                    canny = cv2.Canny(blur, 50, 150)
                    #Draw "STOP" text on the frame
                    cv2.putText(frame, "STOP", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                elif len(approx) > 15:
                    # cut the image
                    x, y, w, h = cv2.boundingRect(cnt)
                    cropped = frame[y:y+h, x:x+w]
                    # convert to grayscale
                    cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                    #draw contours
                    canny = cv2.Canny(cropped_gray, 50, 150)
                    contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
                    #open left.jpg to compare with the cropped image
                    left = cv2.imread('left.jpg', 0)
                    d1 = cv2.matchShapes(canny, left, cv2.CONTOURS_MATCH_I1, 0)
                    d2 = cv2.matchShapes(canny, left, cv2.CONTOURS_MATCH_I2, 0)
                    d3 = cv2.matchShapes(canny, left, cv2.CONTOURS_MATCH_I3, 0)
                    if (d1 + d2 + d3) < 0.25:
                        cv2.putText(frame, "LEFT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, "RIGHT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                elif len(approx) < 3:
                    # cut the image
                    x, y, w, h = cv2.boundingRect(cnt)
                    cropped = frame[y:y+h, x:x+w]
                    # convert to grayscale
                    cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                    #draw contours
                    canny = cv2.Canny(cropped_gray, 50, 150)
                    contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
                    #open left.jpg to compare with the cropped image
                    cv2.putText(frame, "LINE", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                #resize image
                canny_min = cv2.resize(canny, (200,200))
                cv2.imshow("edge_detector", canny_min)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        cv2.imshow('mask', canny)

       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    