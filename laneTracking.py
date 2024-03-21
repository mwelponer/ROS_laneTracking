#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
#from rgb_hsv import BGR_HSV

COLOR_TO_TRACK = [45,149,62]
COLOR_ERROR_PERCENTAGE = 30
WRITE_VIDEO_OUT = False
WRITE_IMAGE_OUT = False
VERBOSE_LOG = False

################################################################################
class BGR_HSV(object):
################################################################################
    def __init__(self):
        pass

    def rgb_hsv(self, rgb):
        assert len(rgb) == 3, "RGB has to have 3 components"
        bgr = [rgb[2], rgb[1], rgb[0]]
        bgr_numpy = np.uint8([[bgr]])
        hsv_numpy = cv2.cvtColor(bgr_numpy, cv2.COLOR_BGR2HSV)
        hsv_numpy_percentage = [hsv_numpy[0][0][0] / 179.0, hsv_numpy[0][0][1]
            / 255.0, hsv_numpy[0][0][2] / 255.0]

        return hsv_numpy[0][0], hsv_numpy_percentage


################################################################################
def white_balance(image):
################################################################################
    #print("[white_balance] Image white balance...")

    result = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)

    #print("[white_balance] Done.")

    return result

################################################################################
def gimpWB(image):
################################################################################
    balanced_img = np.zeros_like(image) #Initialize final image

    for i in range(3): #i stands for the channel index
        hist, bins = np.histogram(image[..., i].ravel(), 256, (0, 256))
        bmin = np.min(np.where(hist>(hist.sum()*0.0005)))
        bmax = np.max(np.where(hist>(hist.sum()*0.0005)))
        #print(bmin, bmax)
        balanced_img[...,i] = np.clip(image[...,i], bmin, bmax)
        balanced_img[...,i] = (balanced_img[...,i]-bmin) / float(bmax - bmin) * 255.0

    return balanced_img

################################################################################
def region_of_interest(image):
################################################################################
    height = image.shape[0]
    width = image.shape[1]
    #print("size: ", width, height)
    triangle = np.array([
        [(int(width/6), height), (int(width*5/6), height),
            (int(width/2), int(height*2/5))] ##### to use with 640x480 video
    ])
    mask = np.zeros((height, width), np.int8)
    cv2.fillPoly(mask, triangle, 255)
    #plt.imshow(mask)
    #plt.show()

    # apply the mask to the image
    image[mask==0] = 0
    #plt.imshow(image)
    #plt.show()

    return image


################################################################################
class LaneTracker(object):
################################################################################
    def __init__(self, rgb_to_track, colour_error_perc=10.0, colour_cal=False,
        camera_topic="/camera/image_raw", cmd_vel_topic="/cmd_vel"):

        self._colour_cal = colour_cal
        self._colour_error_perc = colour_error_perc
        self.rgb_hsv = BGR_HSV()
        self.hsv, hsv_numpy_percentage = self.rgb_hsv.rgb_hsv(rgb=rgb_to_track)

        # define a maximum and a minimum value of the color to track
        min_hsv = self.hsv * (1.0 -(self._colour_error_perc / 100.0))
        max_hsv = self.hsv * (1.0 + (self._colour_error_perc / 100.0))
        self.lower_color = np.array(min_hsv)
        self.upper_color = np.array(max_hsv)

        # We check which OpenCV version is installed.
        (self.major, minor, _) = cv2.__version__.split(".")
        rospy.logwarn("OpenCV Version Installed v."+str(cv2.__version__))

        # This way we process only half the frames
        self.process_this_frame = True

        self.bridge_object = CvBridge()
        ###### subscriber to the camera topic #######
        self.image_sub = rospy.Subscriber(camera_topic, Image, self.camera_callback)
        ###### publisher into the cmd_vel topic #######
        self.cmd_vel_pub = rospy.Publisher(cmd_vel_topic, Twist, queue_size=1)

        if WRITE_VIDEO_OUT:
            self.img_array = []

    def exit(self):
        if WRITE_VIDEO_OUT:
            img_size = (640, 480)
            fourcc = cv2.VideoWriter_fourcc(*'mpeg')
            out = cv2.VideoWriter("out.mp4", fourcc, 20, img_size)
            for i in range(len(self.img_array)):
                out.write(self.img_array[i])
            out.release()

    ##### called everytime we receive an image in the topic ############
    def camera_callback(self, data):

        if self.process_this_frame:
            self.process_this_frame = False
            try:
                # We select bgr8 because its the OpenCV encoding by default
                cv_image = self.bridge_object.imgmsg_to_cv2(data, \
                    desired_encoding="bgr8")
                if WRITE_VIDEO_OUT:
                    self.img_array.append(cv_image) # add frame for the video
                if WRITE_IMAGE_OUT:
                    cv2.imwrite("out.jpg", cv_image)
            except CvBridgeError as e:
                print(e)

            # We get image dimensions and crop the parts of the image we don't 
            # need. Because its image matrix first value is start and second 
            # value is down limit.
            # Select the limits so that it gets the line not too close, not too 
            # far and the minimum portion possible to make process faster.
            # TODO: Get multiple lines so that we can generate paths.

            ######### white balance
            #cv_image = white_balance(cv_image)
            #cv_image = gimpWB(cv_image)
            #cv2.imshow("whiteB", cv_image)
            #cv2.waitKey(1); return

            ######## cut out the triangle of interest
            #region_of_interest(cv_image)

            ####### resize to 20% image, crop upper part to make it faster 
            small_frame = cv2.resize(cv_image, (0, 0), fx=0.2, fy=0.2)
            height, width, channels = small_frame.shape
            #rospy.loginfo("height=%s, width=%s" % (str(height), str(width)))
            #cv2.imshow("whiteB", small_frame)
            #cv2.waitKey(1); return

            ###### crop out upper part
            crop_img = small_frame[int(height*2/5):height, 0:width]
            #height, width, channels = crop_img.shape
            #rospy.loginfo("height=%s, width=%s" % (str(height), str(width)))
            #cv2.imshow("small_frame", small_frame)
            #cv2.waitKey(1); return

            # Convert from RGB to HSV (more stable versus lighting conditions)
            hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
            #cv2.imshow("hsv", hsv)
            #cv2.waitKey(1); return

            # Threshold the HSV image to get only the specific range of colors
            # (the range of the color that we have set)
            mask = cv2.inRange(hsv, self.lower_color, self.upper_color)
            #cv2.imshow("mask", mask)
            #cv2.waitKey(0); return

            # Bitwise-AND mask and original image
            # we combine the range with the original image
            # so we will have an image with the range of colors and everything 
            # else black
            masked_image = cv2.bitwise_and(crop_img, crop_img, mask=mask)
            #cv2.imshow("masked_image", masked_image)
            #cv2.waitKey(0); return

            ########## we then detect the contours ############
            # and from the contours we extract the centroids
            # centroids are blobs of colors, we calculate where is the major 
            # quantity of the searched color basically where the line is
            if self.major == '3':
                # If cv2 version is 3
                (_, contours, _) = cv2.findContours(mask, cv2.RETR_CCOMP, \
                                                    cv2.CHAIN_APPROX_TC89_L1)
            else:
                # If cv2 version is 2 or 4
                (contours, _) = cv2.findContours(mask, cv2.RETR_CCOMP, \
                                                 cv2.CHAIN_APPROX_TC89_L1)
            if VERBOSE_LOG:
                rospy.loginfo("  contours: {}".format(len(contours)))
            centres = []
            for i in range(len(contours)):
                moments = cv2.moments(contours[i])
                try:
                    center = (int(moments['m10'] / moments['m00']), \
                              int(moments['m01'] / moments['m00']))
                    if VERBOSE_LOG: rospy.loginfo("    center {}".format(center))
                    centres.append(center)
                    cv2.circle(masked_image, centres[-1], 10, (0, 255, 0), -1)
                except ZeroDivisionError:
                    if VERBOSE_LOG: rospy.loginfo("    ERR: division by zero!")
                    pass
            #cv2.imshow("center", masked_image)
            #cv2.waitKey(0); return


            ######### find the most centered and closer to us centroid
            index, better_candidate_index = 0
            min_x_dist_from_center = width/2
            max_y_value = 0

            if len(centres) > 0:
                for candidate in centres:
                    cx = candidate[0]
                    cy = candidate[1]
                    x_dist_from_center = abs(width/2 - cx)

                    # if the candidate has the same x as the actual max
                    if x_dist_from_center == min_x_dist_from_center:

                        # if the candidate is closer to us on the y
                        if cy > max_y_value:
                            max_y_value = cy
                            better_candidate_index = index

                    # if the candidate is at the moment the most right
                    elif x_dist_from_center < min_x_dist_from_center:
                        min_x_dist_from_center = x_dist_from_center
                        max_y_value = cy
                        better_candidate_index = index

                    index += 1

                winner = centres[better_candidate_index]
                cv2.circle(masked_image, (int(winner[0]), \
                                          int(winner[1])), 5, (0, 0, 255), -1)
                if VERBOSE_LOG: rospy.loginfo("  best centroid {}".format(winner))

            self.move_robot(height, width, winner[0], winner[1], 0.3, 0.3)
            cv2.waitKey(1)
            #if cv2.waitKey(1) == ord('q'):

            cv2.imshow("IMG", crop_img)
            cv2.imshow("HSV", hsv)
            cv2.imshow("MASK", mask)
            cv2.imshow("RES", masked_image)
        else:
            self.process_this_frame = True

    def move_robot(self, image_dim_y, image_dim_x, cx, cy, linear_vel_base = 0.1, \
                   angular_vel_base = 0.3):
        """
        It move the Robot based on the Centroid Data
        image_dim_x=96, image_dim_y=128
        cx, cy = [(77, 71)]
        """
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0

        FACTOR_LINEAR = 0.001
        FACTOR_ANGULAR = 0.2

        if cx is not None and cy is not None:
            origin = [image_dim_x / 2.0, image_dim_y / 2.0]
            centroid = [cx, cy]
            delta = [centroid[0] - origin[0], centroid[1]]

            #print("origin="+str(origin))
            #print("centroid="+str(centroid))
            #print("delta="+str(delta))

            # -1 because when delta is positive we want to turn right, which 
            # means sending a negative angular
            cmd_vel.angular.z = angular_vel_base * delta[0] * FACTOR_ANGULAR * -1
            # If its further away it has to go faster, closer then slower
            cmd_vel.linear.x = linear_vel_base - delta[1] * FACTOR_LINEAR

        else:
            cmd_vel.angular.z = angular_vel_base * 2
            cmd_vel.linear.x = 0.0 #linear_vel_base * 0.5
            if VERBOSE_LOG: rospy.loginfo("NO CENTROID DETECTED...SEARCHING...")

        #print("SPEED==>["+str(cmd_vel.linear.x)+","+str(cmd_vel.angular.z)+"]")
        self.cmd_vel_pub.publish(cmd_vel)

    def loop(self):
        rospy.spin()

################################################################################
if __name__ == '__main__':
################################################################################
    rospy.init_node('lane_tracker_start', anonymous=True)
    robot_mover = LaneTracker(rgb_to_track=COLOR_TO_TRACK,
        colour_error_perc=COLOR_ERROR_PERCENTAGE, colour_cal=False)
    robot_mover.loop()
    robot_mover.exit()
