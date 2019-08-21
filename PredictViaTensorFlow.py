import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(threshold=np.inf)
import tensorflow as tf

height = 256
width = 256
from PIL import Image


def lsd(img):
    """ Нахождение линий на изображении с помощью LineSegmentDetector """

    lsd = cv2.createLineSegmentDetector(0)

    lines = lsd.detect(img)[0]  # Position 0 of the returned tuple are the detected lines
    # Draw detected lines in the image
    drawn_img = lsd.drawSegments(img, lines)
    return lines, drawn_img


def mask(img):
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    canny = cv2.Canny(blurred, 50, 150)
    im = cv2.blur(canny, (4, 4))

    lines2, lsd_img_canny = lsd(canny)

    # apply a threshold

    cv2.imshow('img', img)
    cv2.imshow('canny', canny)
    cv2.imshow('cleaned', im)
    cv2.imshow("LSD_canny", lsd_img_canny)

    return canny


detection_graph = tf.Graph()
with detection_graph.as_default():
    graph_def_optimized = tf.GraphDef()
    with tf.gfile.GFile('final.pb', 'rb') as f:
        serialized_graph = f.read()

        graph_def_optimized.ParseFromString(serialized_graph)
        tf.import_graph_def(graph_def_optimized, name='')
for op in detection_graph.get_operations():
    print(str(op.name))
# G = tf.Graph()
tf.keras.backend.set_learning_phase(1)


# Class to average lanes with
class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []


def denseCRF(image, final_probabilities):
    h = final_probabilities.shape[0]
    w = final_probabilities.shape[1]
    softmax = final_probabilities.squeeze()

    # softmax = final_probabilities.transpose((2, 0, 1))

    # The input should be the negative of the logarithm of probability values
    # Look up the definition of the softmax_to_unary for more information
    output_probs = np.expand_dims(softmax, 0)
    output_probs = np.append(1 - output_probs, output_probs, axis=0)
    d = dcrf.DenseCRF2D(w, h, 2)

    U = -np.log(output_probs)
    U = U.reshape((2, -1))
    U = np.ascontiguousarray(U)
    img = np.ascontiguousarray(image)

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=20, compat=3)
    d.addPairwiseBilateral(sxy=30, srgb=20, rgbim=img, compat=10)

    Q = d.inference(5)
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w))
    return Q


def road_lines(size):
    """ Takes in a road image, re-sizes for the model,
    predicts the lane to be drawn from the model in G color,
    recreates an RGB image of a lane and merges with the
    original road image.
    """

    # Get image ready for feeding into model

    with detection_graph.as_default():
        y, = tf.import_graph_def(graph_def_optimized, return_elements=['output_node0:0'])
        with tf.Session(graph=detection_graph) as sess:
            op = sess.graph.get_operations()
            [print(m.values()) for m in op][1]
            while cap.isOpened():
                ret, road_image = cap.read()
                small_img_orig = imresize(road_image, (height, width, 3))
                # plt.imshow(small_img)
                # plt.show()
                small_img = np.expand_dims(small_img_orig, axis=0)
                # small_img =small_img.astype(np.float32)

                x = detection_graph.get_tensor_by_name('import/input_1:0')
                # y = detection_graph.get_tensor_by_name('import/output/truediv:0')
                # tf.global_variables_initializer().run()
                time2 = time.time()
                prediction = sess.run(y, feed_dict={x: small_img / 255})
                crf = denseCRF(small_img_orig, prediction[0])
                plt.imshow(Image.fromarray((crf * 255).astype(np.uint8)))
                plt.show()
                # crf2 = crf(small_img_orig, prediction[0])
                time2 = time.time() - time2
                print(time2 * 1000)
                prediction = np.int_(prediction[0] * 255)

                lanes.recent_fit.append(prediction)
                if len(lanes.recent_fit) > 2:
                    lanes.recent_fit = lanes.recent_fit[1:]
                # Calculate average detection
                # Calculate average detection
                lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis=0)

                # Generate fake R & B color dimensions, stack with G
                blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
                lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))
                canny = mask(lane_drawn)
                nonzero = canny.nonzero()
                nonzeroy = np.array(nonzero[0])
                nonzerox = np.array(nonzero[1])

                ploty = np.unique(nonzeroy)

                ploty2 = []

                plotx1 = []
                plotx2 = []

                for pl_y in ploty:
                    accum = []
                    for x_line, y_line in zip(nonzerox, nonzeroy):
                        if (pl_y == y_line):
                            # if ((x_line<170) and (x_line>100)):
                            accum.append(x_line)
                    if len(accum) > 0:
                        ploty2.append(pl_y)

                        plotx1.append(np.min(accum))
                        plotx2.append(np.max(accum))

                ploty2 = np.array(ploty2)
                plotx1 = np.array(plotx1)
                plotx2 = np.array(plotx2)

                # Re-size to match the original image
                left_fit = np.polyfit(ploty2, plotx1, 2)
                right_fit = np.polyfit(ploty2, plotx2, 2)
                left_fitx = left_fit[0] * ploty2 ** 2 + left_fit[1] * ploty2 + left_fit[2]
                right_fitx = right_fit[0] * ploty2 ** 2 + right_fit[1] * ploty2 + right_fit[2]
                pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty2]))])
                pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty2]))])

                cv2.polylines(lane_drawn, np.int_([pts_left]), color=(255, 0, 0), isClosed=False)
                cv2.polylines(lane_drawn, np.int_([pts_right]), color=(0, 0, 255), isClosed=False)
                color_warp = imresize(lane_drawn, (size[1], size[0], 3))

                # Merge the lane drawing onto the original image

                result = cv2.addWeighted(road_image, 1, color_warp, 1, 0)

                cv2.imshow('res', cv2.resize(result, (0, 0), fx=0.3, fy=0.3))
                out.write(result)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

    return 0


# Where to save the output video


# Location of the input video
lanes = Lanes()
cap = cv2.VideoCapture('/home/dnikitin/Videos/Kazam_screencast_00004.mp4')
# cap = cv2.VideoCapture('/videos/Video/20171205_144853496.avi')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter('output_tf.avi', fourcc, 20.0, size)

road_lines(size)

cap.release()
out.release()
cv2.destroyAllWindows()
