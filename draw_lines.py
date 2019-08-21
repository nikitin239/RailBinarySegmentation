import time
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
import loss

config = tf.ConfigProto()
height = 256
width = 256
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


def mask(img):
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    canny = cv2.Canny(blurred, 50, 150)
    cv2.imshow('img', img)
    cv2.imshow('canny2', canny)
    return canny


model = load_model('unet_multiline.h5',
                   custom_objects={'bce_dice_loss': loss.bce_dice_loss, 'dice_coeff': loss.dice_coeff})
def road_lines(image, size):
    """ Takes in a road image, re-sizes for the model,
    predicts the lane to be drawn from the model in G color,
    recreates an RGB image of a lane and merges with the
    original road image.
    """
    # Get image ready for feeding into model
    small_img = cv2.resize(image, (int(width), int(height)))
    small_img = np.array(small_img)
    small_img = small_img[None, :, :, :]
    # Make prediction with neural network (un-normalize value by multiplying by 255)
    prediction = model.predict(small_img / 255)[0]
    pred = np.zeros_like(prediction)
    pred[(prediction > 0.6)] = 1
    pred = pred * 255
    blanks = np.zeros_like(pred).astype(np.uint8)
    lane_drawn = np.dstack((blanks, pred, blanks))
    lane_image1 = mask(lane_drawn)
    lane_image1 = cv2.resize(lane_image1, (size[0], size[1]))
    lane_image1 = np.expand_dims(lane_image1, axis=2)
    # Re-size to match the original image
    lane_image = cv2.resize(lane_drawn, (size[0], size[1]))
    lane_image = lane_image.astype(np.uint8)

    # Merge the lane drawing onto the original image
    result = cv2.addWeighted(image, 1, lane_image, 1, 0)
    return result, lane_image1

cap = cv2.VideoCapture('20171205_144430609.avi')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

while cap.isOpened():
    ret, road_image = cap.read()
    if ret == True:
        time2 = time.time()
        result, line_image = road_lines(road_image, size)
        line_image = cv2.cvtColor(line_image, cv2.COLOR_GRAY2BGR)
        time2 = time.time() - time2
        print(time2 * 1000)
        cv2.imshow('res1', cv2.resize(line_image, (0, 0), fx=0.5, fy=0.5))
        cv2.imshow('res2', cv2.resize(result, (0, 0), fx=0.5, fy=0.5))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
