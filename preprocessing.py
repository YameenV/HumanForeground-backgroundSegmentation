import tensorflow as tf


def processImage(path):
    imgData = tf.io.read_file(path)
    img = tf.image.decode_png(imgData, channels=3)
    img = tf.image.resize(img, (512,512))
    img = tf.image.convert_image_dtype(img, tf.float32) / 255.
    return img

def processMask(path):
    imgData = tf.io.read_file(path)
    img = tf.image.decode_png(imgData, channels=1)
    img = tf.image.resize(img, (512,512))
    img = tf.image.convert_image_dtype(img, tf.float32) / 255.
    return img

if __name__ == "__main__":
    processImage()
    processMask()