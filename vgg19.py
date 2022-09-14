from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19

def convBlock(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def decoderBlock(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = convBlock(x, num_filters)
    return x

def vgg19Unet(input_shape:tuple):
   
    inputs = Input(input_shape)

    vgg19 = VGG19(include_top=False, weights="imagenet", input_tensor=inputs)

    s1 = vgg19.get_layer("block1_conv2").output         
    s2 = vgg19.get_layer("block2_conv2").output         
    s3 = vgg19.get_layer("block3_conv4").output         
    s4 = vgg19.get_layer("block4_conv4").output         

    b1 = vgg19.get_layer("block5_conv4").output         

    d1 = decoderBlock(b1, s4, 512)                     
    d2 = decoderBlock(d1, s3, 256)                     
    d3 = decoderBlock(d2, s2, 128)                     
    d4 = decoderBlock(d3, s1, 64)                      

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="VGG19U-Net")
    return model

if __name__ == "__main__":
    vgg19Unet()