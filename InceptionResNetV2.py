from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, ZeroPadding2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import InceptionResNetV2

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

def inceptionResnetv2Unet(input_shape:tuple):
   
    inputs = Input(input_shape)
    
    encoder = InceptionResNetV2(include_top=False, weights="imagenet", input_tensor=inputs)

    s1 = encoder.get_layer("input_1").output           

    s2 = encoder.get_layer("activation").output        
    s2 = ZeroPadding2D(( (1, 0), (1, 0) ))(s2)         

    s3 = encoder.get_layer("activation_3").output      
    s3 = ZeroPadding2D((1, 1))(s3)                     

    s4 = encoder.get_layer("activation_74").output      
    s4 = ZeroPadding2D(( (2, 1),(2, 1) ))(s4)           

    b1 = encoder.get_layer("activation_161").output     
    b1 = ZeroPadding2D((1, 1))(b1)                      

    d1 = decoderBlock(b1, s4, 512)                     
    d2 = decoderBlock(d1, s3, 256)                     
    d3 = decoderBlock(d2, s2, 128)                     
    d4 = decoderBlock(d3, s1, 64)                      

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="InceptionResNetV2U-Net")
    return model

if __name__ == "__main__":
    inceptionResnetv2Unet()