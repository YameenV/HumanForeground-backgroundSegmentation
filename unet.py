from tensorflow.keras.layers import ( Conv2D, BatchNormalization, 
                                    Activation, MaxPool2D, Conv2DTranspose,
                                    Concatenate, Input )

from tensorflow.keras.models import Model

def convSet(inputs, numFilters):
    x = Conv2D(numFilters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(numFilters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def encoder(inputs, numFilters):
    x = convSet(inputs, numFilters)
    p = MaxPool2D((2,2))(x)
    return x, p

def decoder(input, features, numFilters):
    x = Conv2DTranspose(numFilters,(2,2), strides =2, padding="same")(input)
    x = Concatenate()([x, features])
    x = convSet(x, numFilters)
    return x

def unet(inputShape:tuple):
    inputs = Input(shape = (inputShape))

    s1, p1 = encoder(inputs, 64)
    s2, p2 = encoder(p1, 128)
    s3, p3 = encoder(p2, 256)
    s4, p4 = encoder(p3, 512)
    
    b1 = convSet(p4, 1024)

    d1 = decoder(b1, s4, 512)
    d2 = decoder(d1, s3, 256)
    d3 = decoder(d2, s2, 128)
    d4 = decoder(d3, s1, 64)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)
    model = Model(inputs, outputs, name="UNet")
    return model

if __name__ == "__main__":
    unet()