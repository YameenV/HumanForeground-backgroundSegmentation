# Human Foreground-background Segmentation

## Results :- 
> Ensemble (VGG19, InceptionResnetV2) -> Unet

| DiceCoef   | IOU    | Recall   | Precision   |
|:----------:|:------:|:--------:|:-----------:|
| 0.93      | 0.87  | 0.90    | 0.92       |

![alt text](./FullBody/ensemblef.jpg)



> InceptionResnetV2 -> Unet

| DiceCoef   | IOU    | Recall   | Precision   |
|:----------:|:------:|:--------:|:-----------:|
| 0.89      | 0.82  | 0.87    | 0.89       |

![alt text](./FullBody/fullBodyResnet_1.png)
![alt text](./FullBody/fullBodyResnet_2.png)
![alt text](./FullBody/fullBodyResnet_3.png)


> VGG19 -> Unet

| DiceCoef   | IOU    | Recall   | Precision   |
|:----------:|:------:|:--------:|:-----------:|
| 0.87       | 0.78   | 0.81    | 0.84       |

![alt text](./FullBody/fullBodyVgg161_1.png)
![alt text](./FullBody/fullBodyVgg161_2.png)
![alt text](./FullBody/fullBodyVgg161_3.png)

> Unet

| DiceCoef   | IOU    | Recall   | Precision   |
|:----------:|:------:|:--------:|:-----------:|
| 0.72      | 0.57   | 0.81    | 0.64        |

![alt text](./FullBody/fullBody1.png)
![alt text](./FullBody/fullBody2.png)
![alt text](./FullBody/fullBody3.png)

## Future Improvement
Data Augmentation would have made the model more robust and generalized well



