# Semantic_Segmentation_for_self_driving_cars

### Goals
The objective of this project is to implement a semantic segmentation model on self driving car image dataset. This project uses the UNet architecture model with dropout and BatchNorm layers for the training.

![UNet](https://github.com/rsmeghana8/Semantic_Segmentation_for_self_driving_cars/assets/57563443/6aa69b8e-4212-4388-b8e4-98a25f21074c)


## Documentation
### Dataset description
Get the dataset here: [Self_driving_car_Dataset](https://www.kaggle.com/datasets/kumaresanmanickavelu/lyft-udacity-challenge/data)

The dataset contains 5 sets of 1000 RGB images (Total 5000) and labels for semantic segmentation captured via CARLA self-driving car simulator. The dataset contains 13 different classes.

### Installing requirements
To install all the dependencies,after cloning the repository run the following command:
```
    pip install -r requirements.txt
```

### Prepare the dataset and Train the model

Change the folder paths in 'config.yaml' and hyper-parameters  in the 'params.yaml' file to change the model params. Prepare data and train the model by running 

```
    python main.py

```
### Results

when trained for 50 epochs the model got the following results

|      Epochs   |  Train Accuracy |  Val Accuracy |
| :------------ |:---------------:| -------------:|
|       50      |      98.42       |    98.38     |

Let's look at the Training and Val accuracy plots




Model's training and validation accuarcy curves looks very close, so the model seems to be performing very well.

Let's look at some predictions from the model

![output1](https://github.com/rsmeghana8/Semantic_Segmentation_for_self_driving_cars/assets/57563443/9ae7e9bf-040b-4ed1-83cf-ea7b74c01383)

![output2](https://github.com/rsmeghana8/Semantic_Segmentation_for_self_driving_cars/assets/57563443/d42dc87b-14d3-4d4f-98cf-eedbc7d703d0)

![output3](https://github.com/rsmeghana8/Semantic_Segmentation_for_self_driving_cars/assets/57563443/f6f45a35-8452-4a86-bf5e-2f30b6b26eec)

![output4](https://github.com/rsmeghana8/Semantic_Segmentation_for_self_driving_cars/assets/57563443/431bd349-8d61-408b-a89e-4a48b9a78641)





