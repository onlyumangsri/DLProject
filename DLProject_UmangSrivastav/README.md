# To study the effect of Transfer Learning

Two ResNET models and one modified model are used for this study

## Installation

1.	Python: Make sure you have Python installed on your system. PyTorch requires Python 3.6 or higher.

2.	PyTorch: Install PyTorch library which provides various modules and functions for deep learning. You can install it using pip or conda package manager.

3.	Jupyter Notebook: Install Jupyter Notebook which is an open-source web application that allows you to create and share documents that contain live code, equations, visualizations, and narrative text. Google Colab or Kaggle are other options to use.

4.	Dataset: Datasets are installed from PyTorch library torchvision.datasets. SO external files are not required to be loaded for data for the same.

5.	Hardware requirements: As the size of the datasets are large, you may need a system with a decent CPU and GPU to train your models efficiently. You can use cloud-based services such as Google Colab, AWS, or Azure if you don't have access to a high-performance machine.
Once you have installed all the necessary libraries and set up your environment, you can execute your Jupyter Notebook files on PyTorch for ResNet-18 and ResNet-50 transfer learning.


## Usage

Just simply run the files as such. There is no external dataset dependency that needs to be added manually.

## Observations 

RESNET-50 on CIFAR-10 has an accuracy (test) as high as 80%.
Now, a modified model, the first 18 layers are extracted from the above trained model as:


```python
    def __init__(self, model):
        super().__init__()        
        resnet=model#.resnet
        self.conv1=resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        del self.layer2[3]        
        self.avgpool = resnet.avgpool

        self.fc=torch.nn.Linear(in_features=512, out_features=100, bias=True)

       
```
Now, another model gets made using those 18 layers and a fully connected layer getting attached to it at last:


```python
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
```
Then all the layers’ parameters, except the last FC layer, are kept fixed (not trainable):

```python
for name, param in model.named_parameters():
    if name.startswith('fc'):
        param.requires_grad = True
    else:
        param.requires_grad = False
```
And then the whole model is trained again on CIFAR-100. This is Transfer learning from one model to another.
The accuracy of this model on CIFAR-100 turns out to be around 40%.

But when the same modified model is fine tuned on CIFAR-10, the performance turned out to be almost equal as the original RESNET-50 trained on CIFAR-10.
The first 18 layers of ResNet50 learned useful features for CIFAR10, which were then used to improve the performance of a new model on that same dataset.

The RESNET-18 trained on CIFAR-100 gives the test accuracy as high as 48%.


Three different ways of few shots learning are done here:
1.	Using just 10% of the dataset to train the model of Resnet50 on CIFAR10, the test accuracy turned out to be 57%.

2.	Another method used here is data augmentation with transfer learning (pretrained Resnet50 on imagenet), but there was no significant impact on test accuracy as it turned out to be just 10-12%.

3.	The third method is using just few samples (50) of each class and then train the model (Resnet50 on CIFAR10) with those sample and then put it to test. The accuracy turned out to be just 12%. 
More work needs to be done in this area as this was just some trial. The first method turned out to be favourable to the learning. But, the other two methods also can be fine tuned more.



## License

[MIT](https://choosealicense.com/licenses/mit/)