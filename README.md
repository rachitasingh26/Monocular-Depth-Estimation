## Monocular-Depth-Estimation on Low Power Systems

This project aims to develop a lightweight, efficient model for monocular depth estimation that runs smoothly on mobile systems. By using techniques like quantization, pruning, and knowledge distillation, along with efficient architectures such as MobileNetV3 and depth-wise separable convolutions, we strive to achieve fast inference without compromising depth accuracy.

Implemented a knowledge distillation framework for efficient monocular depth estimation using a Teacher-Student paradigm. This approach enables deployment of accurate depth estimation models on resource-constrained devices while maintaining high-quality predictions.

-------------------------

Dataset Download:-
We trained our model mainly on the NYU Depth V2 dataset consisting of 50688 Kinect-captured RGB–depth pairs across diverse indoor scenes (basement, bedrooms, offices, kitchens, living room, office, study room), plus 2,000 evaluation images (indoor, normal/low light) with depth and confidence maps.
You can access the dataset through kaggle using the following command:
```
curl -L -o ./datasets/nyu-depth-v2.zip https://www.kaggle.com/api/v1/datasets/download/soumikrakshit/nyu-depth-v2
```

------------------------

To train the model, you need to run the training script using:
```
python3 train.py
```
The parameters for model training can be changed in the main function of the script.

-----------------------------

For running inference on cpu using a model saved in the `checkpoints` directory, you can the use following command:
```
python3 inference.py --model_path /path/to/checkpoint --image_dir /path/to/images/dir --no_cuda
```
---------------------------
The results visualizations and checkpoints are automatically saved during training in the `results/visualizations` and `results` folders respectively.
