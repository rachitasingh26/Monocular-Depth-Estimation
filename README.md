## Monocular-Depth-Estimation on Low Power Systems

- This codebase is focused on the training of Knowledge Distillation models on NYU-V2 and SYNTHIA datasets. The `train_baseline.py` and `models_baseline.py` contain the code for training and architecture of our baseline U-Net architecture respectively.
- The `nyu_eda.ipynb` and `synthia_eda.ipynb` contains exploratory data analysis for both the datasets mentioned earlier. 
- The `train.py` and `models/models.py` contains code for our proposed work based on a Teacher-Student architecture.
- Our baseline model overfits pretty early since a pretrained ResNet encoder is being used.
- The graph below shows that the validation loss is pretty much constant and the train loss also plateaus after a few epochs.
<img src="![image](https://github.com/user-attachments/assets/4a4fd461-909f-4ad9-b996-ff7b32b1e256)" width=50% height=50%>

