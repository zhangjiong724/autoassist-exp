# AutoAssist Implementation
Code accompanying the NeurIPS 2019 paper [AutoAssist: A Framework to Accelerate Training of Deep Neural Networks](https://arxiv.org/pdf/1905.03381.pdf).

# Description
The current codebase contains applications on image classification and neural machine translation. 
Batch generation with AutoAssist is realized by implementing an Assistant-sampler (torch.utils.data.sampler).
The Assistant training/prediction is done through custom Assistant-model (torch.nn.model), which needs to be
modified if one wish to apply the AutoAssist architecture on other applications/data formats.
For more implementation detail, refer to "./image_classification/assistant.py".

# More Info
This repository contains the source code 
for the experiments in our NeurIPS 2019 paper 
[AutoAssist: A Framework to Accelerate Training of Deep Neural Networks].
If you find this repository helpful in your publications, please consider citing our paper.
```
@inproceedings{zhang2019autoassist,
  title={AutoAssist: A Framework to Accelerate Training of Deep Neural Networks},
  author={Zhang, Jiong and Yu, Hsiang-fu and Dhillon, Inderjit S},
  booktitle={Conference on Neural Information Processing Systems},
  year={2019}
}
```

For any questions and comments, please send your email to
[zhangjiong724@utexas.edu](mailto:zhangjiong724@utexas.edu)



