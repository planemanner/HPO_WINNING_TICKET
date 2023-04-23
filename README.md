# Introduction
  - This Repository is implementation of [Hyperparameter Optimization with Neural Network Pruning](https://scholar.google.com/citations?view_op=view_citation&hl=ko&user=PywlW4gAAAAJ&citation_for_view=PywlW4gAAAAJ:IjCSPb-OGe4C)
  - This paper claims that any pruned neural network can produce a hyperparameter set which show similar performance with original neural network.
  - **You can find a compromising result which shows we do not use original neural network itself to obtain optimal hyperparameters**
  - By using a pruned neural network, the HPO time decreases remarkably.
  - README update date : 2023 . 04. 16
# Done
  - Minor bugs are fixed.

# Note
  - Actually, I recommend you to just reduce the number of channels for each layer of a neural network.
    - Unfortunately, you would require hard-coded version of pruning when if you use one-shot structured pruning depending on the structure of neural network.
    - If you know some more pretty method, please leave that method in git-issue. 
  - It is much more simple and produces a slim version of a neural network that can be compatible with structured operations provided by Nvidia.

# Requirements
  - pytorch  >= 1.6
  - torchvision >= 0.7.0
  - optuna >= 3.0.0
  - albumentations >= 1.2.0
  - pip install botorch >= 0.6.0

# Usage
  - Prepare datasets : CIFAR100 or TinyImageNet... 
  - COMMAND EXAMPLE
    ```
    python main.py --model_dir [where_to_your_pruned_model_ckpt]
    ```
# Reference
  - You can prepare TinyImageNet dataset in this [link](https://www.kaggle.com/c/tiny-imagenet)

# Contact
  - Anybody wants help for this repository, send an e-mail on smddls77@gmail.com or leave an issue  ]()