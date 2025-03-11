# Interesting small toys: Self-Evolutional AI
This repository contains a toy example of a self-evolving AI system for image classification. The goal is to explore how a model can autonomously evolve its architecture and training parameters by randomly selecting and mutating layers. While experimental and not intended for production, this project demonstrates fundamental ideas behind neural architecture search and evolutionary algorithms in a clear and accessible way.


## Components

Self-Evolutional AI (Toy Example) leverages an evolutionary strategy where:  

1. Dynamic Model Architecture: The system starts with a trivial model and iteratively mutates its structure by randomly adding, removing, or modifying layers.  

2. Evolutionary Training Loop: The best performing models (based on validation accuracy) are selected to generate new candidates, ensuring continuous self-improvement.  

3. Weight Inheritance: New candidate models inherit weights from their parents for unchanged layers, allowing them to build on previously learned features.  

4. Autonomous Hyperparameter Tuning: Training parameters such as learning rate and optimizer type are also evolved, making the process entirely self-directed.  


## Features

1. Randomized Layer Selection: The model randomly chooses mutation operations (e.g., add a convolutional or pooling layer) to change its architecture.
   
2. Continuous Adaptation: Through a repetitive evolutionary loop, the model continuously refines its structure and training strategy.
   
3. Educational Toy: A simple, easy-to-follow example built with PyTorch, showcasing basic ideas behind self-evolving neural networks.
   
4. CIFAR-10 Classification: The example is designed to classify CIFAR-10 images, providing a concrete application of the evolutionary approach.

## Run:


```
python self_evolution.py
```


and 
```
python self_evolution.py
```
