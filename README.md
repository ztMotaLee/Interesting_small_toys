# Interesting small toys: Self-Evolutional AI
This repository contains a toy example of a self-evolving AI system for image classification. The goal is to explore how a model can autonomously evolve its architecture and training parameters by randomly selecting and mutating layers. 

## Components

Self-Evolutional AI (Toy Example) leverages an evolutionary strategy where:  

1. Dynamic Model Architecture: The system starts with a trivial model and iteratively mutates its structure by randomly adding, removing, or modifying layers.  

2. Evolutionary Training Loop: The best performing models (based on validation accuracy) are selected to generate new candidates.

3. Weight Inheritance: New candidate models inherit weights from their parents for unchanged layers.

4. Autonomous Hyperparameter Tuning: Training parameters such as learning rate and optimizer type are also evolved. 


## Features

1. Randomized Layer Selection: The model randomly chooses mutation operations (e.g., add a convolutional or pooling layer) to change its architecture.
   
2. Continuous Adaptation: Through a repetitive evolutionary loop, the model continuously refines its structure and training strategy.
      

## Run:


```
python self_evolution.py
```


and 
```
python self_evolution.py
```
