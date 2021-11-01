# Sensory Exploration

## Usage

Install the requirements with

```
pip install -r requirements.txt
```

Run the training with

```
python main.py
```

## Reward

```
info['new_explored'] * (1 / info['step'])
```

The larger the step number, the smaller the reward the agent receives for newly explored cells, motivating it to explore more at earlier steps.

## Parameters
Parameter | Value | 
---|---|
Iterations | 500
Entropy | 0.1
Lambda | 1 
Seed | 10
Width & Height | 20
Observation | 11
Vision radius | 5
Max steps | 2000
Activation function | relu
Convolution layers | [[16, [3, 3], 2], [32, [3, 3], 2], [32, [3, 3], 1]]
Cell types | UNK, FREE, OCCUPIED

## Results

<img src = gifs/iter_499.gif width="400" height="400">
