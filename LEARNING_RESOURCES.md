# Beginner Learning Resources

This guide is designed for someone learning the project from the beginning. Do not try to understand the full neural network first. Build the required ideas in small steps.

## 1. Suggested study sequence

### Step 1: Python foundations

Learn:

- variables, lists, dictionaries, and loops;
- functions and classes;
- importing modules;
- reading files and working with paths;
- exceptions and virtual environments.

Resources:

- [Python Beginner's Guide](https://wiki.python.org/moin/BeginnersGuide)
- [Official Python Tutorial](https://docs.python.org/3/tutorial/)

Practice in this project:

- Read `hardware_simulator.py`.
- Explain `CrowdSafetyActuator.__init__()`, `trigger_alert()`, and `close()`.
- Change the demo crowd counts and observe the printed status.

### Step 2: NumPy and images

Learn:

- array shape, dtype, indexing, and slicing;
- image channels and pixel values;
- resizing and normalization;
- the difference between `(height, width, channels)` and `(channels, height, width)`.

Resources:

- [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)
- [OpenCV-Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [Pillow Handbook](https://pillow.readthedocs.io/en/stable/handbook/)

Practice in this project:

- Follow image loading in `dataset.py`.
- Print the shape before and after `transforms.ToTensor()`.
- Display an image and its density map with Matplotlib.

### Step 3: Machine-learning basics

Learn:

- features and labels;
- training, validation, and test data;
- loss functions;
- gradient descent and optimizers;
- overfitting;
- MAE, MSE, and RMSE.

Resources:

- [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)
- [scikit-learn Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)

Practice in this project:

- Explain why the image is the input and the density map is the target.
- Calculate a count by summing a small example density array.
- Compare a predicted count with a ground-truth count by hand.

### Step 4: PyTorch basics

Learn:

- tensors and devices;
- `Dataset` and `DataLoader`;
- `nn.Module`;
- forward passes;
- loss, `backward()`, and optimizer steps;
- `model.train()` versus `model.eval()`.

Resources:

- [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
- [PyTorch Dataset and DataLoader Tutorial](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [PyTorch Neural Network Tutorial](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)

Practice in this project:

- Read `CrowdDataset.__getitem__()` in `dataset.py`.
- Run the small test block at the bottom of `model.py`.
- Trace one batch through `CSRNet.forward()`.

### Step 5: Convolutional neural networks

Learn:

- convolution kernels and feature maps;
- pooling and downsampling;
- channels;
- receptive fields;
- transfer learning;
- dilated (atrous) convolution.

Resources:

- [CS231n: Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/)
- [PyTorch Transfer Learning Tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [Torchvision VGG Documentation](https://docs.pytorch.org/vision/stable/models/vgg.html)

Practice in this project:

- Identify the VGG frontend, dilated backend, and one-channel output in `model.py`.
- Explain why the output is a map rather than a single classification label.
- Check why the output resolution is approximately one-eighth of the input.

### Step 6: Crowd counting and CSRNet

Learn:

- point annotations;
- Gaussian kernels;
- density-map integration;
- geometry-adaptive kernels;
- crowd-counting evaluation.

Resources:

- [CSRNet paper (CVPR 2018)](https://openaccess.thecvf.com/content_cvpr_2018/html/Li_CSRNet_Dilated_Convolutional_CVPR_2018_paper.html)
- [CSRNet project page](https://www.semanticscholar.org/paper/CSRNet%3A-Dilated-Convolutional-Neural-Networks-for-Li-Xiong/2b2a1c7d68ab4d6e5c4b8aa8a6c77e3d6c6d6c31)
- [ShanghaiTech dataset overview](https://www.ee.cuhk.edu.hk/~xgwang/ShanghaiTech/)

Practice in this project:

- Read `generate_gaussian_density()` in `make_density_maps.py`.
- Compare raw head points with the generated heatmap in the notebook.
- Verify that summing a density map produces approximately the number of annotations.

### Step 7: Web applications and hardware

Learn:

- Streamlit widgets and reruns;
- model inference in an application;
- latency and FPS;
- serial communication;
- why safety-critical actions need safeguards.

Resources:

- [Streamlit Get Started](https://docs.streamlit.io/get-started)
- [PySerial Documentation](https://pyserial.readthedocs.io/en/latest/)
- [Python `time` module](https://docs.python.org/3/library/time.html)

Practice in this project:

- Run `streamlit run app.py`.
- Test one dataset image before testing batches.
- Run `python hardware_simulator.py --mock` before connecting real hardware.

## 2. A practical first-week plan

### Day 1: understand the files

Read `README.md`, then open `model.py`, `dataset.py`, and `train.py`. Draw the data flow on paper.

### Day 2: inspect one sample

Load one image and one `.h5` file. Print image shape, density-map shape, and density-map sum.

### Day 3: understand the model

Run `python model.py`. Write down the input and output shapes and explain every stage of `forward()`.

### Day 4: understand training

Read one iteration of `train.py`. Identify where the prediction is made, where the loss is computed, and where weights change.

### Day 5: evaluate

Use an existing checkpoint if available. Compare the original image, ground-truth heatmap, and predicted heatmap.

### Day 6: use the dashboard

Launch Streamlit and test the warning and critical sliders with sample images.

### Day 7: review

Explain the complete pipeline without looking at the code:

```text
annotations -> Gaussian density map -> normalized image
-> CSRNet prediction -> sum for count -> heatmap and alert
```

## 3. Beginner questions to answer

1. Why are individual people not detected with bounding boxes?
2. Why must the image and density map be flipped together?
3. Why is the density map downsampled by eight?
4. Why is the density multiplied by `8 ** 2` after resizing?
5. Why does the model use ImageNet normalization?
6. What is the difference between training and evaluation mode?
7. Why is a checkpoint useful?
8. Why can a heatmap be more informative than a single count?
9. Why should threshold values be calibrated for a real venue?
10. Why is mock hardware mode safer for initial testing?

## 4. How to study effectively

- Run small experiments instead of reading every line at once.
- Change one value, such as a threshold or image index, and observe the result.
- Print tensor shapes at each stage.
- Keep a glossary of unfamiliar terms.
- Compare predicted counts with visual heatmaps.
- Treat the model output as an estimate, not automatically as a real-world fact.
- Never connect an untested program directly to safety-critical equipment.

## 5. Recommended project commands

From the project directory and an activated virtual environment:

```powershell
python make_density_maps.py
python model.py
python train.py --dataset part_B --epochs 1 --eval_freq 1
python evaluate.py --checkpoint checkpoints\best_model_part_B.pth --dataset part_B
streamlit run app.py
python hardware_simulator.py --mock
```

Start with one epoch and a small sample while learning. Longer training can be run after you understand the output and have confirmed that the dataset and checkpoint paths are correct.

