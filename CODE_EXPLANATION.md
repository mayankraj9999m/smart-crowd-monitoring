# Smart Crowd Density Monitoring: Beginner Code Explanation

This document explains the project in the order in which data moves through it:

```text
head annotations -> density maps -> PyTorch dataset -> CSRNet -> training
-> evaluation -> Streamlit dashboard -> optional hardware alert
```

The notebook, `Smart_Crowd_Density_Monitoring_CSRNet.ipynb`, demonstrates the same ideas interactively. The `.py` files make the workflow easier to rerun from a terminal.

## 1. What problem does the project solve?

The program estimates how many people are in a crowd image and shows where the crowd is concentrated.

Instead of drawing one bounding box around every person, the model predicts a **density map**:

- A bright area means many people are close together.
- A darker area means fewer people are present.
- Adding all values in the density map gives the estimated crowd count.

This is useful in crowded scenes because people can overlap so heavily that ordinary object detection becomes difficult.

## 2. Important concepts

### Image

An image is stored as a grid of pixels. In this project an RGB image has three channels:

```text
height x width x 3
```

PyTorch usually represents it as:

```text
3 x height x width
```

### Ground-truth point annotations

The ShanghaiTech dataset supplies a point near each person's head. These points are stored in MATLAB `.mat` files.

### Density map

Each head point is replaced by a small Gaussian hill. The sum of all hills is the ground-truth density map. The integral (in practice, the sum of pixels) is approximately the number of people.

### Tensor

A tensor is a multidimensional array used by PyTorch. Images, density maps, model parameters, and batches are all represented as tensors.

### Epoch and checkpoint

- An **epoch** is one complete pass over the training images.
- A **checkpoint** is a saved copy of the model weights, usually saved when validation performance improves.

## 3. `make_density_maps.py`: creating training targets

`generate_gaussian_density()` performs these steps for one image:

1. Read the image size.
2. Read head coordinates from the matching `.mat` file.
3. Create an empty array with one density value per image pixel.
4. Put a normalized Gaussian around every annotated head.
5. Add all Gaussian kernels together.
6. Return the resulting `float32` density map.

The project uses two strategies:

- **Part A:** the Gaussian width depends on the distance to the three nearest neighbors. A crowded area gets a narrower kernel, while a more spread-out area gets a wider one.
- **Part B:** a fixed `sigma = 15` Gaussian is used.

The map is saved beside its image as an HDF5 file:

```text
images/IMG_1.jpg
images/IMG_1.h5
```

The `.h5` file contains a dataset named `density`. Keeping these maps on disk means they do not have to be regenerated during every training epoch.

## 4. `dataset.py`: loading images for PyTorch

`CrowdDataset` subclasses PyTorch's `Dataset`, so it provides:

- `__len__()` — the number of images.
- `__getitem__(idx)` — one image and its target density map.

For each item it:

1. Finds a `.jpg` image.
2. Finds the matching `.h5` density map.
3. Converts the image to RGB.
4. Applies ImageNet normalization because the frontend is based on VGG-16.
5. Randomly flips both image and density map horizontally during training.
6. Resizes the density map to one-eighth of its original height and width.
7. Multiplies the resized map by `8 ** 2` so that its total count is approximately preserved.
8. Adds a channel dimension, producing `(1, height/8, width/8)`.

The image and density map must be flipped together. Flipping only one would make the training label incorrect.

## 5. `model.py`: the CSRNet model

`CSRNet` has three stages.

### Frontend: VGG-16 features

The first 23 layers of VGG-16 extract visual features such as edges, shapes, and patterns. Pooling reduces the feature-map resolution by a factor of eight.

If `pretrained_frontend=True`, these layers start with ImageNet-trained weights. This is called **transfer learning**: knowledge learned from a large image dataset is reused for this task.

### Backend: dilated convolutions

The backend contains six convolution layers with dilation rate 2. A dilated convolution samples pixels with gaps, giving the layer a larger **receptive field** without another pooling operation.

This helps the model understand a wider crowd region while keeping enough spatial detail for a heatmap.

### Output layer

The final `1 x 1` convolution changes 64 feature channels into one channel:

```text
input image -> one-channel density map
```

The model does not directly output a number. The predicted count is calculated as:

```python
predicted_count = predicted_density_map.sum()
```

## 6. `train.py`: learning the model

The training script:

1. Selects CUDA when available, otherwise CPU.
2. Creates training and validation `CrowdDataset` objects.
3. Uses `DataLoader` with batch size 1 because source images have different sizes.
4. Creates a CSRNet model.
5. Uses summed mean-squared error (`nn.MSELoss(reduction="sum")`) between predicted and target density maps.
6. Uses Adam to update model weights.
7. Reduces the learning rate every 10 epochs with `StepLR`.
8. Evaluates on the test split after each configured evaluation interval.
9. Saves the checkpoint with the best validation MAE.

The validation loop uses `torch.no_grad()` and `model.eval()`. These prevent gradient calculation and switch the model to inference behavior.

Run an example:

```powershell
python train.py --dataset part_B --epochs 30 --lr 1e-5 --save_dir checkpoints
```

## 7. `evaluate.py`: measuring predictions

The evaluation script loads a saved checkpoint and runs it over the selected test split.

For every image it:

1. Predicts a density map.
2. Sums predicted and ground-truth maps.
3. Calculates the absolute count error.
4. Accumulates MAE and RMSE-style error.
5. Saves a three-panel visualization for the first few images:
   - original image,
   - ground-truth heatmap,
   - predicted heatmap.

Run it with:

```powershell
python evaluate.py --checkpoint checkpoints\best_model_part_B.pth --dataset part_B --output_dir eval_results
```

In the current code, the printed label `MSE` is calculated as the square root of the mean squared count error, which is technically **RMSE**. MAE remains the mean absolute count error.

## 8. `app.py`: the interactive dashboard

The Streamlit app is the user interface.

### Model loading

`load_csrnet_model()` creates the model, loads an optional `.pth` checkpoint, selects CPU or GPU, and caches the model with `@st.cache_resource`.

### Prediction

`predict_density()`:

1. Converts pixel values from `0..255` to `0..1`.
2. Applies the same ImageNet mean and standard deviation used by the dataset.
3. Changes the image layout to `(batch, channels, height, width)`.
4. Runs inference with gradients disabled.
5. Sums the raw density map to estimate the count.
6. Resizes the map to the original image size.
7. Applies a JET color map and blends it with the original image.
8. Measures inference latency.

### Safety levels

The dashboard compares the predicted count with two user-controlled thresholds:

```text
below warning threshold       SAFE
warning to critical threshold  WARNING
at or above critical threshold CRITICAL
```

The thresholds are operational demo settings, not universal safety standards. A real deployment would need venue-specific capacity studies, human review, and fail-safe controls.

Run the app with:

```powershell
streamlit run app.py
```

## 9. `hardware_simulator.py`: optional actuator integration

`CrowdSafetyActuator` maps a status to a command:

```text
SAFE     -> G
WARNING  -> Y
CRITICAL -> R
```

In mock mode it prints the expected LED, siren, and turnstile action. In non-mock mode it attempts to send the command over a serial port such as `COM3`.

Test the software simulation:

```powershell
python hardware_simulator.py --mock
```

This module is a demonstration interface. It should not directly control safety-critical equipment without authentication, redundant sensors, manual override, and hardware safety testing.

## 10. The notebook

The notebook presents the same pipeline in a more visual order:

1. Project overview and architecture.
2. Imports, reproducibility seed, and hardware detection.
3. Density-map generation and visualization.
4. Dataset loading.
5. CSRNet definition.
6. Validation and training.
7. Evaluation plots.
8. Interactive image testing and batch testing.
9. Hardware actuator demonstration and Arduino firmware example.

Use the notebook when learning or experimenting one cell at a time. Use the scripts when running repeatable training and evaluation jobs.

## 11. Recommended first reading path

If you are new to machine learning, read in this order:

1. Python arrays and functions.
2. Images and NumPy arrays.
3. PyTorch tensors and datasets.
4. Convolutional neural networks.
5. Training, loss, optimizer, and validation.
6. Density maps and CSRNet.
7. Streamlit and serial hardware integration.

The companion file [`LEARNING_RESOURCES.md`](./LEARNING_RESOURCES.md) contains beginner resources for each step.

