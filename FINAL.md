# Smart Crowd Density Monitoring and Safety Classification System Using CSRNet and YOLOv8

## 1. Introduction

Crowd monitoring is an important aspect of public safety management in locations such as railway stations, airports, shopping malls, stadiums, and other public gathering areas. A large or rapidly increasing crowd can lead to congestion, overcrowding, stampedes, and other safety-related incidents.

Traditional crowd monitoring systems generally depend on manual observation or CCTV operators. These approaches have limitations because monitoring multiple camera feeds continuously is difficult, time-consuming, and prone to human error.

The proposed Smart Crowd Density Monitoring and Safety Classification System uses deep learning and computer vision techniques to estimate the number of people in an image and classify the crowd condition into different safety levels.

The system primarily uses:

- CSRNet (Congested Scene Recognition Network): Estimates crowd density by generating a density map.
- YOLOv8: Detects individual people in relatively sparse crowd scenes.
- Safety Classification Module: Converts the estimated crowd count into a safety category.
- Performance Evaluation Module: Measures counting error, accuracy, and inference latency.

The system is evaluated using the ShanghaiTech Crowd Counting Dataset, which contains images from both relatively sparse and highly congested crowd scenes.

## 2. Objectives

The main objectives of the project are:

1. To develop an automated crowd density monitoring system using deep learning.
2. To estimate the number of people present in an image.
3. To generate density maps representing the spatial distribution of people.
4. To use CSRNet for estimating crowd counts in congested scenes.
5. To incorporate YOLOv8 for detecting individuals in sparse scenes.
6. To classify crowd conditions into SAFE, WARNING, and CRITICAL categories.
7. To evaluate the system using MAE, RMSE, relative error, and inference latency.
8. To investigate the performance of the system on different crowd-density datasets.

## 3. Technologies and Software Used

| Technology       | Purpose                                                |
| ---------------- | ------------------------------------------------------ |
| Python           | Main programming language                              |
| PyTorch          | Deep learning model development and training           |
| Torchvision      | Pretrained models and image transformations            |
| OpenCV           | Image processing and computer vision                   |
| NumPy            | Numerical operations                                   |
| Pandas           | Dataset handling and result analysis                   |
| Matplotlib       | Data visualization                                     |
| SciPy            | Gaussian density map generation and KD-tree operations |
| CSRNet           | Crowd density estimation                               |
| YOLOv8           | Individual person detection                            |
| CUDA             | GPU acceleration                                       |
| Jupyter Notebook | Model development and experimentation                  |

### Hardware and environment

- GPU: NVIDIA RTX 2050 with 4 GB VRAM.
- PyTorch version: 2.6.0.
- CUDA version: 12.4.

## 4. Dataset Description

The project uses the ShanghaiTech Crowd Counting Dataset, a widely used dataset for crowd counting research.

It consists of two parts:

- Part A: Contains highly congested crowd scenes with considerable variations in crowd density.
- Part B: Contains relatively less congested scenes, generally captured in street environments.

Each image is accompanied by head-point annotations indicating the locations of people in the image.

### 4.1 Dataset distribution

| Dataset | Training Images | Testing Images | Total |
| ------- | --------------- | -------------- | ----- |
| Part A  | 300             | 182            | 482   |
| Part B  | 400             | 316            | 716   |
| Total   | 700             | 498            | 1198  |

### 4.2 Training and validation split

The training images were divided into training and validation subsets using a 90:10 split with a fixed random seed of 42.

| Dataset | Training | Validation | Official Testing |
| ------- | -------- | ---------- | ---------------- |
| Part A  | 270      | 30         | 182              |
| Part B  | 360      | 40         | 316              |
| Total   | 630      | 70         | 498              |

The official test images were kept separate from the training and validation subsets.

This separation is important because the test dataset should not be used to optimize model parameters.

## 5. System Architecture

The system consists of multiple processing stages.

### 5.1 Input image acquisition

The system receives an image from a camera or an existing image dataset.

The input image is passed to the image preprocessing stage before being processed by the crowd estimation models.

### 5.2 Image preprocessing

Preprocessing prepares the image for deep learning inference.

The major operations include:

- Loading the image.
- Converting the image into RGB format.
- Resizing or cropping the image.
- Converting the image into a PyTorch tensor.
- Normalizing pixel values using ImageNet mean and standard deviation.

The ImageNet normalization operation is:

\[ I_{\text{normalized}} = \frac{I-\mu}{\sigma} \]

where:

- \(I\) represents the input image.
- \(\mu\) represents the ImageNet mean.
- \(\sigma\) represents the ImageNet standard deviation.

The normalization is performed independently for the three RGB channels.

### 5.3 Crowd estimation

The system uses CSRNet as its primary crowd density estimation model.

For relatively sparse images, YOLOv8 is also used to detect individual people.

The output of the selected model is used to estimate the total crowd count.

### 5.4 Safety classification

The estimated crowd count is passed to the safety classification module.

The module categorizes the crowd condition into:

- SAFE
- WARNING
- CRITICAL

The exact thresholds used for these categories should be taken from the implementation code. They should not be assumed from the CSV results alone.

### 5.5 Result generation

The final output includes:

- Predicted crowd count.
- Ground-truth count, when available for evaluation.
- Counting error.
- Reported accuracy.
- Safety classification.
- Active detection engine.
- Inference latency.

## 6. CSRNet: Crowd Density Estimation

### 6.1 Introduction to CSRNet

CSRNet stands for Congested Scene Recognition Network.

It is a deep learning architecture designed to estimate crowd density in images, particularly in highly congested environments.

Traditional object detection methods attempt to identify and draw a bounding box around every individual person. However, this becomes difficult when people are densely packed together because:

- People overlap with each other.
- Some individuals are partially visible.
- Individual body features may be too small.
- Occlusion makes person detection difficult.

CSRNet addresses this problem by predicting a density map rather than detecting every individual person separately.

The total crowd count is obtained by summing the values of the predicted density map.

### 6.2 CSRNet architecture

CSRNet consists of two major components:

1. Frontend feature extractor.
2. Backend density map estimator.

#### A. Frontend feature extractor

The frontend uses the first 23 feature-extraction modules of VGG16 pretrained on ImageNet.

Its main purpose is to extract visual features from the input image.

These features contain information about:

- Human heads.
- Shapes and edges.
- Texture patterns.
- Local crowd structures.
- Spatial characteristics.

Using pretrained VGG16 weights provides a useful starting point for learning crowd-related features.

#### B. Backend density estimator

The backend consists of six dilated convolutional layers.

The channel configuration used in the implementation is:

\[ 512 \rightarrow 512 \rightarrow 512 \rightarrow 256 \rightarrow 128 \rightarrow 64 \]

The backend uses a dilation rate of 2.

Dilated convolutions increase the receptive field without requiring a proportional increase in the number of parameters or a reduction in feature-map resolution.

This allows the network to capture information from a larger spatial region, which is useful for understanding crowded scenes.

#### C. Output layer

The final output layer is a \(1 \times 1\) convolution that produces a single-channel density map.

For a \(512 \times 512\) input crop, the output density map is approximately:

\[ 64 \times 64 \]

This corresponds to a spatial downsampling factor of approximately 8.

### 6.3 Working of CSRNet

The working process can be represented as:

\[ \text{Input Image} \rightarrow \text{VGG16 Frontend} \rightarrow \text{Dilated CNN Backend} \rightarrow \text{Density Map} \rightarrow \text{Crowd Count} \]

The density map represents the estimated distribution of people across the image.

Each pixel in the density map represents a contribution to the estimated crowd count.

The final count is calculated by summing all density-map values:

\[ \boxed{ C_{\text{pred}} = \sum_{i=1}^{H} \sum_{j=1}^{W} D(i,j) } \]

where:

- \(C_{\text{pred}}\) is the predicted crowd count.
- \(D(i,j)\) is the density value at position \((i,j)\).
- \(H\) is the density-map height.
- \(W\) is the density-map width.

For example, if the sum of all density-map values is approximately 250, the estimated crowd count is approximately 250 people.

## 7. Density Map Generation

### 7.1 What is a density map?

A density map is a representation of crowd distribution in an image.

Instead of assigning a bounding box to every person, a density map assigns numerical values to image locations.

Regions containing more people generally have higher accumulated density values.

The ground-truth density map is generated from the head-point annotations provided in the dataset.

### 7.2 Gaussian distribution

Each annotated head point is converted into a Gaussian distribution.

A two-dimensional Gaussian function can be represented as:

\[ G(x,y) = \frac{1}{2\pi\sigma^2} \exp\left( -\frac{(x-x_0)^2+(y-y_0)^2} {2\sigma^2} \right) \]

where:

- \((x_0,y_0)\) represents the annotated head position.
- \(\sigma\) controls the spread of the Gaussian distribution.
- \((x,y)\) represents a position in the image.

The Gaussian distribution produces a smooth density contribution around each annotated head point.

### 7.3 Adaptive Gaussian kernel

The implementation uses an adaptive Gaussian kernel based on the distance between neighboring head annotations.

The nearest-neighbor distances are calculated using a KD-tree.

The standard deviation is determined using:

\[ \sigma = 0.30 \times \overline{d} \]

where \(\overline{d}\) represents the average nearest-neighbor distance.

The calculated sigma is clipped to the range:

\[ 1 \leq \sigma \leq 15 \]

This allows the Gaussian spread to adapt according to the local crowd density.

- In dense regions, neighboring heads are closer together, resulting in smaller Gaussian spreads.
- In sparse regions, neighboring heads are farther apart, resulting in larger Gaussian spreads.

Each Gaussian patch is normalized so that its total contribution is approximately one person.

The final ground-truth density map is generated by adding the Gaussian contributions from all annotated heads.

\[ D(x,y) = \sum_{n=1}^{N} G_n(x,y) \]

where \(N\) represents the total number of annotated people.

Consequently, the sum of the ground-truth density map approximately equals the number of annotated people:

\[ \boxed{ \sum_{x,y} D(x,y) \approx N } \]

### 7.4 Density map caching

Generating density maps repeatedly can increase preprocessing time.

Therefore, the implementation caches generated density maps in NumPy format.

This reduces repeated computation during subsequent training runs.

When MATLAB annotations are unavailable, the implementation can use compatible HDF5 density maps as a fallback.

## 8. Data Preprocessing and Augmentation

The input images and their corresponding density maps must undergo matching transformations.

This is important because the density map must continue to represent the same spatial positions as the input image.

### 8.1 Image cropping

The training images are randomly cropped into:

\[ 512 \times 512 \]

patches.

This allows the model to train on smaller image regions while increasing the number of training samples generated from the original images.

### 8.2 Horizontal flipping

Random horizontal flipping is used as a data augmentation technique.

Both the image and its density map are flipped together.

This helps the model learn crowd patterns independently of horizontal orientation.

### 8.3 Density map resizing

The density map is resized to match the spatial resolution expected by the network.

During resizing, the total density value must be preserved.

This is important because the sum of the density map represents the estimated number of people.

If the density map is resized without preserving its sum, the estimated crowd count may change even though the actual crowd has not changed.

## 9. CSRNet Training Configuration

The implementation uses transfer learning with pretrained VGG16 weights.

The frontend and backend are trained using different learning rates.

### 9.1 Hyperparameters

| Parameter                 | Configuration         |
| ------------------------- | --------------------- |
| Model                     | CSRNet                |
| Frontend                  | Pretrained VGG16      |
| Input crop size           | 512 × 512             |
| Output density map        | Approximately 64 × 64 |
| Batch size                | 1                     |
| Maximum configured epochs | 80                    |
| Optimizer                 | Adam                  |
| Frontend learning rate    | \(10^{-6}\)         |
| Backend learning rate     | \(10^{-4}\)         |
| Weight decay              | \(10^{-4}\)         |
| Loss function             | Mean Squared Error    |
| Gradient clipping         | Maximum norm 10       |
| Learning rate scheduler   | ReduceLROnPlateau     |
| Scheduler factor          | 0.5                   |
| Early stopping patience   | 12                    |
| Mixed precision           | Enabled on CUDA       |

### 9.2 Loss function

The model is trained using Mean Squared Error (MSE) between the predicted density map and the ground-truth density map.

The loss function is:

\[ \boxed{ L_{\text{MSE}} = \frac{1}{HW} \sum_{i=1}^{H} \sum_{j=1}^{W} \left( D_{\text{pred}}(i,j) - D_{\text{gt}}(i,j) \right)^2 } \]

where:

- \(D_{\text{pred}}\) is the predicted density map.
- \(D_{\text{gt}}\) is the ground-truth density map.
- \(H\) and \(W\) are the density-map dimensions.

The implementation uses summed pixel-wise MSE for its training loss. The expression above shows the conventional mean-normalized MSE form; the actual numerical loss scale depends on the reduction used.

The purpose of the loss function is to minimize the difference between the predicted density map and the actual density map.

### 9.3 Optimizer

The Adam optimizer is used to update model parameters.

It combines the advantages of momentum-based optimization and adaptive learning rates.

The frontend learning rate is smaller because the VGG16 feature extractor is already pretrained.

The backend learning rate is larger because the backend is responsible for learning the density estimation task.

### 9.4 Learning rate scheduler

The implementation uses `ReduceLROnPlateau`.

The learning rate is reduced by a factor of 0.5 when the monitored validation metric stops improving.

This helps the model perform smaller parameter updates when learning progress slows.

### 9.5 Early stopping

Early stopping is configured with a patience of 12 epochs.

It is intended to stop training when the monitored validation performance does not improve for the configured number of epochs.

This can help reduce unnecessary training and limit overfitting.

### 9.6 Mixed precision training

Automatic Mixed Precision (AMP) is enabled on CUDA.

It allows selected operations to use lower-precision floating-point calculations.

This can reduce GPU memory consumption and improve computational efficiency.

## 10. CSRNet Training Results

The training notebook contains seven complete training epochs and a partially completed eighth epoch.

Although the maximum training duration was configured as 80 epochs, the recorded execution did not complete all 80 epochs.

### 10.1 Training and validation results

| Epoch | Training Loss | Part A Validation MAE | Part B Validation MAE | Macro Relative Error |
| ----- | ------------- | --------------------- | --------------------- | -------------------- |
| 1     | 45.8547       | 244.5135              | 51.8450               | 38.94%               |
| 2     | 38.3015       | 179.3913              | 32.4562               | 23.98%               |
| 3     | 36.2313       | 121.6280              | 20.1062               | 16.93%               |
| 4     | 31.2658       | 254.7204              | 106.6667              | 78.85%               |
| 5     | 30.8295       | 109.3822              | 12.9162               | 13.08%               |
| 6     | 27.0733       | 185.3864              | 19.0095               | 19.03%               |
| 7     | 33.6549       | 112.1953              | 16.7965               | 14.15%               |

### 10.2 Best recorded checkpoint

The best recorded checkpoint corresponds to epoch 5.

| Metric                | Value    |
| --------------------- | -------- |
| Best recorded epoch   | 5        |
| Part A validation MAE | 109.3822 |
| Part B validation MAE | 12.9162  |
| Macro relative error  | 13.08%   |

The checkpoint was saved separately as:

`csrnet_v2_pretrained_AB_best.pth`

The original Part-B checkpoint was not overwritten.

### 10.3 Interpretation

The training results show that the model's validation performance improved considerably during the first three epochs.

However, epoch 4 shows a substantial increase in validation errors.

The model then achieved its best recorded macro relative error in epoch 5.

The results also show that validation performance fluctuated across epochs. Therefore, the lowest training loss alone should not be used to select the best checkpoint.

The validation metrics are more relevant for checkpoint selection.

## 11. YOLOv8: Person Detection

### 11.1 Introduction

YOLOv8 is a real-time object detection architecture.

Unlike CSRNet, which predicts a density map, YOLOv8 identifies individual objects and generates bounding boxes around them.

In this project, YOLOv8 is used as an additional detection engine for relatively sparse crowd scenes.

### 11.2 Working principle

The input image is passed to the YOLOv8 model.

The model processes the image and produces:

- Bounding box coordinates.
- Object class predictions.
- Confidence scores.

The system identifies detections belonging to the person class.

The estimated count can be obtained by counting the valid person detections.

\[ \boxed{ C_{\text{YOLO}} = \sum_{i=1}^{N} \mathbb{1}(c_i=\text{person}) } \]

where:

- \(N\) is the number of detections.
- \(c_i\) is the predicted class of detection \(i\).
- \(\mathbb{1}\) is an indicator function that returns 1 when the condition is true and 0 otherwise.

### 11.3 CSRNet versus YOLOv8

| Feature                 | CSRNet                                 | YOLOv8                                |
| ----------------------- | -------------------------------------- | ------------------------------------- |
| Main approach           | Density estimation                     | Object detection                      |
| Output                  | Density map                            | Bounding boxes                        |
| Counting method         | Sum of density values                  | Count person detections               |
| Dense crowd scenes      | Designed for congested scenes          | Can struggle with occlusion           |
| Individual localization | Not directly provided                  | Provided                              |
| Main advantage          | Handles highly congested scenes        | Identifies individual people          |
| Main limitation         | Does not directly identify each person | Missed detections can affect counting |

### 11.4 Engine distribution in evaluation

The CSV evaluation files show that:

- CSRNet was used for 487 images.
- YOLOv8 was used for 11 images.

This means that CSRNet was the primary engine in the recorded evaluation.

The YOLOv8 sample size is too small to draw a reliable independent performance comparison between the two models.

## 12. Evaluation Metrics

The system is evaluated using multiple metrics to understand the quality of crowd estimation.

### 12.1 Mean Absolute Error (MAE)

MAE measures the average absolute difference between predicted and ground-truth crowd counts.

\[ \boxed{ MAE = \frac{1}{N} \sum_{i=1}^{N} \left| C_i^{\text{pred}} - C_i^{\text{gt}} \right| } \]

where:

- \(N\) is the total number of images.
- \(C_i^{\text{pred}}\) is the predicted count.
- \(C_i^{\text{gt}}\) is the ground-truth count.

A lower MAE indicates better counting performance.

### 12.2 Root Mean Squared Error (RMSE)

RMSE measures the square root of the average squared counting error.

\[ \boxed{ RMSE = \sqrt{ \frac{1}{N} \sum_{i=1}^{N} \left( C_i^{\text{pred}} - C_i^{\text{gt}} \right)^2 } } \]

RMSE gives greater importance to large errors because the errors are squared before averaging.

### 12.3 Mean Relative Error

Mean Relative Error measures the average counting error relative to the ground-truth count.

\[ \boxed{ MRE = \frac{100}{N} \sum_{i=1}^{N} \frac{ \left|C_i^{\text{pred}}-C_i^{\text{gt}}\right| }{ C_i^{\text{gt}} } } \]

This metric expresses the error as a percentage.

A lower value indicates that the predicted counts are closer to the ground-truth counts relative to their actual sizes.

### 12.4 Accuracy

The CSV files contain an `Accuracy (%)` column.

The average reported accuracy is calculated over the non-missing entries in that column.

However, the exact formula used to generate this field is not established by the CSV files alone.

Therefore, it should be described as reported accuracy, rather than assuming that it is a standard regression accuracy or classification accuracy metric.

### 12.5 Inference latency

Inference latency measures the time required to process an image.

It is recorded in milliseconds.

The average latency is:

\[ \boxed{ T_{\text{avg}} = \frac{1}{N} \sum_{i=1}^{N} T_i } \]

where \(T_i\) is the recorded latency for the \(i\)-th image.

Lower latency generally indicates faster inference.

However, inference latency alone does not establish the complete end-to-end speed of a live monitoring system. Camera capture, preprocessing, data transfer, and result display can add additional time.

## 13. Experimental Results: Part A

The Part A evaluation CSV contains 182 image records.

All 182 records use the CSRNet Congested Density Engine.

### 13.1 Performance metrics

| Metric                     | Result   |
| -------------------------- | -------- |
| Number of images           | 182      |
| Average predicted count    | 484.06   |
| Average ground-truth count | 428.24   |
| Mean Absolute Error        | 104.56   |
| Root Mean Squared Error    | 149.43   |
| Median Absolute Error      | 70.00    |
| Mean Relative Error        | 26.52%   |
| Median Relative Error      | 22.07%   |
| Average reported accuracy  | 73.97%   |
| Average inference latency  | 33.76 ms |
| Median inference latency   | 44.15 ms |
| 95th percentile latency    | 64.37 ms |
| Maximum latency            | 88.10 ms |

### 13.2 Counting performance

The average predicted count is higher than the average ground-truth count.

The mean signed error is:

\[ \text{Mean Signed Error} = \frac{1}{N} \sum_{i=1}^{N} (C_i^{\text{pred}}-C_i^{\text{gt}}) \]

For Part A:

\[ \boxed{ \text{Mean Signed Error}=+55.82 } \]

This indicates an overall tendency toward overestimation in the evaluated images.

The MAE of 104.56 indicates that the average absolute counting difference is approximately 105 people per image.

### 13.3 Safety classification distribution

| Safety Level | Number of Images | Percentage |
| ------------ | ---------------- | ---------- |
| SAFE         | 4                | 2.20%      |
| WARNING      | 51               | 28.02%     |
| CRITICAL     | 127              | 69.78%     |
| Total        | 182              | 100%       |

Most Part A images were classified as CRITICAL.

This is consistent with the highly congested nature of many images in Part A, although the classification thresholds themselves must be verified from the implementation.

### 13.4 Interpretation

The Part A results indicate that crowd counting is more challenging in highly congested scenes.

Possible reasons include:

- Significant occlusion.
- Overlapping individuals.
- High variation in crowd density.
- Small head sizes in distant regions.
- Complex background patterns.

The relatively high MAE and RMSE indicate that the system still has room for improvement on highly congested images.

## 14. Experimental Results: Part B

The Part B evaluation CSV contains 316 image records.

Of these:

- 305 records use the CSRNet Congested Density Engine.
- 11 records use the YOLOv8 Sparse Detector.

### 14.1 Performance metrics

| Metric                     | Result   |
| -------------------------- | -------- |
| Number of images           | 316      |
| Average predicted count    | 121.55   |
| Average ground-truth count | 116.21   |
| Mean Absolute Error        | 14.47    |
| Root Mean Squared Error    | 23.36    |
| Median Absolute Error      | 7.70     |
| Mean Relative Error        | 12.71%   |
| Median Relative Error      | 9.32%    |
| Average reported accuracy  | 87.90%   |
| Average inference latency  | 5.89 ms  |
| Median inference latency   | 5.00 ms  |
| 95th percentile latency    | 11.00 ms |
| Maximum latency            | 46.80 ms |

### 14.2 Counting performance

The mean signed error is:

\[ \boxed{ \text{Mean Signed Error}=+5.33 } \]

This indicates a smaller overall overestimation tendency compared with Part A.

The MAE is 14.47, which means that the average absolute difference between the predicted and ground-truth counts is approximately 14 to 15 people per image.

### 14.3 Safety classification distribution

| Safety Level | Number of Images | Percentage |
| ------------ | ---------------- | ---------- |
| SAFE         | 166              | 52.53%     |
| WARNING      | 123              | 38.92%     |
| CRITICAL     | 27               | 8.54%      |
| Total        | 316              | 100%       |

More than half of the Part B images were classified as SAFE.

The proportion of CRITICAL images is substantially lower than that in Part A.

### 14.4 Interpretation

Part B shows lower counting errors and lower inference latency than Part A.

This is an observed difference in the evaluation results.

Part B contains relatively less congested scenes, which may make crowd estimation easier. However, the difference cannot be attributed exclusively to crowd density because the two dataset parts also differ in scene characteristics and image distributions.

## 15. Combined Evaluation Results

The combined evaluation contains 498 images:

\[ N = 182 + 316 = 498 \]

The following metrics are calculated from the individual predicted and ground-truth counts rather than simply averaging the Part A and Part B metric values.

### 15.1 Combined performance metrics

| Metric                     | Combined Result |
| -------------------------- | --------------- |
| Total images               | 498             |
| Average predicted count    | 254.03          |
| Average ground-truth count | 230.25          |
| Mean Absolute Error        | 47.40           |
| Root Mean Squared Error    | 92.23           |
| Median Absolute Error      | 16.00           |
| Mean Relative Error        | 17.76%          |
| Average reported accuracy  | 82.82%          |
| Average inference latency  | 16.07 ms        |
| Median inference latency   | 5.85 ms         |
| 95th percentile latency    | 56.03 ms        |
| Maximum inference latency  | 88.10 ms        |

The average reported accuracy is calculated using 496 non-missing entries, because two records have missing accuracy values.

### 15.2 Combined safety classification

| Safety Level | Number of Images | Percentage |
| ------------ | ---------------- | ---------- |
| SAFE         | 170              | 34.14%     |
| WARNING      | 174              | 34.94%     |
| CRITICAL     | 154              | 30.92%     |
| Total        | 498              | 100%       |

The combined results show that approximately one-third of the evaluated images fall into each of the three safety categories.

### 15.3 Active engine distribution

| Active Engine                   | Images | Percentage |
| ------------------------------- | ------ | ---------- |
| CSRNet Congested Density Engine | 487    | 97.79%     |
| YOLOv8 Sparse Detector          | 11     | 2.21%      |
| Total                           | 498    | 100%       |

CSRNet is the primary model used in the evaluation.

## 16. Comparison Between Part A and Part B

The following table compares the performance of the system on both dataset parts.

| Metric              | Part A   | Part B  |
| ------------------- | -------- | ------- |
| Images              | 182      | 316     |
| MAE                 | 104.56   | 14.47   |
| RMSE                | 149.43   | 23.36   |
| Mean Relative Error | 26.52%   | 12.71%  |
| Reported Accuracy   | 73.97%   | 87.90%  |
| Average Latency     | 33.76 ms | 5.89 ms |
| CRITICAL Images     | 69.78%   | 8.54%   |

### Observations

1. Part B achieved a considerably lower MAE than Part A.
2. Part B also achieved a lower RMSE, indicating fewer or smaller large counting errors.
3. The mean relative error was lower for Part B.
4. The average reported accuracy was higher for Part B.
5. The average inference latency was lower for Part B.
6. Part A contained a much higher proportion of CRITICAL safety classifications.

The results indicate that the system performed better on the evaluated Part B images than on the evaluated Part A images.

However, these results should not be interpreted as proof that the model will always perform better on every sparse scene than every dense scene.

## 17. Error Analysis

Error analysis helps identify the situations in which the system struggles to estimate crowd counts accurately.

### 17.1 Overestimation

The combined mean signed error is positive:

\[ \boxed{ +23.78 } \]

This indicates that the system has an overall tendency to predict higher crowd counts than the ground truth.

Possible reasons include:

- Incorrect density estimation.
- Background textures resembling crowd features.
- Overlapping density contributions.
- Difficulty estimating head locations in complex scenes.

### 17.2 Underestimation

Underestimation can occur when:

- Individuals are heavily occluded.
- Heads are too small to produce strong visual features.
- People are partially outside the image.
- Crowd regions are extremely dense.

### 17.3 Large errors

RMSE is substantially higher than MAE for both dataset parts.

This indicates that some images have relatively large counting errors.

For example:

| Dataset | MAE    | RMSE   |
| ------- | ------ | ------ |
| Part A  | 104.56 | 149.43 |
| Part B  | 14.47  | 23.36  |

The difference between MAE and RMSE suggests that large errors contribute noticeably to the overall error distribution.

### 17.4 Relative error distribution

The proportion of images within selected relative-error ranges is:

| Metric                           | Part A | Part B |
| -------------------------------- | ------ | ------ |
| Images within 10% relative error | 20.88% | 53.16% |
| Images within 20% relative error | 45.60% | 83.23% |

This provides another way to understand the difference between the two evaluation sets.

## 18. Inference Latency Analysis

Inference latency is important for crowd monitoring because the system may need to process multiple images continuously.

### 18.1 Latency comparison

| Metric          | Part A   | Part B   |
| --------------- | -------- | -------- |
| Average latency | 33.76 ms | 5.89 ms  |
| Median latency  | 44.15 ms | 5.00 ms  |
| 95th percentile | 64.37 ms | 11.00 ms |
| Maximum latency | 88.10 ms | 46.80 ms |

Part B has substantially lower average latency than Part A.

This may reflect differences in the processing paths and characteristics of the evaluated images.

### 18.2 Theoretical processing rate

A rough reciprocal estimate of processing rate can be calculated from average latency:

\[ FPS_{\text{theoretical}} = \frac{1000}{T_{\text{avg}}} \]

For Part A:

\[ FPS_{\text{A}} = \frac{1000}{33.76} \approx 29.62 \]

For Part B:

\[ FPS_{\text{B}} = \frac{1000}{5.89} \approx 169.78 \]

These are theoretical reciprocal latency estimates, not measured end-to-end video frame rates.

Actual video performance depends on additional factors such as camera capture, preprocessing, image transfer, and result rendering.

## 19. Overall System Workflow

The overall system can be represented using the following sequence:

```
                 Input Image / Camera
                         |
                         v
                 Image Preprocessing
                         |
                         v
                Crowd Estimation Stage
                         |
              +----------+----------+
              |                     |
              v                     v
            CSRNet                YOLOv8
         Density Map           Person Detection
              |                     |
              v                     v
         Density Sum          Count Detections
              |                     |
              +----------+----------+
                         |
                         v
                  Estimated Count
                         |
                         v
                Safety Classification
                         |
                         v
              SAFE / WARNING / CRITICAL
                         |
                         v
                  Result Generation
                         |
                         v
             Count, Error, Latency, Status
```

The diagram represents the model alternatives used by the system. It does not imply that both models necessarily process every image simultaneously.

## 20. Advantages of the Proposed System

1. Automated crowd monitoring: Reduces dependence on continuous manual observation.
2. Density estimation: CSRNet can estimate crowd counts without detecting every individual person.
3. Congested scene support: CSRNet is designed to handle crowded environments with overlapping people.
4. Individual detection: YOLOv8 provides bounding-box-based person detection.
5. Safety classification: Converts crowd-count estimates into understandable safety categories.
6. Quantitative evaluation: Uses multiple metrics to measure model performance.
7. GPU acceleration: CUDA allows faster model computation.
8. Adaptability: The architecture can be extended to process images from multiple cameras.

## 21. Limitations

Despite its advantages, the proposed system has several limitations.

### 21.1 Counting errors in highly congested scenes

Part A shows a relatively high MAE and RMSE.

This indicates that the system still struggles with some highly congested scenes.

### 21.2 Dependence on image quality

Low-resolution images, poor lighting, motion blur, and camera angles can affect crowd estimation.

### 21.3 Limited YOLOv8 evaluation

Only 11 images in the evaluation CSV used YOLOv8.

Therefore, the available results are insufficient to establish its independent counting performance.

### 21.4 Safety threshold dependency

The safety classification depends on the thresholds defined in the implementation.

A crowd count alone does not necessarily indicate the actual risk level because safe capacity also depends on the physical area, crowd movement, exits, and environmental conditions.

### 21.5 Training completion

The training configuration specifies a maximum of 80 epochs, but the recorded execution contains seven complete epochs and a partially completed eighth epoch.

Therefore, the results should be described as results from the recorded training execution, not as a completed 80-epoch training experiment.

### 21.6 Real-time deployment

The recorded inference latency does not include every component of a complete live camera monitoring pipeline.

Additional deployment testing is required before claiming guaranteed real-time performance.

## 22. Future Scope

The project can be improved in several ways.

1. Real-time CCTV integration: Connect the system to live camera feeds for continuous monitoring.
2. Multi-camera monitoring: Process multiple camera streams simultaneously.
3. Improved crowd counting: Experiment with newer crowd-counting architectures and stronger training configurations.
4. Better safety assessment: Consider crowd density per unit area rather than relying only on the total crowd count.
5. Movement analysis: Incorporate optical flow or tracking to identify unusual crowd movement.
6. Alert generation: Generate notifications when crowd conditions reach a critical level.
7. Web dashboard: Display crowd counts, density maps, camera status, and safety classifications.
8. Model optimization: Apply quantization, pruning, or other optimization techniques for deployment on resource-constrained devices.
9. Larger evaluation: Test YOLOv8 and CSRNet independently on a larger and consistently defined evaluation set.

## 23. Conclusion

The project presents a deep learning-based approach for automated crowd density monitoring and safety classification using CSRNet and YOLOv8. CSRNet estimates crowd density through density maps, while YOLOv8 provides individual person detection in relatively sparse scenes. The estimated crowd count is subsequently used to classify crowd conditions into SAFE, WARNING, and CRITICAL categories.

The system was evaluated on 498 images from the ShanghaiTech Part A and Part B datasets. The combined evaluation achieved an MAE of 47.40, an RMSE of 92.23, and a mean relative error of 17.76%. The average recorded inference latency was 16.07 ms.

The results show better counting performance on Part B than on Part A, highlighting the challenges associated with highly congested scenes. Overall, the project demonstrates the potential of deep learning-based crowd estimation for intelligent crowd monitoring. Further improvements in counting accuracy, safety assessment, and real-time deployment can make the system more suitable for practical public-space monitoring.

## 24. Important Equations Summary

The following equations are the main mathematical expressions used in the project.

1\. Crowd count from density map

\[ C_{\text{pred}} = \sum_{i=1}^{H} \sum_{j=1}^{W} D(i,j) \]

2\. Gaussian density function

\[ G(x,y) = \frac{1}{2\pi\sigma^2} \exp\left( -\frac{(x-x_0)^2+(y-y_0)^2} {2\sigma^2} \right) \]

3\. Ground-truth density map

\[ D(x,y) = \sum_{n=1}^{N}G_n(x,y) \]

4\. Adaptive Gaussian standard deviation

\[ \sigma = 0.30 \times \overline{d} \]

5\. Mean Absolute Error

\[ MAE = \frac{1}{N} \sum_{i=1}^{N} \left| C_i^{\text{pred}} - C_i^{\text{gt}} \right| \]

6\. Root Mean Squared Error

\[ RMSE = \sqrt{ \frac{1}{N} \sum_{i=1}^{N} \left( C_i^{\text{pred}} - C_i^{\text{gt}} \right)^2 } \]

7\. Mean Relative Error

\[ MRE = \frac{100}{N} \sum_{i=1}^{N} \frac{ \left|C_i^{\text{pred}}-C_i^{\text{gt}}\right| }{ C_i^{\text{gt}} } \]

8\. Mean inference latency

\[ T_{\text{avg}} = \frac{1}{N} \sum_{i=1}^{N}T_i \]

9\. Theoretical processing rate

\[ FPS_{\text{theoretical}} = \frac{1000}{T_{\text{avg}}} \]

10\. YOLOv8 person count

\[ C_{\text{YOLO}} = \sum_{i=1}^{N} \mathbb{1}(c_i=\text{person}) \]

## 25. Suggested Figures for the Report

Include the following figures in the appropriate sections to make the report more complete:

1. Overall system architecture.
2. CSRNet architecture showing the VGG16 frontend and dilated convolution backend.
3. Original input image and its corresponding ground-truth density map.
4. CSRNet-predicted density map.
5. YOLOv8 person detection with bounding boxes.
6. Training loss versus epoch.
7. Part A versus Part B MAE comparison.
8. Safety classification distribution.
9. Part A versus Part B inference latency comparison.

Important reporting note: Keep the CSRNet training-validation results separate from the CSV evaluation results. The available records do not establish that the CSV metrics were produced using the newly saved epoch-5 checkpoint. Also, verify the safety thresholds and the formula used for the CSV's `Accuracy (%)` field directly from the implementation before documenting them as exact definitions.

The equations above are written using standard LaTeX display-math notation, so they should render as mathematical expressions in Markdown viewers that support LaTeX math.