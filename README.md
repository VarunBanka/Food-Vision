# Food Vision Model with TensorFlow

This project involves the creation of a food vision model using TensorFlow and EfficientNetB0 as the base model. The model is trained on the CIFAR-100 dataset, which contains 100 classes of images.

## Project Overview

The goal of this project is to build a deep learning model capable of classifying food images into different categories. We used the EfficientNetB0 architecture for feature extraction and performed fine-tuning to improve the model's accuracy.

## Dataset

We utilized the **CIFAR-100** dataset, which consists of 60,000 32x32 color images in 100 classes, with 600 images per class. There are 50,000 training images and 10,000 test images.

## Model Architecture

The model is based on the **EfficientNetB0** architecture, a state-of-the-art convolutional neural network model that balances accuracy and computational efficiency. Below is the summary of the model:

| Layer (type)                  | Output Shape       | Param #    |
| ----------------------------- | ------------------ | ---------- |
| input_layer (InputLayer)       | (None, 224, 224, 3) | 0          |
| efficientnetb0 (Functional)    | (None, 7, 7, 1280)  | 4,049,571  |
| global_avtaging_pooling_2d     | (None, 1280)        | 0          |
| output_layer (Dense)           | (None, 101)         | 129,381    |
| softmax_float32 (Activation)   | (None, 101)         | 0          |

- **Total params:** 12,452,816 (47.50 MB)
- **Trainable params:** 4,136,929 (15.78 MB)
- **Non-trainable params:** 42,023 (164.16 KB)
- **Optimizer params:** 8,273,864 (31.56 MB)

## Training Details

The model was trained for a total of 8 epochs, including 5 epochs of fine-tuning the top layers. Below are the final training metrics:

- **Training Accuracy:** 99.44%
- **Training Loss:** 0.0293
- **Validation Accuracy:** 83.33%
- **Validation Loss:** 0.8154

### Training Process

1. **Initial Training:** The model was initially trained for 3 epochs with the base layers frozen.
2. **Fine-Tuning:** The top layers of the EfficientNetB0 model were unfrozen, and the entire model was fine-tuned for 5 additional epochs.

### Optimization

- **Optimizer:** Adam
- **Learning Rate:** Adaptive learning rate with a scheduler.
- **Loss Function:** Sparse Categorical Crossentropy

## Results

The model achieved a high training accuracy of 99.44% and a validation accuracy of 83.33% after fine-tuning. Although there is some overfitting indicated by the difference between training and validation accuracy, the model shows promising results for food image classification tasks.

## Running the Project

To run this project, you can use Google Colab. Here's how to get started:

1. Open Google Colab: [Google Colab](https://colab.research.google.com/)
2. Upload the notebook (`.ipynb` file) or clone the repository.
3. Make sure to select a GPU runtime by navigating to `Runtime` > `Change runtime type` and selecting `GPU` under the hardware accelerator.
4. Run the notebook cells to train and evaluate the model.

## Contributing

If you'd like to contribute to this project, feel free to fork the repository and submit a pull request. All contributions are welcome!

## License

This project is licensed under the GNU GENERAL PUBLIC License V3. See the [LICENSE](LICENSE) file for details.
