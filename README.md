# Vision Stack
Vision Stack is a ros package that makes processing image data intuitive and straight forward. By implementing an array-like data structure to stack processing for feature abstraction and analysis for data extraction, Vision Stack provides all the tools necessary for tackling any vision tasks.

Author: Daniel Parra

## Installation

Use the git submodule manager to add vision stack to your repository.

```bash
git submodule add https://github.com/uf-mil/vision_stack.git path/to/submodule
```

## Usage

```python
from PIL import Image
import numpy as np
from vision_stack import VisionStack, BinThresholdingLayer, CannyLayer, ColorMagnificationLayer, CustomLayer, GaussianLayer, GrayscaleLayer, HoughTransformLayer, MinMaxNormalizationLayer, ZScoreNormalizationLayer, RobustScalingLayer, ObjectDetectionLayer, ResizeLayer, RGBMagnificationLayer, SobelLayer, UnderWaterImageEnhancementLayer

# Initialize vision stack data structure
def myCustomImageProcessing(img, *args):
     ...
     return (processed_img, extracted_info) # If you prefer just to do preprocessing of an image, return (processed_img, None) instead

vs = VisionStack(
            layers=[
                BinThresholdingLayer(150,250), # Converts image to grayscale if image is not grayscale and extracts pixels with values between 150 and 250.
                CannyLayer(50,100), # Simplified canny filter that uses cv2.Canny to pass a canny filter over an image with the low value (50) threshold for soft edge detection and the high value (100) for strong edges detection.
                ColorMagnification((23,156,234)), # Highlights objects with this color (23,156,234) in an image.
                CustomLayer('your_layer_name', myCustomImageProcessing, arg1, arg2, ...), # Add your own defined function to the vision stack process and pass as many necessary arguments.
                GaussianLayer((11,11), 50), # Pass a gaussian filter of kernal size (11,11) (kernal size must be of odd numbered dimensions) with a sigma value of 50.
                GrayscaleLayer(), # Convert image to grayscale
                HoughTransformLayer(threshold=100, min_line_length=20, max_line_gap=10, pass_post_processing_img=True), # Pass a Hough Transform filter over an image to extract lines. Setting pass_post_processing_img will push the image with hough transform lines to the next layer.
                # Different types of normalization layers:
                MinMaxNormalizationLayer(),
                ZScoreNormalizationLayer(),
                RobustScalingLayer(),
                ObjectDetectionLayer('path/to/weights.pt|tflite', conf_thres=0.5, iou_thres=0.5, class_names_array=['cls1','cls2','cls3',...], colors_array=[(255,0,0),(0,255,0),(0,0,255),...], pass_post_processing_img = False), # Access YOLO weights and make predictions on the image provided by the previous layer. Setting pass_post_processing_img will push the image with bounding boxes to the next layer.
                ResizeLayer(960, 608), # Resize the image from the previous layer.
                RGBMagnificationLayer('R'|'G'|'B'), # Magnifies the provided channel respectively.
                SobelLayer((5,5)), # Passes a sobel edge detection layer with a kernal size of (5,5) over the image.
                UnderWaterImageEnhancementLayer(), # Uses a generative AI model to improve underwater images (good for murky waters).
                # Include as many layers in any combination as you need
            ],
        )

# Pass image through vision stack
file_path = "path/to/your/image.jpg"

# Open the image file
img = Image.open(file_path)
 
# Convert the image to RGB mode
img = img.convert('RGB')
    
# Convert the image to a NumPy array
img_array = np.array(img)

# Pass img_array through vision_stack
vs.run(in_image = image_array, verbose = True)
# With verbose, if ros is running then topics will be created for each layer to visualize processing.
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.
