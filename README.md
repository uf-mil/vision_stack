# Vision Stack
Vision Stack is a ros package that makes processing image data intuitive and straight forward. By implementing an array-like data structure to stack processing for feature abstraction and analysis for data extraction, Vision Stack provides all the tools necessary for tackling any vision tasks.

Author: Daniel Parra

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
git submodule add https://github.com/uf-mil/vision_stack.git path/to/submodule
```

## Usage

```python
from PIL import Image
import numpy as np
from vision_stack import VisionStack, UnderwaterImageEnhancementLayer, ResizeLayer

# Initialize vision stack data structure
vs = VisionStack(
            layers=[
                ResizeLayer(960, 608),
                UnderWaterImageEnhancementLayer(),
                # Include as many layers as you need
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
