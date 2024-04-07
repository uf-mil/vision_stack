import torch
import onnx
from onnx import helper
import tensorflow as tf
from torchvision import transforms
import onnx_tf
from ..yolov7.models.experimental import attempt_load
from PIL import Image
import os
import shutil

def convert_pt_to_tflite(weights_path, output_path, sample_image_path):
    """
    Input:
        weights_path : string (e.g. path/to/weights.pt)
        output_path : string (e.g. path/to/tflite/weights.tflite)
        sample_image_path : string (e.g. path/to/sample/image.png)
    Output:
        A .tflite weights file that can be processed.
    """
    # Load the PyTorch ResNet50 model
    pytorch_model = attempt_load(weights_path)
    pytorch_model.eval()

    # Export the PyTorch model to ONNX format
    image_path = sample_image_path
    input_data = Image.open(image_path).convert('RGB')
    input_data = input_data.resize((960, 608))
    img_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dummy_input = img_transform(input_data).unsqueeze(0)

    # print(dummy_input)
    onnx_model_path = 'temp.onnx'
    torch.onnx.export(pytorch_model, dummy_input, onnx_model_path, verbose=False, opset_version=12)

    # Load the ONNX model
    onnx_model = onnx.load(onnx_model_path)

    # Define a mapping from old names to new names
    name_map = {"input.1": "input_1"}

    # Initialize a list to hold the new inputs
    new_inputs = []

    # Iterate over the inputs and change their names if needed
    for inp in onnx_model.graph.input:
        if inp.name in name_map:
            # Create a new ValueInfoProto with the new name
            new_inp = helper.make_tensor_value_info(name_map[inp.name],
                                                    inp.type.tensor_type.elem_type,
                                                    [dim.dim_value for dim in inp.type.tensor_type.shape.dim])
            new_inputs.append(new_inp)
        else:
            new_inputs.append(inp)

    # Clear the old inputs and add the new ones
    onnx_model.graph.ClearField("input")
    onnx_model.graph.input.extend(new_inputs)

    # Go through all nodes in the model and replace the old input name with the new one
    for node in onnx_model.graph.node:
        for i, input_name in enumerate(node.input):
            if input_name in name_map:
                node.input[i] = name_map[input_name]

    # Remove ONNX model file
    try:
        os.remove(onnx_model_path)
    except OSError as e:
        print(f"Could not delete {onnx_model_path}\nERROR:{e}")

    # Convert the ONNX model to TensorFlow format
    tf_model_path = 'temp.pb'
    tf_rep = onnx_tf.backend.prepare(onnx_model)
    tf_rep.export_graph(tf_model_path)

    # Convert the TensorFlow model to TensorFlow Lite format
    converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(tf_model_path)
    tflite_model = converter.convert()

    try:
        shutil.rmtree(tf_model_path)
    except OSError as e:
        print(f"Could not delete {tf_model_path}\nERROR:{e}")

    # Save the TensorFlow Lite model to a file
    with open(output_path, 'wb') as f:
        f.write(tflite_model)