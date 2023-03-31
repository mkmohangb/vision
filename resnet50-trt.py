import numpy as np
import os
from PIL import Image
import sys
import tensorrt as trt
import torch

class ModelData(object):
    MODEL_PATH = 'ResNet50.onnx'
    INPUT_SHAPE = (3, 224, 224)
    DTYPE = trt.float32

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
def GiB(val):
    return val * 1 << 30

def build_engine_from_onnx(model_file):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    config.max_workspace_size = GiB(1)

    with open(model_file, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse ONNX file")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    return builder.build_engine(network, config)

def normalize_image(image):
    c, h, w = ModelData.INPUT_SHAPE
    image_arr = (
        np.asarray(image.resize((w,h), Image.ANTIALIAS))
        .transpose([2,0,1])
        .astype(trt.nptype(ModelData.DTYPE))
        .ravel()
    )
    return (image_arr/254.0 - 0.45) / 0.225

def execute(engine, context, img):
    dtypeMapping = {
        trt.bool: torch.bool,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.float16: torch.float16,
        trt.float32: torch.float32
    }
    bindings = []
    outputs = []
    device = torch.device('cuda:0')
    stream = torch.cuda.Stream(device=device)
    input = torch.asarray(img, device=device)
    bindings.append(input.contiguous().data_ptr())
    for binding in engine:
        if not engine.binding_is_input(binding):
            print("output binding is ", dtypeMapping[engine.get_binding_dtype(binding)])
            output = torch.empty(size=tuple(engine.get_binding_shape(binding)),
                                 dtype=dtypeMapping[engine.get_binding_dtype(binding)],
                                 device=device)
            outputs.append(output)
            bindings.append(output.data_ptr())

    context.execute_async_v2(bindings, stream.cuda_stream)
    stream.synchronize()
    return [outputs[0].cpu().detach().numpy()]

def main():
    #test_case = "tabby_tiger_cat.jpg"
    test_case = sys.argv[1]
    onnx_model_file = ModelData.MODEL_PATH
    labels = open("class_labels.txt", "r").read().split('\n')

    engine = build_engine_from_onnx(onnx_model_file)
    context = engine.create_execution_context()
    image = normalize_image(Image.open(test_case))
    trt_outputs = execute(engine, context, image)

    pred = labels[np.argmax(trt_outputs[0])]
    if "_".join(pred.split()) in os.path.splitext(os.path.basename(test_case))[0]:
        print("Correctly recognized " + test_case + " as " + pred)
    else:
        print("Incorrectly recognized " + test_case + " as " + pred)


if __name__ == "__main__":
    main()