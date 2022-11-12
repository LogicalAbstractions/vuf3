from pathlib import Path

import onnx
from onnxsim import simplify

from models.classification.classification_module import ClassificationModule
from models.model_configuration import ModelConfiguration


def export_classification_model_to_onnx(onnx_path: Path,
                                        model: ClassificationModule,
                                        model_configuration: ModelConfiguration) -> None:
    print("Exporting to onnx: " + str(onnx_path))
    model.export_onnx(model_configuration, onnx_path)

    simplified_onnx_path = onnx_path.with_suffix(".simplified.onnx")

    print("Simplifying onnx model to: " + str(simplified_onnx_path))

    onnx_model = onnx.load(str(onnx_path))
    model_simp, check = simplify(onnx_model)

    if not check:
        print("Model could not be simplified")

    onnx.save(model_simp, str(simplified_onnx_path))
