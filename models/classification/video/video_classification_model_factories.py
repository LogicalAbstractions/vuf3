from typing import Dict, List

from torch.nn import Module
from torchvision.models.video import r3d_18, mvit_v2_s, s3d, r2plus1d_18
from pytorchvideo import models
from torchvision.models.video.mvit import MSBlockConfig, _mvit

from models.classification.classification_model_factories import register_classification_model
from models.classification.classification_module import ClassificationModule
from models.classification.video.video_classification_model_configuration import VideoClassificationModelConfiguration
from models.classification.video.video_classification_model_provider import VideoClassificationModelProvider, \
    VideoClassificationModelFactory

from utilities.configuration.configuration_reader import ConfigurationReader


def register_video_classification_model(id: str, factory: VideoClassificationModelFactory):
    register_classification_model(VideoClassificationModelProvider(id, factory))


def create_mvit2(configuration_reader: ConfigurationReader, model_configuration: VideoClassificationModelConfiguration,
                 num_classes: int) -> Module:
    config: Dict[str, List] = {
        "num_heads": [1, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8],
        "input_channels": [96, 96, 192, 192, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 768],
        "output_channels": [96, 192, 192, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 768, 768],
        "kernel_q": [
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
        ],
        "kernel_kv": [
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
        ],
        "stride_q": [
            [1, 1, 1],
            [1, 2, 2],
            [1, 1, 1],
            [1, 2, 2],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 2, 2],
            [1, 1, 1],
        ],
        "stride_kv": [
            [1, 8, 8],
            [1, 4, 4],
            [1, 4, 4],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 1, 1],
            [1, 1, 1],
        ],
    }

    block_setting = []
    for i in range(len(config["num_heads"])):
        block_setting.append(
            MSBlockConfig(
                num_heads=config["num_heads"][i],
                input_channels=config["input_channels"][i],
                output_channels=config["output_channels"][i],
                kernel_q=config["kernel_q"][i],
                kernel_kv=config["kernel_kv"][i],
                stride_q=config["stride_q"][i],
                stride_kv=config["stride_kv"][i],
            )
        )

    return _mvit(
        spatial_size=(model_configuration.clip.width, model_configuration.clip.width),
        temporal_size=model_configuration.clip.get_frame_count(),
        block_setting=block_setting,
        residual_pool=True,
        residual_with_cls_embed=False,
        rel_pos_embed=True,
        proj_after_attn=True,
        stochastic_depth_prob=0.2,
        weights=None,
        progress=False,
        num_classes=num_classes
    )


def create_r3d_18(configuration_reader: ConfigurationReader,
                  model_configuration: VideoClassificationModelConfiguration,
                  num_classes: int) -> Module:
    return r3d_18(weights=None, num_classes=num_classes)


def create_s3d(configuration_reader: ConfigurationReader, model_configuration: VideoClassificationModelConfiguration,
               num_classes: int) -> Module:
    return s3d(weights=None, num_classes=num_classes)


def create_mvit(configuration_reader: ConfigurationReader, model_configuration: VideoClassificationModelConfiguration,
                num_classes: int) -> Module:
    return models.create_multiscale_vision_transformers(
        spatial_size=(model_configuration.clip.height, model_configuration.clip.width),
        temporal_size=model_configuration.clip.get_frame_count(), head_num_classes=num_classes)


def create_csn(configuration_reader: ConfigurationReader, model_configuration: VideoClassificationModelConfiguration,
               num_classes: int) -> Module:
    return models.create_csn(model_num_class=num_classes)


def create_resnet(configuration_reader: ConfigurationReader, model_configuration: VideoClassificationModelConfiguration,
                  num_classes: int) -> Module:
    return models.create_resnet(model_num_class=num_classes)


# Multipathway bug
def create_slowfast(configuration_reader: ConfigurationReader,
                    model_configuration: VideoClassificationModelConfiguration,
                    num_classes: int):
    return models.create_slowfast(model_num_class=num_classes)


def create_r2dplus(configuration_reader: ConfigurationReader,
                   model_configuration: VideoClassificationModelConfiguration, num_classes: int) -> Module:
    return r2plus1d_18(weights=None, num_classes=num_classes)


def register_all_video_classification_models():
    register_video_classification_model("r3d_18", create_r3d_18)
    register_video_classification_model("mvit", create_mvit)
    register_video_classification_model("mvit2", create_mvit2)
    register_video_classification_model("s3d", create_s3d)
    register_video_classification_model("r2d_plus", create_r2dplus)
    register_video_classification_model("slowfast", create_slowfast)
    register_video_classification_model("resnet", create_resnet)
    register_video_classification_model("csn", create_csn)
