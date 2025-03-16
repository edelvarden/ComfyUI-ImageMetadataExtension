import json
import os
import re
from datetime import datetime
from pathlib import Path

from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np

import folder_paths
from comfy.cli_args import args

from .base import BaseNode
from ..capture import Capture
from .. import hook
from ..trace import Trace
from ..defs.combo import SAMPLER_SELECTION_METHOD

import piexif
import piexif.helper


# refer. https://github.com/comfyanonymous/ComfyUI/blob/38b7ac6e269e6ecc5bdd6fefdfb2fb1185b09c9d/nodes.py#L1411
class SaveImageWithMetaData(BaseNode):
    OUTPUT_FORMATS = [
        "png", "png_with_json", "jpg", "jpg_with_json", "webp", "webp_with_json"
    ]
    QUALITY_OPTIONS = ["max", "high", "medium", "low"]
    METADATA_OPTIONS = ["full", "default", "workflow_only", "none"]
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the saved file. You can include formatting options like %date:yyyy-MM-dd% or %seed%, and combine them as needed, e.g., %date:hhmmss%_%seed%."}),
                "subdirectory_name": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Custom directory to save the images. Leave empty to use the default output "
                        "directory. You can include formatting options like %date:yyyy-MM-dd%."
                    ),
                }),
                "output_format": (s.OUTPUT_FORMATS, {
                    "tooltip": "The format in which the images will be saved."
                }),
            },
            "optional": {
                "extra_metadata": ("EXTRA_METADATA", {
                    "tooltip": "Additional metadata to be included with the saved image. This can contain key-value pairs for extra information."
                }),
                "quality": (s.QUALITY_OPTIONS, {
                    "tooltip": "Quality levels:"
                            "\n'max' / 'lossless WebP' - 100"
                            "\n'high' - 80"
                            "\n'medium' - 60"
                            "\n'low' - 30 (lower quality, smaller file size)"
                            "\n\nNote: PNG images ignore this setting. "
                }),
                "metadata_scope": (s.METADATA_OPTIONS, {
                    "tooltip": "Choose the metadata to save: "
                            "\n'full' - default + extra metadata, "
                            "\n'default' - same as SaveImage node, "
                            "\n'workflow_only' - workflow metadata only, "
                            "\n'none' - no metadata."
                }),
                "include_batch_num": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include batch numbers in filenames."
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    DESCRIPTION = "Saves the input images with metadata to your ComfyUI output directory."

    pattern_format = re.compile(r"(%[^%]+%)")
    
    def parse_output_format(self, output_format):
        save_workflow_json = output_format.endswith("_with_json")
        base_format = output_format.replace("_with_json", "")
        return base_format, save_workflow_json

    
    def save_images(self, images, filename_prefix="ComfyUI", subdirectory_name="", prompt=None,
                    extra_pnginfo=None, extra_metadata=None, output_format="png", 
                    quality=100, metadata_scope="full", 
                    include_batch_num=True):
        
        if extra_metadata is None:
            extra_metadata = {}
        
        base_format, save_workflow_json = self.parse_output_format(output_format)
        
        pnginfo_dict = self.generate_metadata(extra_metadata) if metadata_scope == "full" else {}

        filename_prefix = self.format_filename(filename_prefix, pnginfo_dict) + self.prefix_append

        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
        )
        
        if subdirectory_name:
            subdirectory_name = self.format_filename(subdirectory_name, pnginfo_dict)
            full_output_folder = os.path.join(self.output_dir, subdirectory_name)
            filename = filename_prefix

        results = list()
        os.makedirs(full_output_folder, exist_ok=True)

        for batch_number, image in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            metadata = self.prepare_pnginfo(pnginfo_dict, batch_number, len(images), prompt, extra_pnginfo, metadata_scope)

            filename_with_batch_num = f"{filename}_{batch_number:05}" if include_batch_num else filename

            file = f"{filename_with_batch_num}.{base_format}"

            if os.path.exists(os.path.join(full_output_folder, file)):
                file = self.find_next_available_filename(full_output_folder, filename_with_batch_num, base_format)

            quality_value = self.get_quality_value(quality)

            if base_format == "webp":
                if quality == 100:
                    img.save(os.path.join(full_output_folder, file), "WEBP", lossless=True)
                else:
                    img.save(os.path.join(full_output_folder, file), "WEBP", quality=quality_value)
            elif base_format == "png":
                img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            else:
                img.save(os.path.join(full_output_folder, file), optimize=True, quality=quality_value)

            if base_format in ["jpg", "webp"]:
                exif_bytes = piexif.dump({
                    "Exif": {
                        piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(Capture.gen_parameters_str(pnginfo_dict), encoding="unicode")
                    }
                })
                piexif.insert(exif_bytes, os.path.join(full_output_folder, file))

            results.append({"filename": file, "subfolder": full_output_folder, "type": self.type})

        # Save the JSON metadata for the whole batch once
        if save_workflow_json and images is not None:
            images_length = len(images)
            if images_length > 0:
                last_batch_number = images_length - 1
                json_filename = f"{filename}_{last_batch_number:05}.json" if include_batch_num else f"{filename}.json"
                batch_json_file = os.path.join(full_output_folder, json_filename)
                with open(batch_json_file, "w", encoding="utf-8") as f:
                    json.dump(extra_pnginfo["workflow"], f)

        return {"ui": {"images": results}}

    def find_next_available_filename(self, full_output_folder, filename, base_format):
        """
        Finds the next available filename by checking existing files in the directory.
        """
        existing_files = {f.stem for f in Path(full_output_folder).glob(f"{filename}_*.{base_format}")}
        counter = 1
        while f"{filename}_{counter}" in existing_files:
            counter += 1
        return f"{filename}_{counter}.{base_format}"

    def get_quality_value(self, quality):
        return {"max": 100, "high": 80, "medium": 60, "low": 30}.get(quality, 100)

    def generate_metadata(self, extra_metadata):
        """
        Merging extra metadata with the base metadata.
        """
        pnginfo_dict = self.gen_pnginfo(SAMPLER_SELECTION_METHOD[0], 0, True)
        pnginfo_dict.update({k: v.replace(",", "/") for k, v in extra_metadata.items() if k and v})
        return pnginfo_dict

    def prepare_pnginfo(self, pnginfo_dict, batch_number, total_images, prompt, extra_pnginfo, metadata_scope):
        """
        Return final PNG metadata with batch information, parameters, and optional prompt details.
        """
        if metadata_scope == "none":
            return None

        metadata = PngInfo()
        pnginfo_copy = pnginfo_dict.copy()

        if total_images > 1:
            pnginfo_copy["Batch index"] = batch_number
            pnginfo_copy["Batch size"] = total_images

        if metadata_scope == "full":
            parameters = Capture.gen_parameters_str(pnginfo_copy)
            metadata.add_text("parameters", parameters)

        if prompt is not None and metadata_scope != "workflow_only":
            metadata.add_text("prompt", json.dumps(prompt))

        if extra_pnginfo and isinstance(extra_pnginfo, dict):
            for k, v in extra_pnginfo.items():
                value_to_add = json.dumps(v) if not isinstance(v, str) else v
                metadata.add_text(k, value_to_add)

        return metadata

    @classmethod
    def gen_pnginfo(cls, sampler_selection_method, sampler_selection_node_id, save_civitai_sampler):
        inputs = Capture.get_inputs()
        trace_tree_from_this_node = Trace.trace(hook.current_save_image_node_id, hook.current_prompt)
        inputs_before_this_node = Trace.filter_inputs_by_trace_tree(inputs, trace_tree_from_this_node)

        sampler_node_id = Trace.find_sampler_node_id(trace_tree_from_this_node, sampler_selection_method, sampler_selection_node_id)
        trace_tree_from_sampler_node = Trace.trace(sampler_node_id, hook.current_prompt)
        inputs_before_sampler_node = Trace.filter_inputs_by_trace_tree(inputs, trace_tree_from_sampler_node)

        return Capture.gen_pnginfo_dict(inputs_before_sampler_node, inputs_before_this_node, save_civitai_sampler)

    @classmethod
    def format_filename(cls, filename, pnginfo_dict):
        """
        Replaces placeholders in the filename with actual values like date, seed, prompt, etc.
        """
        result = re.findall(cls.pattern_format, filename)
        
        now = datetime.now()
        date_table = {
            "yyyy": f"{now.year}",
            "MM": f"{now.month:02d}",
            "dd": f"{now.day:02d}",
            "hh": f"{now.hour:02d}",
            "mm": f"{now.minute:02d}",
            "ss": f"{now.second:02d}",
        }

        for segment in result:
            parts = segment.strip("%").split(":")
            key = parts[0]

            if key == "seed":
                filename = filename.replace(segment, str(pnginfo_dict.get("Seed", "")))

            elif key in {"width", "height"}:
                size = pnginfo_dict.get("Size", "x").split("x")
                value = size[0] if key == "width" else size[1]
                filename = filename.replace(segment, value)

            elif key in {"pprompt", "nprompt"}:
                prompt = pnginfo_dict.get(f"{'Positive' if key == 'pprompt' else 'Negative'} prompt", "").replace("\n", " ")
                length = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
                filename = filename.replace(segment, prompt[:length].strip() if length else prompt.strip())

            elif key == "model":
                model = os.path.splitext(os.path.basename(pnginfo_dict.get("Model", "")))[0]
                length = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
                filename = filename.replace(segment, model[:length] if length else model)

            elif key == "date":
                date_format = parts[1] if len(parts) > 1 else "yyyyMMddhhmmss"
                for k, v in date_table.items():
                    date_format = date_format.replace(k, v)
                filename = filename.replace(segment, date_format)

        return filename


class CreateExtraMetaData(BaseNode):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "key1": ("STRING", {"default": "", "multiline": False}),
                "value1": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "key2": ("STRING", {"default": "", "multiline": False}),
                "value2": ("STRING", {"default": "", "multiline": False}),
                "key3": ("STRING", {"default": "", "multiline": False}),
                "value3": ("STRING", {"default": "", "multiline": False}),
                "key4": ("STRING", {"default": "", "multiline": False}),
                "value4": ("STRING", {"default": "", "multiline": False}),
                "extra_metadata": ("EXTRA_METADATA",),
            },
        }

    RETURN_TYPES = ("EXTRA_METADATA",)

    FUNCTION = "create_extra_metadata"

    def create_extra_metadata(
        self,
        extra_metadata={},
        key1="",
        value1="",
        key2="",
        value2="",
        key3="",
        value3="",
        key4="",
        value4="",
    ):
        extra_metadata.update(
            {
                key1: value1,
                key2: value2,
                key3: value3,
                key4: value4,
            }
        )
        return (extra_metadata,)
