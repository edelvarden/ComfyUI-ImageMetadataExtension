import json
import os
import re
from datetime import datetime

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
    OUTPUT_FORMATS = ["png", "png_with_json", "jpg", "jpg_with_json", "webp", "webp_with_json"]
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
                "metadata_scope": (s.METADATA_OPTIONS, {
                    "tooltip": "Choose the metadata to save: "
                            "\n'full' - default + extra metadata, "
                            "\n'default' - same as SaveImage node, "
                            "\n'workflow_only' - workflow metadata only, "
                            "\n'none' - no metadata."
                }),
                "include_batch_num": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include batch numbers in the filenames."
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
    
    def save_images(self, images, filename_prefix="ComfyUI", subdirectory_name="", prompt=None,
        extra_pnginfo=None, extra_metadata={}, output_format="png", lossless_webp=True,
        quality=100, metadata_scope="full", include_batch_num=True):
        
        # Parse file format and determine if JSON metadata should be saved
        base_format, save_workflow_json = self.parse_output_format(output_format)
        
        # Set metadata based on selection
        pnginfo_dict = self.generate_metadata(extra_metadata) if metadata_scope == "full" else {}

        # Format filename prefix
        filename_prefix = self.format_filename(filename_prefix, pnginfo_dict) + self.prefix_append

        # Determine output folder
        if subdirectory_name:
            subdirectory_name = self.format_filename(subdirectory_name, pnginfo_dict)
            full_output_folder = os.path.join(self.output_dir, subdirectory_name)
            filename = filename_prefix
        else:
            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
                filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
            )

        # Ensure the output folder exists
        os.makedirs(full_output_folder, exist_ok=True)

        results = []

        for batch_number, image in enumerate(images):
            img_array = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

            # Prepare metadata for PNG format
            metadata = self.prepare_pnginfo(pnginfo_dict, batch_number, len(images), prompt, extra_pnginfo, metadata_scope)

            # Add batch number if the parameter is set
            if include_batch_num:
                filename_with_batch_num = f"{filename}_{batch_number:05}"
            else:
                filename_with_batch_num = filename

            # Regex pattern to extract counters from filenames like "filename_00001_1.png"
            counter_pattern = re.compile(rf"{re.escape(filename_with_batch_num)}_(\d+)\.{base_format}")

            file = f"{filename_with_batch_num}.{base_format}"

            # Check if file exists and find the highest counter used
            if os.path.exists(os.path.join(full_output_folder, file)):
                counter = 1
                # Scan for files with the same base name and find the highest counter
                existing_files = os.listdir(full_output_folder)
                max_counter = 0
                for existing_file in existing_files:
                    match = counter_pattern.match(existing_file)
                    if match:
                        max_counter = max(max_counter, int(match.group(1)))
                counter = max_counter + 1  # Start the counter from the highest number + 1

                file = f"{filename_with_batch_num}_{counter}.{base_format}"

            # Save image
            if base_format == "png":
                # Save PNG format
                img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            else:
                # Save other formats (JPEG, WEBP)
                parameters = Capture.gen_parameters_str(pnginfo_dict)
                img.save(
                    os.path.join(full_output_folder, file),
                    optimize=True,
                    quality=quality,
                    lossless=lossless_webp,
                )
                # Save EXIF data for JPEG/WEBP
                exif_bytes = piexif.dump({
                    "Exif": {
                        piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(parameters, encoding="unicode")
                    }
                })
                piexif.insert(exif_bytes, os.path.join(full_output_folder, file))

            results.append({
                "filename": file,
                "subfolder": full_output_folder,
                "type": self.type
            })

            # Conditionally save JSON metadata
            if save_workflow_json:
                self.save_json_metadata(extra_pnginfo["workflow"], os.path.join(full_output_folder, file))

        return {"ui": {"images": results}}

    
    def parse_output_format(self, output_format):
        """
        Parse the file format to extract the base format and determine if JSON metadata should be saved.
        """
        save_workflow_json = "json" in output_format
        base_format = output_format.split("_")[0] 
        return base_format, save_workflow_json
    
    def save_json_metadata(self, metadata, file_path):
        """
        Saves metadata as a JSON file with the same name as the image.
        """
        json_file = f"{os.path.splitext(file_path)[0]}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f)


    def generate_metadata(self, extra_metadata):
        """
        Generates PNG metadata by merging extra metadata with the base metadata.
        """
        pnginfo_dict = self.gen_pnginfo(SAMPLER_SELECTION_METHOD[0], 0, True)
        pnginfo_dict.update({k: v.replace(",", "/") for k, v in extra_metadata.items() if k and v})
        return pnginfo_dict

    def prepare_pnginfo(self, pnginfo_dict, batch_number, total_images, prompt, extra_pnginfo, metadata_scope):
        """
        Prepares PNG metadata with batch information, parameters, and optional prompt details.
        """
        if metadata_scope == "none":
            return
        
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
                metadata.add_text(k, json.dumps(v))

        return metadata


    @classmethod
    def gen_pnginfo(cls, sampler_selection_method, sampler_selection_node_id, save_civitai_sampler):
        # Retrieve all node inputs
        inputs = Capture.get_inputs()

        # Trace the inputs leading up to this node
        trace_tree_from_this_node = Trace.trace(hook.current_save_image_node_id, hook.current_prompt)
        inputs_before_this_node = Trace.filter_inputs_by_trace_tree(inputs, trace_tree_from_this_node)

        # Locate the sampler node
        sampler_node_id = Trace.find_sampler_node_id(trace_tree_from_this_node, sampler_selection_method, sampler_selection_node_id)

        # Trace inputs leading up to the sampler node
        trace_tree_from_sampler_node = Trace.trace(sampler_node_id, hook.current_prompt)
        inputs_before_sampler_node = Trace.filter_inputs_by_trace_tree(inputs, trace_tree_from_sampler_node)

        # Generate metadata from inputs
        return Capture.gen_pnginfo_dict(inputs_before_sampler_node, inputs_before_this_node, save_civitai_sampler)

    @classmethod
    def format_filename(cls, filename, pnginfo_dict):
        """
        Replaces placeholders in the filename with actual values like date, seed, prompt, etc.
        """
        result = re.findall(cls.pattern_format, filename)
        
        now = datetime.now()
        date_table = {
            "yyyy": str(now.year),
            "yy": str(now.year)[-2:],  # Get the last two digits of the year
            "MM": str(now.month).zfill(2),
            "dd": str(now.day).zfill(2),
            "hh": str(now.hour).zfill(2),
            "mm": str(now.minute).zfill(2),
            "ss": str(now.second).zfill(2),
        }

        for segment in result:
            parts = segment.replace("%", "").split(":")
            key = parts[0]

            if key == "seed":
                filename = filename.replace(segment, str(pnginfo_dict.get("Seed", "")))

            elif key == "width":
                width = pnginfo_dict.get("Size", "x").split("x")[0]
                filename = filename.replace(segment, str(width))

            elif key == "height":
                height = pnginfo_dict.get("Size", "x").split("x")[1]
                filename = filename.replace(segment, str(height))

            elif key == "pprompt":
                prompt = pnginfo_dict.get("Positive prompt", "").replace("\n", " ")
                if len(parts) >= 2:
                    prompt = prompt[:int(parts[1])]
                filename = filename.replace(segment, prompt.strip())

            elif key == "nprompt":
                prompt = pnginfo_dict.get("Negative prompt", "").replace("\n", " ")
                if len(parts) >= 2:
                    prompt = prompt[:int(parts[1])]
                filename = filename.replace(segment, prompt.strip())

            elif key == "model":
                model = os.path.splitext(os.path.basename(pnginfo_dict.get("Model", "")))[0]
                if len(parts) >= 2:
                    model = model[:int(parts[1])]
                filename = filename.replace(segment, model)

            elif key == "date":
                date_format = parts[1] if len(parts) >= 2 else "yyyyMMddhhmmss"
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
