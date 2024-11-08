import json
import os

from . import hook
from .defs.captures import CAPTURE_FIELD_LIST
from .defs.meta import MetaField

from nodes import NODE_CLASS_MAPPINGS
from execution import get_input_data
from comfy_execution.graph import DynamicPrompt


class Capture:
    @classmethod
    def get_inputs(cls):
        inputs = {}
        prompt = hook.current_prompt
        extra_data = hook.current_extra_data
        outputs = hook.prompt_executer.caches.outputs

        for node_id, obj in prompt.items():
            class_type = obj["class_type"]
            obj_class = NODE_CLASS_MAPPINGS[class_type]
            node_inputs = prompt[node_id]["inputs"]
            input_data = get_input_data(
                node_inputs, obj_class, node_id, outputs, DynamicPrompt(prompt), extra_data
            )
            for node_class, metas in CAPTURE_FIELD_LIST.items():
                if class_type == node_class:
                    for meta, field_data in metas.items():
                        validate = field_data.get("validate")
                        if validate is not None and not validate(
                            node_id, obj, prompt, extra_data, outputs, input_data
                        ):
                            continue

                        if meta not in inputs:
                            inputs[meta] = []

                        value = field_data.get("value")
                        if value is not None:
                            inputs[meta].append((node_id, value))
                            continue

                        selector = field_data.get("selector")
                        if selector is not None:
                            v = selector(
                                node_id, obj, prompt, extra_data, outputs, input_data
                            )
                            if isinstance(v, list):
                                for x in v:
                                    inputs[meta].append((node_id, x))
                            elif v is not None:
                                inputs[meta].append((node_id, v))
                            continue

                        field_name = field_data["field_name"]
                        value = input_data[0].get(field_name)
                        if value is not None:
                            format_func = field_data.get("format")
                            v = value
                            if isinstance(value, list) and len(value) > 0:
                                v = value[0]
                            if format_func is not None:
                                v = format_func(v, input_data)
                            if isinstance(v, list):
                                for x in v:
                                    inputs[meta].append((node_id, x))
                            else:
                                inputs[meta].append((node_id, v))
        return inputs

    @classmethod
    def get_lora_strings_and_hashes(cls, inputs_before_sampler_node):
        lora_names = inputs_before_sampler_node.get(MetaField.LORA_MODEL_NAME, [])
        lora_weights = inputs_before_sampler_node.get(MetaField.LORA_STRENGTH_MODEL, [])
        lora_hashes = inputs_before_sampler_node.get(MetaField.LORA_MODEL_HASH, [])

        lora_strings = []
        lora_hashes_string = []

        for name, weight, hash in zip(lora_names, lora_weights, lora_hashes):
            if name and weight and hash:
                lora_name_clean = os.path.splitext(os.path.basename(name[1]))[0].replace(' ', '_')

                # Use it for both the prompt and the hash
                lora_strings.append(f"<lora:{lora_name_clean}:{weight[1]}>")
                lora_hashes_string.append(f"{lora_name_clean}: {hash[1]}")

        return lora_strings, ", ".join(lora_hashes_string)


    @classmethod
    def gen_pnginfo_dict(cls, inputs_before_sampler_node, inputs_before_this_node, save_civitai_sampler=True):
        pnginfo_dict = {}

        def update_pnginfo_dict(inputs, metafield, key):
            x = inputs.get(metafield, [])
            if x:
                value = x[0][1]
                
                # Only add non-empty values for other fields
                if value is not None and value != "":
                    pnginfo_dict[key] = value
                    return value  # Return the value that was set
            
            return None  # Return None if no value was set or value is empty

        
        positive_prompt = ""
        positive_prompt += update_pnginfo_dict(inputs_before_sampler_node, MetaField.POSITIVE_PROMPT, "Positive prompt")

        # Get Lora strings and hashes
        lora_strings, lora_hashes_string = cls.get_lora_strings_and_hashes(inputs_before_sampler_node)

        # Append Lora models to the positive prompt, which is required for the Civitai website to parse and apply Lora weights.
        # Format: <lora:Lora_Model_Name:weight_value>. Example: <lora:Lora_Name_00:0.6> <lora:Lora_Name_01:0.8>
        if lora_strings:
            positive_prompt += " " + " ".join(lora_strings)

        pnginfo_dict["Positive prompt"] = positive_prompt.strip()
        update_pnginfo_dict(inputs_before_sampler_node, MetaField.NEGATIVE_PROMPT, "Negative prompt")

        update_pnginfo_dict(inputs_before_sampler_node, MetaField.STEPS, "Steps")

        sampler_names = inputs_before_sampler_node.get(MetaField.SAMPLER_NAME, [])
        schedulers = inputs_before_sampler_node.get(MetaField.SCHEDULER, [])

        if save_civitai_sampler:
            pnginfo_dict["Sampler"] = cls.get_sampler_for_civitai(sampler_names, schedulers)
        else:
            if sampler_names:
                pnginfo_dict["Sampler"] = sampler_names[0][1]
                if schedulers:
                    scheduler = schedulers[0][1]
                    if scheduler != "normal":
                        pnginfo_dict["Sampler"] += "_" + scheduler

        update_pnginfo_dict(inputs_before_sampler_node, MetaField.CFG, "CFG scale")
        update_pnginfo_dict(inputs_before_sampler_node, MetaField.SEED, "Seed")
        update_pnginfo_dict(inputs_before_sampler_node, MetaField.CLIP_SKIP, "Clip skip")

        image_widths = inputs_before_sampler_node.get(MetaField.IMAGE_WIDTH, [])
        image_heights = inputs_before_sampler_node.get(MetaField.IMAGE_HEIGHT, [])
        if image_widths and image_heights:
            pnginfo_dict["Size"] = f"{image_widths[0][1]}x{image_heights[0][1]}"

        update_pnginfo_dict(inputs_before_sampler_node, MetaField.MODEL_NAME, "Model")
        update_pnginfo_dict(inputs_before_sampler_node, MetaField.MODEL_HASH, "Model hash")
        update_pnginfo_dict(inputs_before_this_node, MetaField.VAE_NAME, "VAE")
        update_pnginfo_dict(inputs_before_this_node, MetaField.VAE_HASH, "VAE hash")

        # Add Lora hashes, based on https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/82a973c04367123ae98bd9abdf80d9eda9b910e2/extensions-builtin/Lora/scripts/lora_script.py#L78
        if lora_hashes_string:
            pnginfo_dict["Lora hashes"] = f'"{lora_hashes_string}"'

        pnginfo_dict.update(cls.gen_loras(inputs_before_sampler_node))
        pnginfo_dict.update(cls.gen_embeddings(inputs_before_sampler_node))

        hashes_for_civitai = cls.get_hashes_for_civitai(inputs_before_sampler_node, inputs_before_this_node)
        if len(hashes_for_civitai) > 0:
            pnginfo_dict["Hashes"] = json.dumps(hashes_for_civitai)

        return pnginfo_dict



    @classmethod
    def extract_model_info(cls, inputs, meta_field_name, prefix):
        model_info_dict = {}
        model_names = inputs.get(meta_field_name, [])
        model_hashes = inputs.get(f"{meta_field_name}_HASH", [])

        for index, (model_name, model_hash) in enumerate(zip(model_names, model_hashes)):
            field_prefix = f"{prefix}_{index}"
            model_info_dict[f"{field_prefix} name"] = os.path.splitext(os.path.basename(model_name[1]))[0]
            model_info_dict[f"{field_prefix} hash"] = model_hash[1]

        return model_info_dict

    @classmethod
    def gen_loras(cls, inputs):
        return cls.extract_model_info(inputs, MetaField.LORA_MODEL_NAME, "Lora")

    @classmethod
    def gen_embeddings(cls, inputs):
        return cls.extract_model_info(inputs, MetaField.EMBEDDING_NAME, "Embedding")

    @classmethod
    def gen_parameters_str(cls, pnginfo_dict):
        result = pnginfo_dict.get("Positive prompt", "") + "\n"

        negative_prompt = pnginfo_dict.get("Negative prompt", "")
        if negative_prompt:
            result += "Negative prompt: " + negative_prompt + "\n"

        s_list = []
        pnginfo_dict_without_prompt = {
            k: v
            for k, v in pnginfo_dict.items()
            if k not in {"Positive prompt", "Negative prompt"}
        }
        for k, v in pnginfo_dict_without_prompt.items():
            s = str(v).strip().replace("\n", " ")
            s_list.append(f"{k}: {s}")

        return result + ", ".join(s_list)

    @classmethod
    def get_hashes_for_civitai(cls, inputs_before_sampler_node, inputs_before_this_node):
        resource_hashes = {}
        model_hashes = inputs_before_sampler_node.get(MetaField.MODEL_HASH, [])
        if model_hashes:
            resource_hashes["model"] = model_hashes[0][1]

        vae_hashes = inputs_before_this_node.get(MetaField.VAE_HASH, [])
        if vae_hashes:
            resource_hashes["vae"] = vae_hashes[0][1]

        lora_model_names = inputs_before_sampler_node.get(MetaField.LORA_MODEL_NAME, [])
        lora_model_hashes = inputs_before_sampler_node.get(MetaField.LORA_MODEL_HASH, [])
        for lora_model_name, lora_model_hash in zip(lora_model_names, lora_model_hashes):
            lora_model_name = os.path.splitext(os.path.basename(lora_model_name[1]))[0]
            resource_hashes[f"lora:{lora_model_name}"] = lora_model_hash[1]

        embedding_names = inputs_before_sampler_node.get(MetaField.EMBEDDING_NAME, [])
        embedding_hashes = inputs_before_sampler_node.get(MetaField.EMBEDDING_HASH, [])
        for embedding_name, embedding_hash in zip(embedding_names, embedding_hashes):
            embedding_name = os.path.splitext(os.path.basename(embedding_name[1]))[0]
            resource_hashes[f"embed:{embedding_name}"] = embedding_hash[1]

        return resource_hashes

    @classmethod
    def get_sampler_for_civitai(cls, sampler_names, schedulers):
        """
        Get the pretty sampler name for Civitai in the form of `<Sampler Name> <Scheduler name>`.
            - `dpmpp_2m` and `karras` will return `DPM++ 2M Karras`
        
        If there is a matching sampler name but no matching scheduler name, return only the matching sampler name.
            - `dpmpp_2m` and `exponential` will return only `DPM++ 2M`

        if there is no matching sampler and scheduler name, return `<sampler_name>_<scheduler_name>`
            - `ipndm` and `normal` will return `ipndm`
            - `ipndm` and `karras` will return `ipndm_karras`

        Reference: https://github.com/civitai/civitai/blob/main/src/server/common/constants.ts
        
        Last update: https://github.com/civitai/civitai/blob/a2e6d267eefe6f44811a640c570739bcb078e4a5/src/server/common/constants.ts#L138-L165
        """

        def sampler_with_karras_exponential(sampler, scheduler):
            match scheduler:
                case "karras":
                    sampler += " Karras"
                case "exponential":
                    sampler += " Exponential"
            return sampler
        
        def sampler_with_karras(sampler, scheduler):
            if scheduler == "karras":
                return sampler + " Karras"
            return sampler

        if len(sampler_names) > 0:
            sampler = sampler_names[0][1]
        if len(schedulers) > 0:
            scheduler = schedulers[0][1]

        match sampler:
            case "euler" | "euler_cfg_pp":
                return "Euler"
            case "euler_ancestral" | "euler_ancestral_cfg_pp":
                return "Euler a"
            case "heun" | "heunpp2":
                return "Huen"
            case "dpm_2":
                return sampler_with_karras("DPM2", scheduler)
            case "dpm_2_ancestral":
                return sampler_with_karras("DPM2 a", scheduler)
            case "lms":
                return sampler_with_karras("LMS", scheduler)
            case "dpm_fast":
                return "DPM fast"
            case "dpm_adaptive":
                return "DPM adaptive"
            case "dpmpp_2s_ancestral":
                return sampler_with_karras("DPM++ 2S a", scheduler)
            case "dpmpp_sde" | "dpmpp_sde_gpu":
                return sampler_with_karras("DPM++ SDE", scheduler)
            case "dpmpp_2m":
                return sampler_with_karras("DPM++ 2M", scheduler)
            case "dpmpp_2m_sde" | "dpmpp_2m_sde_gpu":
                return sampler_with_karras("DPM++ 2M SDE", scheduler)
            case "dpmpp_3m_sde" | "dpmpp_3m_sde_gpu":
                return sampler_with_karras_exponential("DPM++ 3M SDE", scheduler)
            case "lcm":
                return "LCM"
            case "ddim":
                return "DDIM"
            case "uni_pc" | "uni_pc_bh2":
                return "UniPC"
            
        if scheduler == "normal":
            return sampler
        return sampler + "_" + scheduler
        