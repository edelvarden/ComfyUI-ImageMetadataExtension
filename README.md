# ComfyUI-ImageMetadataExtension

Custom node for [ComfyUI](https://github.com/comfyanonymous/ComfyUI). It adds additional metadata for saved images, ensuring compatibility with the Civitai website.

This is a fork of [nkchocoai/ComfyUI-SaveImageWithMetaData](https://github.com/nkchocoai/ComfyUI-SaveImageWithMetaData).

Key differences:
- Simplified the node by removing unnecessary fields for general use.
- Included metadata for Lora weights.
- The `save_prompt` option is set to `true` by default. When set to `false`, positive and negative prompts are excluded from the additional metadata. Lora weights will not be included in the prompt, but Lora sources will still be listed. Set it to `false` if you don't want to share your prompt.

## Installation

Clone this repository into the ComfyUI `custom_nodes` directory.

```
cd <ComfyUI directory>/custom_nodes
git clone https://github.com/edelvarden/ComfyUI-ImageMetadataExtension.git
```

Basic usage looks like ([workflow.json](assets/workflow.json)):

![workflow-preview](assets/Capture1.PNG)

Lora strings are automatically added to the prompt area, allowing the Civitai website to understand the weights you used. Other metadata is also successfully included.

![website-preview](assets/Capture2.PNG)
