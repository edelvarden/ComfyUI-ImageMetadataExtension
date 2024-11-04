# ComfyUI-ImageMetadataExtension

Custom node for [ComfyUI](https://github.com/comfyanonymous/ComfyUI). It adds additional metadata for saved images, ensuring compatibility with the Civitai website.

This is a fork of [nkchocoai/ComfyUI-SaveImageWithMetaData](https://github.com/nkchocoai/ComfyUI-SaveImageWithMetaData).

**Key differences:**
- Simplified the node by removing unnecessary fields for general use.
- Included metadata for LoRa weights.
- The `subdirectory_name` field allows you to specify a custom name or use mask values to create a subdirectory for saved images. For example, using the mask `%date:yyyy-MM%` ([formatting options](#formatting-options)) will create a directory named with the current year and month (e.g., `2024-10`), organizing your images by the date they were generated.
- The `file_format` field allows you to specify the format for saving images. Supported formats include PNG, JPG, and WebP.
- The `include_extra_metadata` option is set to `true` by default. When enabled, it embeds additional metadata into saved images, including model usage details, which automatically populate the "Resources" field on the Civitai website. When set to `false`, only default metadata is included.

## Installation

### Recommended Installation

Use the [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager) to install.

```
ComfyUI-ImageMetadataExtension
```

### Manual Installation

1. Navigate to the `custom_nodes` directory inside your ComfyUI folder.
2. Clone this repository:

  ```bash
   cd <ComfyUI directory>/custom_nodes
   git clone https://github.com/edelvarden/ComfyUI-ImageMetadataExtension.git
  ```

## Usage

Basic usage looks like ([workflow.json](assets/workflow.json)):

![workflow-preview](assets/Capture1.PNG)

Lora strings are automatically added to the prompt area, allowing the Civitai website to understand the weights you used. Other metadata is also successfully included.

![website-preview](assets/Capture2.PNG)

## Formatting Options
- The `filename_prefix` and `subdirectory_name` support the following options:

| Key             | Information to be Replaced            |
| --------------- | ------------------------------------- |
| %seed%          | Seed value                            |
| %width%         | Image width                           |
| %height%        | Image height                          |
| %pprompt%       | Positive prompt                       |
| %pprompt:[n]%   | First n characters of positive prompt |
| %nprompt%       | Negative prompt                       |
| %nprompt:[n]%   | First n characters of negative prompt |
| %model%         | Checkpoint name                       |
| %model:[n]%     | First n characters of checkpoint name |
| %date%          | Date of generation (yyyyMMddhhmmss)  |
| %date:[format]% | Date of generation                    |

- See the following table for the identifiers specified by `[format]` in `%date:[format]%`:

| Identifier | Description                 |
| ---------- | --------------------------- |
| yyyy       | Year                        |
| yy         | Short year format           |
| MM         | Month                       |
| dd         | Day                         |
| hh         | Hour                        |
| mm         | Minute                      |
| ss         | Second                      |
