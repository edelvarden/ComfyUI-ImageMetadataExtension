# LoraLoaderWithPreviews - https://github.com/X-T-E-R/ComfyUI-EasyCivitai-XTNodes
from ..meta import MetaField
from ..formatters import calc_model_hash, calc_lora_hash, convert_skip_clip


SAMPLERS = {
    "LoraLoaderWithPreviews": {
    },
}


CAPTURE_FIELD_LIST = {
"LoraLoaderWithPreviews": {
  MetaField.LORA_MODEL_NAME: {"field_name": "model_name"},
  MetaField.LORA_MODEL_HASH: {
    "field_name": "model_name",
    "format": calc_lora_hash,
  },
  MetaField.LORA_STRENGTH_MODEL: {"field_name": "strength_model"},
  MetaField.LORA_STRENGTH_CLIP: {"field_name": "strength_clip"},
},
}