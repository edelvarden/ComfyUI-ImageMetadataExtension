#https://github.com/Light-x02/ComfyUI-FluxSettingsNode
from ..meta import MetaField
from ..formatters import calc_model_hash, calc_lora_hash, convert_skip_clip



CAPTURE_FIELD_LIST = {
    "WeiLinComfyUIPromptToLoras":
        {
            MetaField.POSITIVE_PROMPT: {"field_name": "positive"},
            MetaField.NEGATIVE_PROMPT: {"field_name": "negative"},
        },
}