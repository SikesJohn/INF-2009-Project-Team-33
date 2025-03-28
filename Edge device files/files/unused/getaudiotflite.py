

import torch
from speechbrain.pretrained import SpeakerRecognition

# Load the pre-trained speaker recognition model
classifier = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp_spkrec"
)

print(classifier.mods)

# Export to ONNX
dummy_input = torch.randn(1, 16000)  # Example input tensor

print(len(dummy_input))

torch.onnx.export(
    classifier,               # The model
    dummy_input,              # A dummy input tensor (audio waveform)
    "ecapa_tdnn.onnx",        # Output ONNX file name
    input_names=["input"],    # The name of the input
    output_names=["output"],  # The name of the output
    opset_version=11,         # ONNX opset version
    do_constant_folding=True  # Whether to apply constant folding optimizations
)
