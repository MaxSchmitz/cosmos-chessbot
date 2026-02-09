Cosmos-Reason2: Physical AI Common Sense and Embodied Reasoning Models
Cosmos | Code

Model Overview
Description:
NVIDIA Cosmos Reason 2 is an open, customizable, 8B-parameter reasoning vision language model (VLM) for physical AI and robotics that enables robots and vision AI agents to reason like humans, using prior knowledge, physics understanding and common sense to understand and act in the real world. This model understands space, time, and fundamental physics, and can serve as a planning model to reason what steps an embodied agent might take next.

New features with Cosmos Reason 2:

Enhanced physical AI reasoning with improved spatio-temporal understanding and timestamp precision.
Supports object detection with 2D/3D point localization and bounding box coordinates with reasoning explanations and labels.
Improved long-context understanding up to 256K input tokens.
Use cases:

Video analytics AI agents — Extract valuable insights and perform root-cause analysis on massive volumes of video data. These agents can be used to analyze and understand recorded or live video streams across city and industrial operations. Jumpstart your development of video analytics AI agents by using the NVIDIA Blueprint for video search and summarization (VSS) with Cosmos Reason as the VLM.
Data curation and annotation — Enable developers to automate high-quality curation and annotation of massive, diverse training datasets. Experience NVIDIA Cosmos Curator, powered by Cosmos Reason, a framework that enables developers to quickly filter, annotate, and deduplicate large amounts of sensor data necessary for physical AI development.
Robot planning and reasoning — Act as the brain for deliberate, methodical decision-making in a robot vision language action (VLA) model. Now robots such as humanoids and autonomous vehicles (AV) can interpret environments and complex commands, break them down into tasks and execute them using common sense, even in unfamiliar environments. Explore the NVIDIA Isaac GR00T-Dreams blueprint, which generates vast amounts of synthetic trajectory data using NVIDIA Cosmos world foundation models.
Explore the Cosmos Cookbook, a technical guide that delivers end-to-end workflows, implementation recipes, and detailed examples for building, fine-tuning, and deploying Cosmos Reason in production-ready environments.

The model is ready for commercial use.

Model Developer: NVIDIA

Model Versions
The Cosmos-Reason2 includes the following model:

Cosmos-Reason2-2B: Given a text prompt and an input video, think and generate the answer with respect to the input text prompt and video.
Cosmos-Reason2-8B: Given a text prompt and an input video, think and generate the answer with respect to the input text prompt and video.
License:
This model is released under the NVIDIA Open Model License. Additional Information: Apache License 2.0.

For a custom license, please contact cosmos-license@nvidia.com.

Under the NVIDIA Open Model License, NVIDIA confirms:

Models are commercially usable.
You are free to create and distribute Derivative Models.
NVIDIA does not claim ownership to any outputs generated using the Models or Derivative Models.
Important Note: If You bypass, disable, reduce the efficacy of, or circumvent any technical limitation, safety guardrail or associated safety guardrail hyperparameter, encryption, security, digital rights management, or authentication mechanism (collectively “Guardrail”) contained in the Model without a substantially similar Guardrail appropriate for your use case, your rights under this Agreement NVIDIA Open Model License Agreement will automatically terminate.

Deployment Geography:
Global

Use Case:
Physical AI: Space, time, fundamental physics understanding and embodied reasoning, encompassing robotics, and autonomous vehicles (AV).

Release Date:
Github: 12/19/2025
Huggingface: 12/19/2025.
Model Architecture:
Architecture Type: A Multi-modal LLM consists of a Vision Transformer (ViT) for vision encoder and a Dense Transformer model for LLM. Network Architecture: Qwen3-VL-8B-Instruct.

Cosmos-Reason2-8B is post-trained based on Qwen3-VL-8B-Instruct and follows the same model architecture.

Number of model parameters:

Cosmos-Reason2-8B: 8,767,123,696

Input
Input Type(s): Text+Video/Image

Input Format(s):

Text: String

Video: mp4

Image: jpg

Input Parameters:

Text: One-dimensional (1D)

Video: Three-dimensional (3D)

Image: Two-dimensional (2D)

Other Properties Related to Input:

Use FPS=4 for input video to match the training setup.

Append Answer the question in the following format: <think>\nyour reasoning\n</think>\n\n<answer>\nyour answer\n</answer>. in the system prompt to encourage long chain-of-thought reasoning response.

Output
Output Type(s): Text

Output Format: String

Output Parameters: Text: One-dimensional (1D)

Other Properties Related to Output:

Recommend using 4096 or more output max tokens to avoid truncation of long chain-of-thought response.
Our AI model recognizes timestamps added at the bottom of each frame for accurate temporal localization.
Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA’s hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions. <br>
Software Integration
Runtime Engine(s):

Transformers
Supported Hardware Microarchitecture Compatibility:

NVIDIA Blackwell
NVIDIA Hopper
Note: We have only tested doing inference with BF16 precision.

Operating System(s):

Linux (We have not tested on other operating systems.)
The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.

Usage
See Cosmos-Reason2 for details.

Post Training: Cosmos-Reason2 provides examples of supervised fine-tuning and reinforcement learning on embodied reasoning datasets.
Training and Evaluation Sections:
Cosmos-Reason2-8B model was trained and evaluated on the same datasets used for Cosmos-Reason1-7B, in addition to the following newly added datasets.

Training Datasets:
Data Collection Method:

EgoExo4D: Hybrid: Automatic/Sensors
PerceptionTest: Hybrid: Automatic/Sensors
Language Table: Hybrid: Automatic/Sensors
IntPhys: Hybrid: Automatic/Sensors
InfLevel: Hybrid: Automatic/Sensors
CLEVRER: Hybrid: Automatic/Sensors
Labeling Method:

EgoExo4D: Hybrid: Human,Automated
PerceptionTest: Hybrid: Human,Automated
Language Table: Hybrid: Human,Automated
IntPhys: Hybrid: Hybrid: Human,Automated
InfLevel: Hybrid: Hybrid: Human,Automated
CLEVRER: Hybrid: Hybrid: Human,Automated
The combined datasets span multimodal video, sensor signals, and structured physical-reasoning tasks, providing broad coverage for training world-model reasoning capabilities.

Evaluation Datasets:
Data Collection Method:

EgoExo4D: Hybrid: Automatic/Sensors
PerceptionTest: Hybrid: Automatic/Sensors
Language Table: Hybrid: Automatic/Sensors
IntPhys: Hybrid: Automatic/Sensors
InfLevel: Hybrid: Automatic/Sensors
CLEVRER: Hybrid: Automatic/Sensors
Labeling Method:

EgoExo4D: Hybrid: Human,Automated
PerceptionTest: Hybrid: Human,Automated
Language Table: Hybrid: Human,Automated
IntPhys: Hybrid: Hybrid: Human,Automated
InfLevel: Hybrid: Hybrid: Human,Automated
CLEVRER: Hybrid: Hybrid: Human,Automated
The combined datasets span multimodal video, sensor signals, and structured physical-reasoning tasks, providing broad coverage for training world-model reasoning capabilities.

Dataset Format
Modality: Video (mp4) and Text

Inference:
Test Hardware: H100, A100

We suggest using fps=4 for the input video and max_tokens=4096 to avoid truncated response.

import transformers
import torch

model_name = "nvidia/Cosmos-Reason2-2B"
model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
    model_name, dtype=torch.float16, device_map="auto", attn_implementation="sdpa"
)
processor: transformers.Qwen3VLProcessor = (
    transformers.AutoProcessor.from_pretrained(model_name)
)

video_messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}],
    },
    {"role": "user", "content": [
            {
                "type": "video", 
                "video": "file:///path/to/your/video.mp4",
                "fps": 4,
            },
            {"type": "text", "text": (
                    "Is it safe to turn right? Answer the question using the following format:\n\n<think>\nYour reasoning.\n</think>\n\nWrite your final answer immediately after the </think> tag."
                )
            },
        ]
    },
]

# Process inputs
inputs = processor.apply_chat_template(
    video_messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
    fps=4,
)
inputs = inputs.to(model.device)

# Run inference
generated_ids = model.generate(**inputs, max_new_tokens=4096)
generated_ids_trimmed = [
    out_ids[len(in_ids) :]
    for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
]
output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)

System Requirements and Performance
This model requires a minimum of 32 GB of GPU memory. Inference latency for a single generation across different NVIDIA GPU platforms will be published shortly.

Quality Benchmarks
For comparative evaluation, we present benchmark scores using the Physical AI Bench Leaderboard.

PAI Bench Leaderboard

Ethical Considerations
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

Users are responsible for model inputs and outputs. Users are responsible for ensuring safe integration of this model, including implementing guardrails as well as other safety mechanisms, prior to deployment.

For more detailed information on ethical considerations for this model, please see the subcards of Explainability, Bias, Safety & Security, and Privacy below.

Please report security vulnerabilities or NVIDIA AI Concerns here.

Plus Plus (++) Promise
We value you, the datasets, the diversity they represent, and what we have been entrusted with. This model and its associated data have been:

Verified to comply with current applicable disclosure laws, regulations, and industry standards.
Verified to comply with applicable privacy labeling requirements.
Annotated to describe the collector/source (NVIDIA or a third-party).
Characterized for technical limitations.
Reviewed to ensure proper disclosure is accessible to, maintained for, and in compliance with NVIDIA data subjects and their requests.
Reviewed before release.
Tagged for known restrictions and potential safety implications.
Bias
Field	Response
Participation considerations from adversely impacted groups protected classes in model design and testing:	None
Measures taken to mitigate against unwanted bias:	The training video sources contain multiple physical embodiments and environments including human, car, single arm robot, bimanual robot in indoor and outdoor environments. By training on numerous and various physical interactions and curated datasets, we strive to provide a model that mitigates biases towards certain embodiments or environments.
Explainability
Field	Response
Intended Application & Domain:	Physical AI Reasoning
Model Type:	Transformer
Intended Users:	Physical AI developers
Output:	Text
Describe how the model works:	Given a video/image and a text prompt, the model first converts the video/image into tokens using a vision encoder and a special translator called a projector. These video tokens are combined with the text prompt and fed into the core model, which uses a mix of LLM modules and techniques. This enables the model to think step-by-step and provide detailed, logical responses.
Technical Limitations:	The model may not follow the video or text input accurately in challenging cases, where the input video shows complex scene composition and temporal dynamics. Examples of challenging scenes include: fast camera movements, overlapping human-object interactions, low lighting with high motion blur, and multiple people performing different actions simultaneously.
Verified to have met prescribed NVIDIA quality standards:	Yes
Performance Metrics:	Quantitative and Qualitative Evaluation. Cosmos-Reason2 proposes the embodied reasoning benchmark and physical common sense benchmark to evaluate accuracy with visual question answering.
Potential Known Risks:	The model's output can generate all forms of texts, including what may be considered toxic, offensive, or indecent.
Licensing:	NVIDIA Open Model License. Additional Information: Apache License 2.0.
Privacy
Field	Response
Generatable or reverse engineerable personal data?	No
Personal data used to create this model?	None Known
Was consent obtained for any personal data used?	None Known
How often is dataset reviewed?	Before Release
Is there provenance for all datasets used in training?	Yes
Does data labeling (annotation, metadata) comply with privacy laws?	Yes
Applicable Privacy Policy	NVIDIA Privacy Policy
Safety
Field	Response
Model Application(s):	Physical AI common sense understanding and embodied reasoning
Describe the life critical impact (if present).	None Known
Use Case Restrictions:	NVIDIA Open Model License. Additional Information: Apache License 2.0.
Model and dataset restrictions:	The Principle of least privilege (PoLP) is applied limiting access for dataset generation and model development. Restrictions enforce dataset access during training, and dataset license constraints adhered to. Model checkpoints are made available on Hugging Face, and may become available on cloud providers' model catalog.
