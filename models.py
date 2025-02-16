import os

os.environ["HF_HOME"] = ""
# Set the env variable to use the hf_token
os.environ["HF_TOKEN"] = ""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


def get_model_and_tokenizer(model_name, device="auto", max_memory=None):
    if model_name == "Llama-2-7b-chat-hf":
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            device_map=device,
        )
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        tokenizer.model_max_length = 4096
    elif model_name == "zephyr-7b-beta":
        if not max_memory:
            model = AutoModelForCausalLM.from_pretrained(
                "HuggingFaceH4/zephyr-7b-beta",
                device_map=device,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                "HuggingFaceH4/zephyr-7b-beta",
                device_map=device,
                max_memory=max_memory
            )
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
        tokenizer.model_max_length = 8192
    elif model_name == "Mistral-7B-Instruct-v0.2":
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
            device_map=device,
        )
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        tokenizer.model_max_length = 4096
    elif model_name == "Llama-2-13b-chat-hf":
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-13b-chat-hf",
            torch_dtype="auto",
            device_map=device,
        )
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
        tokenizer.model_max_length = 4096
    elif model_name == "Mixtral-8x7B-Instruct-v0.1":
        model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            load_in_4bit=True,
            device_map=device,
        )
        tokenizer.model_max_length = 32768
    elif model_name == "gemma-7b-it":
        tokenizer = AutoTokenizer.from_pretrained(
            "google/gemma-7b-it",
        )
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-7b-it",
            device_map=device,
            torch_dtype=torch.float16,
        )
        tokenizer.model_max_length = 8192
    elif model_name == "vicuna-13b-v1.5":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            "lmsys/vicuna-13b-v1.5",
            quantization_config=bnb_config,
            device_map=device,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "lmsys/vicuna-13b-v1.5",
        )
    elif model_name == "Orca-2-13b":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Orca-2-13b",
            quantization_config=bnb_config,
            device_map=device,
        )
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Orca-2-13b")
        tokenizer.chat_template = AutoTokenizer.from_pretrained(
            "HuggingFaceH4/zephyr-7b-beta"
        ).chat_template

    elif model_name == "dolphin-2.9-llama3-8b":
        model = AutoModelForCausalLM.from_pretrained(
            "cognitivecomputations/dolphin-2.9-llama3-8b", device_map=device,)
        tokenizer = AutoTokenizer.from_pretrained("cognitivecomputations/dolphin-2.9-llama3-8b")
        tokenizer.model_max_length = 8192
    elif model_name == "Mistral-7B-Instruct-v0.3":
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.3", device_map=device,token="", max_memory=max_memory)
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
        tokenizer.model_max_length = 8192

    elif model_name == "Phi-3-small-8k-instruct":
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-small-8k-instruct", device_map=device,trust_remote_code=True,torch_dtype="auto")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-small-8k-instruct",trust_remote_code=True)
        tokenizer.model_max_length = 8192

    elif model_name == "Phi-3-mini-4k-instruct":
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct", device_map=device,torch_dtype="auto",)
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        tokenizer.model_max_length = 4096

    elif model_name == "Phi-3-medium-4k-instruct":
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-medium-4k-instruct", device_map=device,torch_dtype="auto",)
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-medium-4k-instruct")
        tokenizer.model_max_length = 4096

    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    print("+++++ Model Max Length", tokenizer.model_max_length)
    return model, tokenizer
