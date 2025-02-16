from constants import HUMAN_RETRIES_PER_TURN
import json
import requests
from time import sleep
from requests.structures import CaseInsensitiveDict
import pandas as pd
import torch
import re
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "<endpoint>"
)


def call_model(
    prompt,
    endpoint="",
    retry=HUMAN_RETRIES_PER_TURN,
    dv3_token_file_path=None,
):
    response = None
    apiKey = ""
    leo_request = {"url": endpoint, "timeout": 500, "verify": True}

    while retry > 0:
        try:
            headers = CaseInsensitiveDict()
            headers["Content-Type"] = "application/json"
            headers["InferenceApiKey"] = apiKey
            endpoint = "" # Put endpoint here
            leo_request["url"] = endpoint
            leo_request["headers"] = headers
            leo_request["json"] = prompt
            response = requests.post(**leo_request)
            response.raise_for_status()
            response = json.loads(response.content.decode("utf-8"))
            # print(response)
            # break
            response_text = response["choices"][0]["text"].rstrip()
            return response_text
        except Exception as e:
            print(
                f"Exception in call_model with exception {e} and response {response}. "
                "Please ensure you have sufficient access & you are logged in Azure CLI using `az login`."
            )
            sleep(30)
            retry -= 1
    raise Exception("call_model failed for all retries")


def get_payload(
    prompt,
    max_tokens=300,
    temperature=0.1,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop=["<|im_end|>"],
    model="gpt-4o",
):
    return {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "stop": stop,
        "model": model,
    }


def load_human_prompt(human_prompt_name):
    with open(human_prompt_name, "r") as file:
        return file.read()


def replace_data_humanbot(human_prompt, datarow, conv_str, num_turn):
    pmpt = human_prompt.format(
        ConversationReplace = conv_str,
        UserPersona = datarow["UserPersona"],
        UserTask = datarow["UserTask"],
        Seed = datarow["Seed"],
        DetailedPolicy = datarow["DetailedPolicy"],
    )
    #print("++++++ Num Turn:", num_turn)
    #print("++++++ Prompt:\n", pmpt)
    return pmpt


def add_message(conversation, author, text, agent="USER"):
    if author == "USER":
        message = {"role": "user", "content": text}
    else:
        message = {"role": "assistant", "content": text}
    conversation.append(message)


def get_conv_string(conversation, agent="USER"):
    role_map = {"user": agent, "assistant": "AI"}
    conversation = [message for message in conversation if message["role"] != "system"]

    if len(conversation) == 0:
        return "<AI>Hi! How can I help you today?</AI>"
    return "\n".join(
        [
            f"<{role_map[message['role']]}>{message['content']}</{role_map[message['role']]}>"
            for message in conversation
        ]
    )


def evaluate(conversation, eval_prompt):
    eval_conv_str = format_conv_for_eval(conversation)
    formatted_prompt = eval_prompt.format(conv_str=eval_conv_str)
    payload = get_payload(
        formatted_prompt,
        temperature=0,
        max_tokens=200,
        model="gpt-4o",
    )
    response = call_model(payload)
    # print("++++ Full Response",response)
    if "</Q2>" not in response:
        completed_prompt = formatted_prompt + response
        payload = get_payload(
            completed_prompt,
            temperature=0,
            max_tokens=150,
            model="gpt4o",
        )
        response2 = call_model(payload)
        response += response2
    response = response.split("<|im_end|>")[0].strip()

    # Regex function to extract the score between <Q1> and </Q1>
    score = re.search(r"<Q2>(.*?)</Q2>", response).group(1)

    return (
        score,
        response.replace("\n", "\\n").replace("\t", "\\t"),
        eval_conv_str.replace("\n", "\\n").replace("\t", "\\t"),
    )


def evaluate_aaai(conversation, eval_chain, client):
    eval_conv_str = format_conv_for_eval(conversation)
    final_chain = eval_chain + [{"role": "user", "content": eval_conv_str}]
    # print(final_chain)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=final_chain,
        max_tokens=200,
        temperature=0.0,
    )
    response_text = response.choices[0].message.content

    # Regex function to extract the score between <Q2> and </Q2>
    score = re.search(r"<Q2>(.*?)</Q2>", response_text).group(1)

    return (
        score,
        response_text.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r"),
        eval_conv_str.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r"),
    )


def reduce_conv_turns(converstion, is_user=True):
    # Given the conversation list of dictionaries, reduce the conversation to either system message + last 2 user messages
    # or first 2 messages + last 2 user messages if the first message is not a system message
    if is_user:
        last_n = -4
    else:
        last_n = -3
    if converstion[0]["role"] == "system":
        return converstion[:1] + converstion[last_n:]
    else:
        return converstion[:2] + converstion[last_n:]
    return converstion


def format_conv_for_eval(converstion):
    if converstion[0]["role"] == "system":
        valid_conv = converstion[1:]
    else:
        valid_conv = converstion[2:]

    role_map = {"user": "USER", "assistant": "AI"}
    return "".join(
        [
            f"<{role_map[message['role']]}>{message['content']}</{role_map[message['role']]}>"
            for message in valid_conv
        ]
    )


def human(
    conversation,
    user_model,
    user_tokenizer,
    human_prompt,
    datarow,
    num_turn,
    agent="USER",
):
    conv_str = get_conv_string(conversation)
    prompt = replace_data_humanbot(human_prompt, datarow, conv_str, num_turn)
    print(prompt)
    # Tokenize the prompt
    model_inputs = user_tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    if len(model_inputs[0]) > user_tokenizer.model_max_length - 1000:
        print(model_inputs.shape)
        reduced_conv = reduce_conv_turns(conversation, is_user=True)
        conv_str = get_conv_string(reduced_conv)
        prompt = replace_data_humanbot(human_prompt, datarow, conv_str)

    generated_ids = user_model.generate(
        model_inputs,
        max_new_tokens=300,
        do_sample=True,
        pad_token_id=user_tokenizer.eos_token_id,
        temperature=0.15,
        top_p=0.8,
        repetition_penalty=1.25,
    )

    response = user_tokenizer.decode(
        generated_ids[0][len(model_inputs[0]) :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    message_string = "<USER>"
    if message_string not in response:
        # print(response)
        response = response.split("<ASSISTANT>")[0]
        response = response.split("<AI>")[0]
        response = response.rstrip()
        new_prompt = prompt + response + "\n" + message_string
        model_inputs = user_tokenizer.encode(new_prompt, return_tensors="pt").to("cuda")
        generated_ids = user_model.generate(
            model_inputs,
            max_new_tokens=300,
            do_sample=True,
            pad_token_id=user_tokenizer.eos_token_id,
            temperature=0.1,
            top_p=1.0,
            repetition_penalty=1.25,
        )
        print("+++ Initial Response\n", response)
        response = user_tokenizer.decode(
            generated_ids[0][len(model_inputs[0]) :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        print("++ Full Response\n", response)
    else:
        print("++ Full Response\n", response)

    start_tag = f"<{agent}>"
    end_tag = f"</{agent}>"

    response = response.replace(f"{start_tag}{end_tag}", "")
    start_index = response.find(start_tag)

    # if start_index != -1 and end_index != -1:
    #     response = response[start_index + len(start_tag) : end_index]
    if start_index != -1:
        response = response[start_index + len(start_tag) :]
    end_index = response.find(end_tag)
    if end_index != -1:
        response = response[:end_index]
    find_backtick = response.find("```")
    if find_backtick != -1:
        response = response[:find_backtick]
    find_ai_response = response.find("<AI>")
    if find_ai_response != -1:
        response = response[:find_ai_response]
    find_generate_acc_response = response.find("Generate according to")
    if find_generate_acc_response != -1:
        response = response[:find_generate_acc_response]
    user_starter = response.find("User:")
    if user_starter != -1:
        response = response[user_starter:]

    assistant_tag = response.find("<|assistant|>")
    if assistant_tag != -1:
        response = response[:assistant_tag]

    response.replace("```", "")
    response.replace("</USER", "")
    response = response.split("\n")[0]
    response = response.strip()
    # If the response starts and ends with a ", remove them
    if response.startswith('"') and response.endswith('"'):
        response = response[1:-1]

    return response


def converse(
    conversation,
    model,
    tokenizer,
    user_model,
    user_tokenizer,
    datarow,
    human_prompt,
    max_turns=3,
    device="cuda",
):
    num_turn = 0
    print("Going here1")
    human_response = None
    while (num_turn < max_turns) and (human_response != "STOP"):
        human_response = human(
            conversation, user_model, user_tokenizer, human_prompt, datarow, num_turn
        )
        add_message(conversation, "USER", human_response)
        print(f"Human: {human_response}")
        if isinstance(model, str):
            print("Going here")
            bot_response = chatbot_gpt(conversation, model, tokenizer)
        else:
            bot_response = chatbot(conversation, model, tokenizer, device)
        add_message(conversation, "AI", bot_response)
        print(f"Bot: {bot_response}")

        num_turn += 1
    return conversation

def converse_seeds(
        conversation,
    model,
    tokenizer,
    datarow,
    device="cuda",
):
    human_response = None
    human_response = datarow["topic"]
    add_message(conversation, "USER", human_response)
    print(f"Human: {human_response}")
    if isinstance(model, str):
        bot_response = chatbot_gpt(conversation, model, tokenizer)
    else:
        bot_response = chatbot(conversation, model, tokenizer, device)
    add_message(conversation, "AI", bot_response)
    print(f"Bot: {bot_response}")
    return conversation

def chatbot_gpt(conversation, model, client):
    response = client.chat.completions.create(
        model=model,
        messages=conversation,
        max_tokens=400,
        temperature=0.25,
        top_p=0.8,
    )
    return response.choices[0].message.content


def chatbot(conversation, model, tokenizer, device="cuda"):
    encodeds = tokenizer.apply_chat_template(
        conversation, return_tensors="pt", add_generation_prompt=True
    )
    # print(tokenizer.decode(encodeds[0]))
    model_inputs = encodeds.to(device)
    if len(model_inputs[0]) > tokenizer.model_max_length - 1000:
        print(model_inputs.shape)
        reduced_conv = reduce_conv_turns(conversation, is_user=False)
        encodeds = tokenizer.apply_chat_template(
            reduced_conv, return_tensors="pt", add_generation_prompt=True
        )
        model_inputs = encodeds.to(device)

    generated_ids = model.generate(
        model_inputs,
        max_new_tokens=400,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    decoded = tokenizer.decode(
        generated_ids[0][len(encodeds[0]) :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    decoded = decoded.replace("<|assistant|>", "").strip()
    decoded = decoded.replace("**Assistant:**", "").strip()
    decoded = decoded.replace("Assistant:", "").strip()
    decoded = decoded.replace("**Response:**", "").strip()
    decoded = decoded.replace("AI:", "").strip()
    return decoded


def skip_conversation_if_already_done(line, conversation_file):
    try:
        conv_df = pd.read_csv(conversation_file, sep="\t", header=None)
    except pd.errors.EmptyDataError:
        return False
    if line["index"] in conv_df[0].tolist():
        return True
    return False


def init_conversation(model_name, dataset_name):
    manifest = ""
    if dataset_name.startswith("van"):
        manifest = "Vanilla"
        # return []
    system_msg_mapping = {
        "Vanilla": """You are a helpful assistant."""
        }
    if model_name in [
        "Mistral-7B-Instruct-v0.2",
        "Mixtral-8x7B-Instruct-v0.1",
        "gemma-7b-it",
        "Mistral-7B-Instruct-v0.3",
    ]:
        return [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": system_msg_mapping[manifest]},
        ]
    else:
        return [{"role": "system", "content": system_msg_mapping[manifest]}]
