""" Language Model API """

import ast
import json
import logging
import random
import re
import subprocess
import time

import boto3
import tiktoken
from openai import OpenAI

LOGGER = logging.getLogger(__name__)


CONTEXT_LENGTHS = {
    "amazon.titan-tg1-large": 8000,
    "amazon.titan-text-express-v1": 8000,
    "anthropic.claude-instant-v1": 100000,
    "anthropic.claude-v2:1": 200000,
    "anthropic.claude-3-sonnet-20240229-v1:0": 200000,
    "anthropic.claude-3-haiku-20240307-v1:0": 200000,
    "mistral.mistral-large-2402-v1:0": 32000,
    "mistral.mistral-small-2402-v1:0": 32000,
    "meta.llama3-70b-instruct-v1:0": 8000,
}

SUPPORTED_MODELS = [
    "anthropic.claude-instant-v1",
    "anthropic.claude-v2:1",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
    "openai.gpt-3.5-turbo-0613",
    "mistral.mistral-large-2402-v1:0",
    "mistral.mistral-small-2402-v1:0",
    "meta.llama3-70b-instruct-v1:0",
]

BEDROCK_MODELS = [
    "anthropic.claude-instant-v1",
    "anthropic.claude-v2:1",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
    "mistral.mistral-large-2402-v1:0",
    "mistral.mistral-small-2402-v1:0",
    "meta.llama3-70b-instruct-v1:0",
]


class LMClass:
    """LMClass defines a class that encapsulates
    an AI model and allows conversing with it"""

    # Initialize with model name and optional account number
    def __init__(self, model_name, account_number=None):
        assert (
            model_name in SUPPORTED_MODELS
        ), f"{model_name} is not defined and tested"
        # define the LLM parameters
        self.model_name = model_name
        self.temperature = 0.1
        self.top_k = 20
        self.top_p = 0.25
        self.context_length_dict = -1

        self.bedrock_model = 0
        if model_name in BEDROCK_MODELS:
            self.bedrock_model = 1
            # get the AWS account number if not provided
            if account_number is None:
                print("Enter the AWS account number: ")
                self.account_number = int(input())
            else:
                self.account_number = account_number

    # Define callable method to initiate conversation
    def __call__(self, prompt):
        if self.bedrock_model:
            response = call_bedrock_lm(
                model_id=self.get_model_name(),
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                account_number=self.account_number,
                prompt=prompt,
            )
        else:
            response = call_openai_lm(
                model_id=self.get_model_name(),
                temperature=self.temperature,
                prompt=prompt,
            )

        return response

    def get_model_owner(self):
        if "anthropic" in self.model_name:
            return "anthropic"
        elif "amazon" in self.model_name:
            return "amazon"
        elif "openai" in self.model_name:
            return "openai"
        elif "mistral" in self.model_name:
            return "mistral"
        elif "llama" in self.model_name:
            return "meta"

        raise ValueError(f"There is no owner for {self.model_name} model.")

    def get_model_name(self):
        return self.model_name.replace("openai.", "")


def aws_auth(
    account, service_name="bedrock-runtime", region_name="us-east-1"
):
    """Get the AWS client service (default: bedrock) for the account number"""
    aws_cred = json.loads(
        subprocess.check_output(
            f"ada credentials print --account {account}  --provider conduit "
            "--role IibsAdminAccess-DO-NOT-DELETE",
            shell=True,
        )
    )
    aws_service = boto3.client(
        service_name=service_name,
        region_name=region_name,
        aws_access_key_id=aws_cred["AccessKeyId"],
        aws_secret_access_key=aws_cred["SecretAccessKey"],
        aws_session_token=aws_cred["SessionToken"],
    )
    return aws_service


def count_tokens(text: str, model_owner: str, model_name=None):
    """Count the number of tokens in a text"""
    if model_owner in ["amazon", "anthropic", "meta", "mistral"]:
        # 1.3 is just a heuristic
        n_tokens = len(text.split()) * 1.3
    elif model_owner == "openai":
        assert (
            model_name is not None
        ), "You must provide a model_name for OpenAI models."
        n_tokens = count_token_openai(text, model_name)
    else:
        raise ValueError(f"Model {model_owner} is not defined for this code.")
    return n_tokens


def get_context_length(
    *, prompt_in, prompt_out="", model_owner=None, model_name=None
):
    """Get the context length for the input and output prompts"""
    len_context = {}
    len_context["in"] = count_tokens(prompt_in, model_owner, model_name)
    len_context["out"] = count_tokens(prompt_out, model_owner, model_name)
    return len_context


def make_bedrock_body(
    *, model_id, prompt, temperature, top_k, top_p, encode_only=False
):
    """Create the bedrock body"""
    if encode_only:
        # TODO: make sure it works for embedding purpose
        return {}

    else:
        if "amazon" in model_id:
            body = json.dumps(
                {
                    "inputText": prompt,
                    "textGenerationConfig": {
                        "maxTokenCount": 1024,  # TODO: refactor
                        "stopSequences": [],
                        "temperature": temperature,
                        "topP": top_p,
                    },
                }
            )

        elif "anthropic" in model_id:
            num_input_tokens = count_tokens(prompt, model_owner="anthropic")
            # Regardless of the context window, if max_tokens_to_sample is
            # greater than 8191, the service throws an exception.
            max_tokens_to_sample = min(
                CONTEXT_LENGTHS[model_id] - num_input_tokens, 8191
            )

            if "claude-3" in model_id:
                body = json.dumps(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": prompt}],
                            }
                        ],
                        "max_tokens": max_tokens_to_sample,
                        "stop_sequences": [],
                        # "temperature": temperature,  # TODO: Not in # guidelines  # noqa: E501
                        # "top_k": top_k,  # TODO: Not in guidelines
                        # "top_p": top_p,  # TODO: Not in guidelines
                        "anthropic_version": "bedrock-2023-05-31",
                    }
                )
            else:  # Old Versions
                body = json.dumps(
                    {
                        "prompt": prompt,
                        "max_tokens_to_sample": max_tokens_to_sample,
                        "stop_sequences": ["\n\nAssistant:", "\n\nHuman:"],
                        "temperature": temperature,
                        "top_k": top_k,
                        "top_p": top_p,
                        "anthropic_version": "bedrock-2023-05-31",
                    }
                )
        elif "mistral" in model_id:
            body = json.dumps(
                {
                    "prompt": f"<s>[INST]{prompt}[/INST]",
                    "max_tokens": 1024,
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                }
            )
        elif "meta" in model_id:
            body = json.dumps(
                {
                    "prompt": prompt,
                    "max_gen_len": 512,
                    "temperature": temperature,
                    "top_p": top_p,
                }
            )
        else:
            raise ValueError(
                f"Model {model_id} is not defined for this code."
            )

    return body


# invoke bedrock call
def invoke_bedrock(bedrock, model_id, body):
    start = time.time()
    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json",
    )

    end = time.time()

    elapsed_secs = end - start

    return response, elapsed_secs


def invoke_bedrock_streaming(bedrock, model_id, body, retries=1):
    """Invoke the LM model and return the output as chunks"""
    start = time.time()

    attempt = 1

    while True:
        try:
            stream_response = bedrock.invoke_model_with_response_stream(
                modelId=model_id, body=body
            )

            raw_chunks = []

            for i, response in enumerate(stream_response["body"]):
                if "chunk" in response:
                    raw_chunks.append(response["chunk"]["bytes"])
                else:
                    raise Exception(str(response))

            end = time.time()

            elapsed_secs = end - start
            chunks = [
                json.loads(raw_chunk.decode())["completion"]
                for raw_chunk in raw_chunks
            ]

            return chunks, elapsed_secs
        except Exception as e:
            if attempt >= retries:
                raise e
            else:
                backoff = attempt * (5 + int(5 * random.random()))
                LOGGER.warning("Error on attempt %d: %s", attempt, str(e))
                LOGGER.warning("Error occurred using body %s", body)
                LOGGER.warning(
                    "Sleeping for %d seconds before retrying.", backoff
                )
                time.sleep(backoff)
                attempt += 1


def report_request_stats(
    msg: str, lm_response: str, elapsed_secs: float, model_owner: str
):
    """log the number of tokens and statistics on our AWS-service calls"""
    msg = "" if msg is None else msg
    n_tokens = count_tokens(lm_response, model_owner=model_owner)
    tokens_per_sec = n_tokens / elapsed_secs
    if len(msg):
        LOGGER.debug("%s elapsed seconds %f", msg, elapsed_secs)
    else:
        LOGGER.debug("elapsed seconds %f", elapsed_secs)
    LOGGER.debug("%d tokens, %f tokens/sec", n_tokens, tokens_per_sec)


def stream_bedrock_lm_response(
    bedrock: object,
    model_id: str,
    prompt: str,
    temperature: float,
    top_k: int,
    top_p: float,
    retries: int,
    msg: str = None,
):
    """Invoke the LM streaming model and return the output as a string"""
    body = make_bedrock_body(
        model_id=model_id,
        prompt=prompt,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    chunks, elapsed_secs = invoke_bedrock_streaming(
        bedrock, model_id, body, retries=retries
    )
    lm_response = "".join(chunks)
    report_request_stats(
        msg, lm_response, elapsed_secs, model_owner=model_id.split(".")[0]
    )
    return lm_response


def get_bedrock_lm_response(
    bedrock: object,
    model_id: str,
    prompt: str,
    temperature: float,
    top_k: int,
    top_p: float,
    msg: str = None,
    debug: bool = False,
):
    """Invoke the LM model and return the output as a string"""
    body = make_bedrock_body(
        model_id=model_id,
        prompt=prompt,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    lm_output, _ = invoke_bedrock(bedrock, model_id, body)

    if "amazon" in model_id:
        bedrock_response = json.loads(lm_output.get("body").read()).get(
            "results"
        )[0]
        if debug:
            print(f'Stop Reason: {bedrock_response.get("completionReason")}')
        output_text = bedrock_response.get("outputText")

    elif "anthropic" in model_id:
        bedrock_response = json.loads(
            lm_output.get("body").read().decode("utf8")
        )
        if "claude-3" in model_id:
            assert (
                len(bedrock_response["content"]) == 1
            ), "the response has to be 1 item only"
            output_text = bedrock_response["content"][0]["text"]
        else:
            if debug:
                print(f'Stop Reason: {bedrock_response["stop_reason"]}')
            output_text = bedrock_response["completion"]
    elif "mistral" in model_id:
        bedrock_response = json.loads(lm_output.get("body").read())
        assert (
            len(bedrock_response["outputs"]) == 1
        ), "the response has to be 1 item only"
        if debug:
            print(
                f'Stop Reason: {bedrock_response["outputs"][0]["stop_reason"]}'
            )
        output_text = bedrock_response["outputs"][0]["text"]
    elif "meta" in model_id:
        bedrock_response = json.loads(lm_output.get("body").read())
        if debug:
            print(f'Stop Reason: {bedrock_response["stop_reason"]}')
        output_text = bedrock_response["generation"]

    return output_text


def call_bedrock_lm(
    model_id: str,
    temperature: float,
    top_k: int,
    top_p: float,
    prompt: str,
    account_number: int,
):
    """
    Invoke bedrock and get response from the LM model and return the output as
    a string.
    """
    bedrock = aws_auth(account=account_number)
    try:
        lm_response = get_bedrock_lm_response(
            bedrock=bedrock,
            model_id=model_id,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            prompt=prompt,
        )
        return lm_response
    except Exception as e:
        LOGGER.error("Was not able to get LM response: %s", str(e))
        raise e


def call_openai_lm(model_id: str, temperature: float, prompt: str):
    """Connect to OpenAI Client and complete the conversation"""
    try:
        # create the open-ai client
        client = OpenAI()
        # use the completion function to get model response
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        # get the response conent
        lm_response = completion.choices[0].message.content

    except Exception as e:
        LOGGER.error("Was not able to get LM response: %s", str(e))
        raise e

    return lm_response


def stream_bedrock_lm(
    model_id: str,
    temperature: float,
    top_k: int,
    top_p: float,
    retries: int,
    prompt: str,
    account_number: int,
):
    """
    Invoke Bedrock for STREAMING LM output and return the response in a stream.
    """
    bedrock = aws_auth(account=account_number)
    for i in range(retries):
        try:
            lm_response = stream_bedrock_lm_response(
                bedrock=bedrock,
                model_id=model_id,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                retries=retries,
                prompt=prompt,
            )
            return lm_response
        except Exception as e:
            LOGGER.error("Was not able to get LM response: %s", str(e))
            raise e


def get_text_between_tags(text, xml_tag="<answer>"):
    """Get the text between two tags"""
    start_tag_re = re.escape(xml_tag)
    # create end tag
    end_tag = xml_tag.replace("<", "</")
    end_tag_re = re.escape(end_tag)
    # Regex to find text between the tags
    regex = rf"(?<={start_tag_re})(.*?)(?={end_tag_re})"
    # Find all matches
    matches = re.findall(regex, text, flags=re.DOTALL)
    assert len(matches) <= 1, (
        f"There must be 0 or 1 match within {xml_tag} tags. You "
        f"found ({len(matches)}: {matches}"
    )
    if len(matches):
        return matches[0].strip()
    else:
        return text.strip()


def parse_lm_response(result):
    """Parse the LM response into a set of words"""
    if "<answer>" in result:
        # get the string between <answer> and </answer> from "result"
        result_only_answer = get_text_between_tags(result, xml_tag="<answer>")
        # hacky way to remove "set(" & ")" when print is in a wrong format
        if "set(" in result_only_answer:
            result_only_answer = result_only_answer.replace("set(", "")[:-1]
        # check the set pattern and make the Python set
        pattern = r"[a-zA-Z0-9 ]+"
        letters = re.findall(pattern, result_only_answer.replace(" ", ""))
        result_clean = ""
        for letter in letters:
            result_clean += '"' + letter + '",'
        result_clean = "{" + result_clean[:-1] + "}"
        result_obj = try_convert_ints(ast.literal_eval(result_clean))
    else:
        if "set()" in result:
            result_obj = set()
        else:
            result_obj = {-1}

    return result_obj


def try_convert_ints(num_set):
    """Convert integers to integers"""
    converted_set = set()
    for val in num_set:
        try:
            converted_set.add(int(val))
        except ValueError:
            converted_set.add(val)

    # parse the output when it includes the text in the answer
    if converted_set in [{"theemptyset"}, {"emptyset"}, {"Theemptyset"}]:
        converted_set = set()

    return converted_set


def count_token_openai(prompt, model):
    """Return the number of tokens used by a list of messages.
    This script is based on the code written in OpenAI cookbook
    https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # every prompt follows <|start|>{role/name}\n{content}<|end|>\n
        )
    elif "gpt-3.5-turbo" in model:
        print(
            "Warning: gpt-3.5-turbo may update over time. Returning num "
            "tokens assuming gpt-3.5-turbo-0613."
        )
        return count_token_openai(prompt, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print(
            "Warning: gpt-4 may update over time. Returning num tokens "
            "assuming gpt-4-0613."
        )
        return count_token_openai(prompt, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"count_token_openai() is not implemented for model {model}. "
            "See https://github.com/openai/openai-python/blob/main/chatml.md "
            "for information on how messages are converted to tokens."
        )
    num_tokens = 0
    num_tokens += tokens_per_message
    num_tokens += len(encoding.encode(prompt))
    return num_tokens


def count_token_words_openai(prompt, model):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(prompt))
