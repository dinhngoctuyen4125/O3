import os
import re
import json
import argparse
import random
from tqdm import tqdm

import fire
import torch
import transformers
from peft import PeftModel
# from src.peft_model_hacked import PeftModel
from transformers import GenerationConfig, LlamaTokenizer, AutoConfig
from transformers import LlamaForCausalLM
# from src.modeling_llama_hacked import LlamaForCausalLM
import sys

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

import json
import os.path as osp
from typing import Union
import numpy as np

def set_seed(seed: int):
    """Fix PRNG seed for reproducable experiments.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def check_api_usage(generated_line, target_api, alias_dict):
    """Kiểm tra xem API sinh ra có khớp với target hay không bằng Regex."""
    for key, value in alias_dict.items():
        if value == target_api:
            pattern = r'\b' + re.escape(key) + r'\b'
            if re.search(pattern, generated_line):
                return True
    return False

class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        # if not template_name:
        #     # Enforce the default here, so the constructor can be called with '' and will not break.
        #     template_name = "alpaca"
        # file_name = osp.join("templates", f"{template_name}.json")
        # if not osp.exists(file_name):
        #     raise ValueError(f"Can't read {file_name}")
        # with open(file_name) as fp:
        #     self.template = json.load(fp)
        self.template = {
            "description": "Template used by Alpaca-LoRA.",
            "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
            "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
            "response_split": "### Response:"
        }

        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
            self,
            instruction: str,
            input: Union[None, str] = None,
            label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


def main():
    parser = argparse.ArgumentParser(description='Evaluation')

    # Defining arguments
    # ./data/scienceqa_SD_5/scienceqa_biology_test_SD.json
    # ./data/scienceqa_RD_5/scienceqa_not_biology_test_RD.json 85.93%
    # ./data/scienceqa_SD_5/scienceqa_biology_train_SD.json 94
    # ./data/scienceqa_SD_5/scienceqa_biology_test_SD.json 92
    # ./data/scienceqa_SD/scienceqa_biology_test_SD.json
    # "/tank/local/cgo5577/MoLA_f/datasets/CL_biology_test_scienceq_all.json",
    parser.add_argument('--test_dataset', type=str, default="./data/scienceqa_RD_5/scienceqa_not_biology_test_RD.json",
                        help='test_dataset')
    parser.add_argument('--base_model', type=str, default="./llama2_qaall", help='base_model')
    # ./lora_bio_random_test_force_retain ./lora_bio_random_test_k_l
    # lora_bio_random_test_o lora_bio_random_test_oo
    # lora_bio_random_test lora_bio_random_o o: 6.05 2.02
    # ./lora_bio_random_test_force_retain_gemma_1
    parser.add_argument('--lora_weights', type=str, default="./lora_bio_random_test_force_retain_gemma_0.1",
                        help='lora_model')
    parser.add_argument('--max_batch_size', type=int, default=8, help='base_model')
    parser.add_argument('--seed', type=int, default=8, help='base_model')
    # ./SCALE_0.1_seed_0_o_unlearn_lora_force_checkpoints_5/lora_force_random_biology_force
    # lora_bio_random_test_force_no_retain biology_test_SD 9.07% RD 35.63%
    # lora_bio_random_test_force_retain_gemma_1 biology_test_SD 11.08% RD 67.76% llama2_base
    # lora_bio_random_test_force_retain_gemma_2 biology_test_SD 36.27% RD 75.04%
    # lora_bio_random_test_force_retain_gemma_3 biology_test_SD 77.83% RD 81.88%
    # ./SCALE_0.1_seed_0_o_unlearn_lora_force_checkpoints_4_retain/lora_force_random_biology_force acc: 39.8%
    # lora_bio_random_test_force_retain_gemma_1_0.1 biology_test_SD 39.8%
    # lora_bio_random_test_force_retain_gemma_0.1 biology_test_SD 13.85% acc: 60.54%
    # lora_bio_random_test no retain f * 5r RD 50


    # Parsing arguments
    args = parser.parse_args()
    set_seed(args.seed)
    data_a = json.load(open(args.test_dataset, encoding='utf-8'))
    base_model = args.base_model
    lora_weights = args.lora_weights
    max_batch_size = args.max_batch_size

    print(args.test_dataset)
    print(args.base_model)
    print(args.lora_weights)
    load_8bit = False

    path = "/".join(lora_weights.split("/")[:-1])
    if not os.path.exists(path):
        os.mkdir(path)
    result_file = path + "/seed{}_{}_{}".format(str(args.seed), lora_weights.split("/")[-1], args.test_dataset.split("/")[-1])
    print(result_file)

    tokenizer = LlamaTokenizer.from_pretrained(base_model, padding_side='left')
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.bfloat16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.bfloat16,
        )

        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.bfloat16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    print(model.config.pad_token_id, tokenizer.pad_token_id)
    print(model.config.bos_token_id, tokenizer.bos_token_id)
    print(model.config.eos_token_id, tokenizer.eos_token_id)
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    max_new_tokens = 128
    save_every = 200
    MAX_PROMPT_LENGTH = 2048

    num_deprecated = 0
    num_replacement = 0
    num_mismatch = 0
    results = []
    outputs = []

    for start_idx in tqdm(range(0, len(data_a), max_batch_size)):
        end_idx = min(start_idx + max_batch_size, len(data_a))
        batch = data_a[start_idx:end_idx]

        # Tạo prompt cho từng sample
        batch_prompts = []
        batch_meta = []
        for example in batch:
            code_context = example["probing input new"]

            # Cắt ngắn nếu quá dài
            encoded_context = tokenizer.encode(code_context, add_special_tokens=False)
            if len(encoded_context) > MAX_PROMPT_LENGTH - 200:
                encoded_context = encoded_context[-(MAX_PROMPT_LENGTH - 200):]
                code_context = tokenizer.decode(encoded_context)

            prompt_text = (
                f"Complete and output the next line for the following Python function:\n"
                f"```python\n"
                f"{code_context}"
            )
            batch_prompts.append(prompt_text)
            batch_meta.append({
                "code_context": code_context,
                "deprecated_api": example.get("deprecated api", []),
                "replacement_api": example.get("replacement api", ""),
                "alias_dict": example.get("alias dict", {})
            })

        inputs = tokenizer(batch_prompts, padding=True, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_LENGTH)
        input_ids = inputs["input_ids"].to(device)

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )

        # Trích xuất phần text sinh ra (chỉ lấy token mới)
        input_length = input_ids.shape[1]
        generated_tokens = generation_output.sequences[:, input_length:]
        generated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        for idx, generated_text in enumerate(generated_texts):
            meta = batch_meta[idx]

            clean_text = generated_text.replace("```python", "").replace("```", "").strip()
            first_line_generated = clean_text.split('\n')[0]

            # Nối dòng cuối của prompt với code gen ra
            last_line_of_prompt = meta["code_context"].split('\n')[-1]
            full_line_to_check = last_line_of_prompt + first_line_generated

            is_replacement = check_api_usage(full_line_to_check, meta["replacement_api"], meta["alias_dict"])
            is_deprecated = any(check_api_usage(full_line_to_check, api, meta["alias_dict"]) for api in meta["deprecated_api"])


            record = {
                "prompt": meta["code_context"],
                "deprecated_api": meta["deprecated_api"],
                "replacement_api": meta["replacement_api"],
                "generated_content": full_line_to_check
            }

            if is_replacement:
                num_replacement += 1
            elif is_deprecated:
                num_deprecated += 1
            else:
                num_mismatch += 1

            results.append(record)
            outputs.append(full_line_to_check)
            print(f"[dep={is_deprecated}, rep={is_replacement}] {full_line_to_check[:120]}")

        if end_idx % save_every == 0 or end_idx == len(data_a):
            total = len(results)
            print(f"{total}/{len(data_a)}, deprecated: {num_deprecated}, replacement: {num_replacement}, mismatch: {num_mismatch}, saving to {result_file}")
            data = {}
            data['total'] = total
            data['num_deprecated'] = num_deprecated
            data['num_replacement'] = num_replacement
            data['num_mismatch'] = num_mismatch
            data['results'] = results
            with open(result_file, 'w') as f:
                json.dump(data, f, indent=2, separators=(',', ': '))



if __name__ == "__main__":
    main()