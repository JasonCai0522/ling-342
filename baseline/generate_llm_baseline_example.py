# pip install accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# input_text = "Write me a poem about Machine Learning."
# input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

# outputs = model.generate(**input_ids, max_new_tokens=32)
# print(tokenizer.decode(outputs[0]))

# https://huggingface.co/google/gemma-2-2b-it

# function that generates a lot of responses to one prompt and saves them to a csv file with columbs: prompts, responses

def gen_resp_to_csv(prompt: str, num_responses: int, filename: str):
    import pandas as pd

    responses = []
    for _ in range(num_responses):
        print(_)


        input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**input_ids, 
                                    max_new_tokens=500,
                                    do_sample=True,
                                    temperature=0.8,
                                    top_p=0.95,)
        response = tokenizer.decode(outputs[0], skip_special_tokens = True)
        responses.append(response)

    df_prompts_responses = pd.DataFrame({"prompts": [prompt] * num_responses, "responses": responses})
    df_prompts_responses.to_csv(filename, index=False)

gen_resp_to_csv("Generate a 200 word story", 3, "test.csv")