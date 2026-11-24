# pip install accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
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

    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    prompt_length = inputs["input_ids"].shape[1]

    responses = []
    for _ in range(num_responses):
        print(_)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=500,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
            )
        response = tokenizer.decode(
            outputs[0][prompt_length:], skip_special_tokens=True
        ).strip()
        responses.append(response)

    df_prompts_responses = pd.DataFrame({"prompts": [prompt] * num_responses, "responses": responses})
    df_prompts_responses.to_csv(filename, index=False)

gen_resp_to_csv("Generate a 200 word story", 100, "stories.csv")
