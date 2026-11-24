from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig



# GPROTrainer reward functions must have the parameters prompts and completions. There are other required params, but they
# are not necessary for our purposes, so we just put them into kwargs
def anger_reward_func(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    scores: list[float] = []
    
    for prompt, completion in zip(prompts, completions):
        pass    
        # Get some score from the classifier

        # Maybe modify it? Punish explicit anger? Or we could remove explicit anger words before passing into classifier?

        # returns a list of floats, a score for each completion


    return scores

# This will just be a dataset of prompts that we will pass into the llm.
# So will be something like a bunch of story prompts. We can probably keep this relatively small, ~30 maybe?
dataset = {}

grpo_config = GRPOConfig(
    output_dir="./trained_model",

)

lora_config = LoraConfig()



trainer = GRPOTrainer(
    model = "google/gemma-2-2b-it",
    reward_funcs = anger_reward_func,
    train_dataset=dataset,
    peft_config= lora_config
)

