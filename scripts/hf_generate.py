from pathlib import Path
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from litgpt.utils import CLI


@torch.no_grad()
def run_generation(
    checkpoint_dir: Path,
    prompt: str = "What food do llamas eat?",
    finetuned_path: Path = None,
    max_new_tokens: int = 100,
    top_k: int = 50,
    top_p: float = 1.0,
    temperature: float = 0.0,
    use_chat_template: bool = False,
    precision: str = "32-true",
):

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, model_max_length=2048, padding_side="left")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    if finetuned_path is not None:
        state_dict = torch.load(finetuned_path, weights_only=False)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir,
            state_dict=state_dict,
            torch_dtype={"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision],
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint_dir).to(device)

    def apply_conv_template(input_text, tokenizer):
        message = [
            {"role": "user", "content": input_text},
        ]
        return tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

    if use_chat_template:
        prompt = apply_conv_template(prompt, tokenizer)

    # input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=not use_chat_template)["input_ids"].to(
    #     model.device
    # )
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"].to(model.device)
    # if last token is eos, remove it
    if input_ids[0][-1] == tokenizer.eos_token_id:
        input_ids[0] = input_ids[0][:-1]

    print(input_ids)

    # hstates = model(input_ids)
    # breakpoint()

    response = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        do_sample=True if temperature > 0.0 else False,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    response_text = tokenizer.decode(response[0], skip_special_tokens=False)

    print(response_text)


if __name__ == "__main__":
    CLI(run_generation)
