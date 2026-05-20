import os 
import tiktoken
import torch
import argparse
import re
from model import GPT
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description = "Start a chat with a finetuned model of GPT2 or the pretrained one.")

    #general inference arguments
    parser.add_argument("--ft", action = "store_true", help = "Use the latest finetuned model if present.")
    parser.add_argument("--temperature", type = float, default = 1.0, help = "Temperature used to generate the next token.")

    return parser.parse_args()

def terminal_chat(model, device = "cuda", temperature = 1.0):
    model.to(device)
    enc = tiktoken.get_encoding("gpt2")
    print("="*50)
    print("Welcome to your GPT2 LoRA finetuned model - (type 'exit' or 'q' to exit)")
    print("="*50)

    while True:
        input_txt = input("\nYou: ")

        if input_txt.lower() in ['exit', 'q']:
            print("Goodbye")
            break
        if not input_txt.strip():
            continue

        prompt = torch.tensor(enc.encode(input_txt), dtype = torch.long).unsqueeze(0).to(device)
        amp_dtype = torch.float16 if device == "cuda" else torch.bfloat16

        with torch.no_grad():
            # with torch.amp.autocast(device_type = device, dtype = amp_dtype):
                generated_text = ""
                print("\nAssistant: ")
                for t in model.generate(idx = prompt, max_new_tokens = 100, temperature = temperature):
                    word = enc.decode([t])
                    generated_text += word
                    print(word, end = "", flush = True)

                    if any(generated_text.endswith(seq) for seq in ["\n"]):
                        break

def get_latest(dir: str, prefix="model_ft_"):
    """
    Scans the specified directory and returns the Path to the most recent 
    model file matching the 'prefix_YYYYMMDD_HHMMSS' format.
    
    Returns None if no matching files are found.
    """
    
    directory = Path(dir)
    timestamp_pattern = re.compile(rf"^{prefix}\d{{8}}_\d{{6}}")
    matching_files = []

    for file in directory.iterdir():
        if file.is_file() and timestamp_pattern.match(file.name):
            matching_files.append(file)
    if not matching_files:
        return None
    
    latest_model = max(matching_files, key = lambda f: f.name)
    return latest_model


def main():
    args = parse_args()

    if args.ft:
        latest_model = get_latest("./models/")
        if not latest_model:
            print("A finetuned model wasn't found so the pretrained will be used.")
            model = GPT.from_pretrained()
        else:
            pass
            #implement the logic to fuse lora model
    else:
        model = GPT.from_pretrained()
    

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    terminal_chat(model = model, device = device, temperature = args.temperature)
            


if __name__ == "__main__":
    main()

