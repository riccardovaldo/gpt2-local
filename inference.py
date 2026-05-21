import os 
import tiktoken
import torch
import argparse
from model import GPT
from lora import LoRA
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description = "Start a chat with a finetuned model of GPT2 or the pretrained one.")

    #general inference arguments
    parser.add_argument("--checkpoint", type = str, default = None, help = "Provide the path to a checkpoint file for either full ft or lora adapters .pt")
    parser.add_argument("--temperature", type = float, default = 1.0, help = "Temperature used to generate the next token.")

    return parser.parse_args()

def terminal_chat(model, device = "cuda", temperature = 1.0):
    model.to(device).eval()
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

        generated_tokens = []
        printed_text = []

        with torch.no_grad():
            with torch.amp.autocast(device_type = device, dtype = amp_dtype):
                print("\nAssistant: ")
                for t in model.generate(idx = prompt, max_new_tokens = 100, temperature = temperature):

                    generated_tokens.append(t)
                    current_text = enc.decode(generated_tokens)
                    #extracting the new generated words that havent been printed
                    new_chars = current_text[len(printed_text):]
                    print(new_chars, end = "", flush = True)
                    printed_text = current_text

                    if "\n" in new_chars:
                        break
            print()




def main():
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    args = parse_args()
    model = GPT.from_pretrained()
    
    if args.checkpoint is None:
        print("No checkpoint provided. Running vanilla pretrained model.")
        
    else:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
        else:
            try:
                print(f"Loading checkpoint from {ckpt_path}...")
                checkpoint = torch.load(ckpt_path, map_location = device)
                state_dict = checkpoint.get("model_state_dict", checkpoint)
                is_lora = any("lora_"  in k for k in state_dict.keys())

                if is_lora:
                    print("Detected Lora checkpoint...")
                    model_lora = LoRA.inject_lora(model)
                    model_lora.load_state_dict(state_dict, strict = False)
                    model_lora.merge_and_unload()
                    print("Lora adapters loaded and weights merged.")
                    model = model_lora
                else:
                    print("Loading the finetuned weights of the model.")
                    model.load_state_dict(state_dict, strict = False)
            except:
                raise KeyError("Error in retrieving and laoding the weights.")


    
    terminal_chat(model = model, device = device, temperature = args.temperature)
            


if __name__ == "__main__":
    main()

