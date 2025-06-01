import torch
import json
import os

# This import is crucial and assumes torchtext is installed in the environment
# where this script is run.
try:
    from torchtext.vocab import Vocab
except ImportError:
    print("ERROR: torchtext is not installed or cannot be imported.")
    print("Please install torchtext (e.g., 'pip install torchtext==0.17.1' or the version used to create your vocabs)")
    print("in the environment where you run this script.")
    exit(1)

def convert_vocab_pth_to_json(pth_file_path, json_file_path):
    """
    Loads a torchtext.vocab.Vocab object from a .pth file,
    extracts its stoi and itos attributes, and saves them to a JSON file.
    """
    if not os.path.exists(pth_file_path):
        print(f"ERROR: Input file not found: {pth_file_path}")
        print("Please ensure the .pth vocabulary file is in the correct location.")
        print("You might need to download it from your Hugging Face Hub repository if it's not local.")
        return False

    print(f"Loading vocab from: {pth_file_path}")
    try:
        # The .pth file was saved directly as a Vocab object
        # map_location is important if the .pth was saved on a GPU and loaded on CPU
        vocab_obj = torch.load(pth_file_path, map_location=torch.device('cpu'))
    except Exception as e:
        print(f"Error loading {pth_file_path}: {e}")
        print("Ensure you have the correct version of torch and torchtext installed that was used to save these files.")
        return False

    if not isinstance(vocab_obj, Vocab):
        print(f"ERROR: Expected loaded object from {pth_file_path} to be a torchtext.vocab.Vocab, but got {type(vocab_obj)}")
        return False

    print(f"Extracting stoi and itos from {pth_file_path}...")
    # Using get_stoi() and get_itos() is generally safer than direct attribute access
    # if the Vocab class implementation changes minor details.
    stoi = vocab_obj.get_stoi()
    itos = vocab_obj.get_itos()

    vocab_data = {
        "stoi": stoi,
        "itos": itos
    }

    print(f"Saving vocab data to: {json_file_path}")
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=4)
    print(f"Successfully converted {pth_file_path} to {json_file_path}")
    return True

if __name__ == "__main__":
    # Define the paths to your .pth files and desired .json output files
    # These paths assume the script and .pth files are in the project root.
    # Adjust if your .pth files are elsewhere.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    src_pth_path = os.path.join(base_dir, "vocab_src.pth")
    tgt_pth_path = os.path.join(base_dir, "vocab_tgt.pth")

    src_json_path = os.path.join(base_dir, "vocab_src.json")
    tgt_json_path = os.path.join(base_dir, "vocab_tgt.json")

    print("Starting vocabulary conversion script...")
    print("IMPORTANT:")
    print("1. This script requires 'torch' and 'torchtext' (the version used to create your .pth vocab files, e.g., 0.17.x or 0.18.x) to be installed in your current Python environment.")
    print("2. Ensure your 'vocab_src.pth' and 'vocab_tgt.pth' files are in the same directory as this script, or update the paths.")
    print("   You might need to download them from your Hugging Face Hub repository first if they are not local.\n")

    success_src = convert_vocab_pth_to_json(src_pth_path, src_json_path)
    success_tgt = convert_vocab_pth_to_json(tgt_pth_path, tgt_json_path)

    if success_src and success_tgt:
        print("\nConversion complete for both vocabularies.")
        print(f"New files created: {src_json_path}, {tgt_json_path}")
        print("Please UPLOAD these JSON files to your Hugging Face Hub repository.")
        print("After uploading, we can proceed with refactoring app.py.")
    else:
        print("\nVocabulary conversion encountered errors. Please check the messages above.")
