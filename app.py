import streamlit as st
import torch
import sentencepiece as spm

from model import TransformerModel

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS = BASE_DIR / 'models' 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
sp_en = spm.SentencePieceProcessor()

sp_en.load(str(MODELS / "spm_en.model"))

sp_hi = spm.SentencePieceProcessor()
sp_hi.load(str(MODELS / "spm_hi.model"))

# Load model
model = TransformerModel(
    input_dim=sp_en.get_piece_size(),
    output_dim=sp_hi.get_piece_size()
).to(DEVICE)

checkpoint = torch.load(str(MODELS / "best_model.pth"), map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def translate(sentence, max_len=25):
    src_tokens = sp_en.encode(sentence, out_type=int)
    src_tensor = torch.LongTensor(src_tokens).unsqueeze(0).to(DEVICE)

    trg_indexes = [2]  # BOS

    for _ in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(src_tensor, trg_tensor)

        next_token = output[:, -1, :].argmax(1).item()
        trg_indexes.append(next_token)

        if next_token == 3:  # EOS
            break

    return sp_hi.decode(trg_indexes[1:-1])


st.title("English to Hindi Translator (Transformer)")

sentence = st.text_area("Enter English sentence:")

if st.button("Translate"):
    if sentence.strip():
        result = translate(sentence)
        st.success(result)
    else:
        st.warning("Please enter a sentence.")