import streamlit as st
# Streamlit yapılandırma ayarı ilk komut olarak gelmelidir
st.set_page_config(layout="wide")

import math
import torch
import torch.nn as nn

# sentencepiece modülü ile ilgili sorun var
spm_available = False
try:
    import sentencepiece as spm
    spm_available = True
except ModuleNotFoundError:
    st.error("sentencepiece modülü yüklenemedi! Modelin çalışması için bu modül gereklidir.")
    st.info("Metin üretimi devre dışı, ancak arayüz görüntülenebilir.")


# CONFIG
MODEL_PATH     = "model_v11.pth"
TOKENIZER_PATH = "fikra_tokenizer.model"
VOCAB_SIZE     = 10000
MAX_LEN        = 50
GEN_MAX_LENGTH = 150

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenizer - sadece sentencepiece kullanılabilir olduğunda çalışır
sp = None
if spm_available:
    try:
        sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    except Exception as e:
        st.error(f"Tokenizer yüklenemedi: {e}")
        spm_available = False

# Model Definition (same as before)...
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=500):
        super().__init__()
        pe=torch.zeros(max_len,d_model)
        pos=torch.arange(max_len).unsqueeze(1).float()
        div=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000)/d_model))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer("pe",pe.unsqueeze(0))
    def forward(self,x): return x + self.pe[:,:x.size(1)]

class SimpleDecoderLLM(nn.Module):
    def __init__(self,vocab_size,d_model=256,nhead=4,num_layers=2,max_len=50):
        super().__init__()
        self.embed=nn.Embedding(vocab_size,d_model)
        self.pos=PositionalEncoding(d_model,max_len)
        layer=nn.TransformerDecoderLayer(d_model,nhead)
        self.trans=nn.TransformerDecoder(layer,num_layers)
        self.fc=nn.Linear(d_model,vocab_size)
        self.d_model=d_model; self.max_len=max_len

    def forward(self,tgt,mem):
        x=self.embed(tgt)*math.sqrt(self.d_model)
        x=self.pos(x).transpose(0,1)
        mask=nn.Transformer.generate_square_subsequent_mask(x.size(0)).to(x.device)
        out=self.trans(x,mem,tgt_mask=mask)
        return self.fc(out).transpose(0,1)

# Load model with key‑mapping fix
model=SimpleDecoderLLM(VOCAB_SIZE,max_len=MAX_LEN).to(device)
ckpt=torch.load(MODEL_PATH,map_location=device)
mapped={}
for k,v in ckpt.items():
    nk=k.replace("token_embedding.","embed.")\
         .replace("pos_encoder.pe","pos.pe")\
         .replace("transformer_decoder.","trans.")\
         .replace("fc_out.","fc.")
    mapped[nk]=v
model.load_state_dict(mapped,strict=False)
model.eval()

def generate_text(prompt):
    if not spm_available or sp is None:
        return "Üzgünüm, sentencepiece modülü yüklenemediği için metin üretemiyorum. Lütfen modülü yükleyip tekrar deneyin."
    
    try:
        tokens=sp.encode(prompt,out_type=int)[-MAX_LEN:]
        gen=tokens.copy()
        mem=torch.zeros(1,1,model.d_model,device=device)
        for _ in range(GEN_MAX_LENGTH-len(tokens)):
            inp=torch.tensor([gen[-MAX_LEN:]],device=device)
            logits=model(inp,mem)[:,-1]/0.7
            next_tok=torch.multinomial(torch.softmax(logits,dim=-1),1).item()
            gen.append(next_tok)
            txt=sp.decode(gen)
            if next_tok==sp.eos_id() or (len(gen)>=50 and txt.strip()[-1] in ".!?"):
                break
        return sp.decode(gen)
    except Exception as e:
        return f"Metin üretiminde hata: {e}"

# Streamlit UI
st.title("🤖 FıkraBot")

if "history" not in st.session_state:
    st.session_state.history=[]

def on_submit():
    user = st.session_state.user_input.strip()
    if not user:
        return
    out = generate_text(user)
    # "??" ve "⁇" karakterlerini kaldır
    out = out.replace("??", "").replace("⁇", "")
    st.session_state.history.append(("Sen", user))
    st.session_state.history.append(("FıkraBot", out))
    st.session_state.user_input = ""


# Display chat
for sender,msg in st.session_state.history:
    st.markdown(f"**{sender}:** {msg}")

# Input at bottom
st.text_input("Senin mesajın:", key="user_input", on_change=on_submit)
