from flask import Flask, request, jsonify
from flask_cors import CORS
import math
import torch
import torch.nn as nn
import os
import sys

# sentencepiece modülü ile ilgili sorun var
spm_available = False
try:
    import sentencepiece as spm
    spm_available = True
    print("sentencepiece modülü başarıyla yüklendi")
except ModuleNotFoundError:
    print("sentencepiece modülü yüklenemedi! Modelin çalışması için bu modül gereklidir.")
    print("Metin üretimi devre dışı, ancak API çalışabilir.")

# Mevcut klasörün üst dizinine erişim
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# CONFIG - Orijinal app.py'den kopyalanan
MODEL_PATH = os.path.join(ROOT_DIR, "model_v11.pth")
TOKENIZER_PATH = os.path.join(ROOT_DIR, "fikra_tokenizer.model")
VOCAB_SIZE = 10000
MAX_LEN = 50
GEN_MAX_LENGTH = 150

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenizer - sadece sentencepiece kullanılabilir olduğunda çalışır
sp = None
if spm_available:
    try:
        sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
        print(f"Tokenizer başarıyla yüklendi: {TOKENIZER_PATH}")
    except Exception as e:
        print(f"Tokenizer yüklenemedi: {e}")
        spm_available = False

# Model Definition - Orijinal app.py'den kopyalanan
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

# Model yükleme - orijinal app.py'den kopyalanan
model = None
try:
    print(f"Model yükleniyor: {MODEL_PATH}")
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
    print("Model başarıyla yüklendi")
except Exception as e:
    print(f"Model yüklenirken hata: {e}")

# Metin üretme fonksiyonu - orijinal app.py'den kopyalanan
def generate_text(prompt):
    if not spm_available or sp is None:
        return "Üzgünüm, sentencepiece modülü yüklenemediği için metin üretemiyorum. Lütfen modülü yükleyip tekrar deneyin."
    
    if model is None:
        return "Üzgünüm, model yüklenemedi. Lütfen modeli kontrol edip tekrar deneyin."
    
    try:
        print(f"Prompt alındı: '{prompt}'")
        tokens=sp.encode(prompt,out_type=int)[-MAX_LEN:]
        gen=tokens.copy()
        mem=torch.zeros(1,1,model.d_model,device=device)
        
        print("Metin üretimi başlıyor...")
        for _ in range(GEN_MAX_LENGTH-len(tokens)):
            inp=torch.tensor([gen[-MAX_LEN:]],device=device)
            logits=model(inp,mem)[:,-1]/0.7
            next_tok=torch.multinomial(torch.softmax(logits,dim=-1),1).item()
            gen.append(next_tok)
            txt=sp.decode(gen)
            if next_tok==sp.eos_id() or (len(gen)>=50 and txt.strip()[-1] in ".!?"):
                break
                
        result = sp.decode(gen)
        # "??" ve "⁇" karakterlerini kaldır
        result = result.replace("??", "").replace("⁇", "")
        print(f"Üretilen metin: '{result[:30]}...'")
        return result
    except Exception as e:
        print(f"Metin üretiminde hata: {e}")
        return f"Metin üretiminde hata: {e}"

# Flask app
app = Flask(__name__)
CORS(app)  # Tüm kökenlere izin ver

@app.route('/generate', methods=['POST', 'OPTIONS'])
def api_generate():
    # OPTIONS isteğini ele al
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    # Gelen veriyi işle
    try:
        data = request.json
        if not data:
            return jsonify({"error": "JSON verisi gerekli"}), 400
        
        # 'prompt' veya 'message' anahtarını kabul et
        prompt = None
        if 'prompt' in data:
            prompt = data['prompt']
        elif 'message' in data:
            prompt = data['message']
        
        if not prompt:
            return jsonify({"error": "Prompt veya message gerekli"}), 400
        
        # Metni üret
        result = generate_text(prompt)
        
        # Yanıtı döndür
        return jsonify({"text": result, "status": "success"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "up",
        "model_loaded": model is not None,
        "tokenizer_loaded": sp is not None and spm_available
    })

if __name__ == '__main__':
    # Port 5000 yerine 5001 kullan (MacOS'ta AirPlay 5000 portunu kullanıyor)
    app.run(debug=True, host='0.0.0.0', port=5001)
