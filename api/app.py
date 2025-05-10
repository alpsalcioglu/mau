from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import math
import torch
import torch.nn as nn
import os
import sys
import datetime
import sentencepiece as spm



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model_v11.pth")
TOKENIZER_PATH = os.path.join(BASE_DIR, "fikra_tokenizer.model")
VOCAB_SIZE = 10000
MAX_LEN = 50
GEN_MAX_LENGTH = 150

app = Flask(__name__)


CORS(app)


def add_cors_headers(f):
    def decorated_function(*args, **kwargs):
        resp = f(*args, **kwargs)
        

        if isinstance(resp, tuple):
            body, status = resp
            if len(resp) > 2:
                headers = resp[2]
            else:
                headers = {}
        else:
            body = resp
            status = 200
            headers = {}
        

        headers['Access-Control-Allow-Origin'] = '*'
        headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        
        return make_response(jsonify(body) if body is not None else '', status, headers)
    
    decorated_function.__name__ = f.__name__
    return decorated_function


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


try:
    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    print("Tokenizer başarıyla yüklendi")
except Exception as e:
    print(f"Tokenizer yüklenirken hata: {e}")
    sp = None


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


model = None
try:
    model = SimpleDecoderLLM(VOCAB_SIZE, max_len=MAX_LEN).to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device)
    mapped = {}
    for k, v in ckpt.items():
        nk = k.replace("token_embedding.", "embed.")\
             .replace("pos_encoder.pe", "pos.pe")\
             .replace("transformer_decoder.", "trans.")\
             .replace("fc_out.", "fc.")
        mapped[nk] = v
    model.load_state_dict(mapped, strict=False)
    model.eval()
    print("Model başarıyla yüklendi")
except Exception as e:
    print(f"Model yüklenirken hata: {e}")

def generate_text(prompt):
    """
    Verilen prompt'a göre metin üreten fonksiyon.
    Çok basitleştirilmiş ve doğrudan çalışan bir versiyonu.
    """
    try:
        print(f"Generate text fonksiyonu çağrıldı. Prompt: {prompt}")
        

        if sp is None:
            print("Hata: Tokenizer yüklenemedi")
            return {"error": "Tokenizer yüklenemedi"}
        

        if model is None:
            print("Hata: Model yüklenemedi")
            return {"error": "Model yüklenemedi"}

        try:

            input_ids = sp.encode(prompt, out_type=int)[-MAX_LEN:]
            gen = input_ids.copy()
            mem = torch.zeros(1, 1, model.d_model, device=device)
            for _ in range(GEN_MAX_LENGTH-len(input_ids)):
                inp = torch.tensor([gen[-MAX_LEN:]], device=device)
                logits = model(inp, mem)[:, -1]/0.7
                next_tok = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
                gen.append(next_tok)
                txt = sp.decode(gen)
                if next_tok == sp.eos_id() or (len(gen) >= 50 and txt.strip()[-1] in ".!?"):
                    break
            result = sp.decode(gen)

            result = result.replace("??", "").replace("⁇", "")
            print(f"Başarılı yanıt üretildi: {result[:30]}...")
            return {"text": result, "status": "success"}
            
        except Exception as e:
            print(f"Model üretim hatası: {str(e)}")
            return {"error": f"Model üretim hatası: {str(e)}"}
    except Exception as e:
        print(f"Genel hata: {str(e)}")
        return {"error": str(e)}

@app.route('/generate', methods=['POST', 'OPTIONS'])
def api_generate():
    response_headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Content-Type': 'application/json'
    }
    

    if request.method == 'OPTIONS':
        return jsonify({}), 200, response_headers
    
    print("API isteği alındı!")
    print("Request data:", request.data)
    print("Request JSON:", request.json)
    
    try:
        data = request.json
        if not data:
            print("Hata: JSON verisi boş")
            return jsonify({"error": "JSON verisi gereklidir"}), 400, response_headers
        

        if 'prompt' in data:
            prompt = data['prompt']
        elif 'message' in data:
            prompt = data['message']
        else:
            print("Hata: 'prompt' veya 'message' anahtarı bulunamadı")
            print("Mevcut anahtarlar:", data.keys())
            return jsonify({"error": "'prompt' veya 'message' alanı gereklidir"}), 400, response_headers
        
        print(f"Alınan prompt: '{prompt}'")
        

        if prompt.lower() == "test":
            test_response = {"text": "Bu bir test yanıtıdır.", "status": "success"}
            return jsonify(test_response), 200, response_headers
        
        result = generate_text(prompt)
        print("Sonuç:", result)
        
        return jsonify(result), 200, response_headers
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
        return jsonify({"error": str(e)}), 500, response_headers

@app.route('/health', methods=['GET'])
def health_check():
    status = {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": sp is not None
    }
    return jsonify(status)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
