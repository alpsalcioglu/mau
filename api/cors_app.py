from flask import Flask, request, jsonify
from flask_cors import CORS
import math
import torch
import os
import sys
import datetime

# Model dosyaları
MAX_LEN = 144
PAD_ID = 0
GEN_MAX_LENGTH = 150

app = Flask(__name__)
# CORS'u tamamen etkinleştir, tüm kaynaklara izin ver
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Globals
model = None
sp = None

print("Tokenizer yükleniyor...")
try:
    # sentencepiece modülünü içe aktar
    import sentencepiece as spm
    
    # Tokenizer'ı yükle
    sp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../model/vt.model")
    sp = spm.SentencePieceProcessor()
    sp.load(sp_path)
    print("Tokenizer başarıyla yüklendi")
except Exception as e:
    print(f"Tokenizer yüklenirken hata: {e}")

print("Model yükleniyor...")
try:
    import numpy as np
    
    # Torch modülünü içe aktar
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Mini modeli yükle
    class MiniGPT(torch.nn.Module):
        def __init__(self, d_model=768, n_heads=12, n_layers=12, vocab_size=32000):
            super().__init__()
            self.d_model = d_model
            # Embedding ve LM head için aynı ağırlığı kullan
            self.wte = torch.nn.Embedding(vocab_size, d_model)
            self.h = torch.nn.ModuleList([torch.nn.MultiheadAttention(d_model, n_heads), 
                             torch.nn.Sequential(torch.nn.Linear(d_model, 4*d_model), 
                                        torch.nn.GELU(), 
                                        torch.nn.Linear(4*d_model, d_model))
                            for _ in range(n_layers)])
            self.ln_f = torch.nn.LayerNorm(d_model)
            
        def forward(self, x, past=None):
            x = self.wte(x)
            for i in range(0, len(self.h), 2):
                attn, _ = self.h[i](x.transpose(0, 1), x.transpose(0, 1), x.transpose(0, 1))
                x = x + attn.transpose(0, 1)
                x = x + self.h[i+1](x)
            x = self.ln_f(x)
            return torch.matmul(x, self.wte.weight.T)

    # Model yükleme
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../model/mini.pt")
    model = MiniGPT(d_model=384, n_heads=6, n_layers=6, vocab_size=len(sp))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
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
        
        # Tokenizer kontrolü
        if sp is None:
            print("Hata: Tokenizer yüklenemedi")
            return {"error": "Tokenizer yüklenemedi"}
        
        # Model kontrolü
        if model is None:
            print("Hata: Model yüklenemedi")
            return {"error": "Model yüklenemedi"}
            
        try:
            # Prompt'u tokenize et
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
            # "??" ve "⁇" karakterlerini kaldır
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
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        return response
        
    print("API isteği alındı!")
    print("Request data:", request.data)
    print("Request JSON:", request.json)
    
    try:
        data = request.json
        if not data:
            print("Hata: JSON verisi boş")
            return jsonify({"error": "JSON verisi gereklidir"}), 400
        
        # 'prompt' veya 'message' anahtarını kabul et
        if 'prompt' in data:
            prompt = data['prompt']
        elif 'message' in data:
            prompt = data['message']
        else:
            print("Hata: 'prompt' veya 'message' anahtarı bulunamadı")
            print("Mevcut anahtarlar:", data.keys())
            return jsonify({"error": "'prompt' veya 'message' alanı gereklidir"}), 400
        
        print(f"Alınan prompt: '{prompt}'")
        
        # Test için basit yanıt
        if prompt.lower() == "test":
            test_response = {"text": "Bu bir test yanıtıdır! CORS çalışıyor!", "status": "success"}
            return jsonify(test_response)
        
        result = generate_text(prompt)
        print("Sonuç:", result)
        
        return jsonify(result)
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    health = {
        "status": "up",
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_loaded": model is not None,
        "tokenizer_loaded": sp is not None
    }
    return jsonify(health)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
