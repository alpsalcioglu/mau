document.addEventListener('DOMContentLoaded', () => {
  console.log('DOM yüklendi, sayfa başlatılıyor');
  
  // HTML ID'leri ile uyumlu olarak elementleri seçiyoruz
  const messageForm = document.getElementById('messageForm');
  const userInput = document.getElementById('userInput');
  const messagesContainer = document.getElementById('messages');
  const chatContainer = document.querySelector('.chat-container');
  
  console.log('Form bulundu mu:', !!messageForm);
  console.log('Input bulundu mu:', !!userInput);
  console.log('Messages bulundu mu:', !!messagesContainer);
  
  // Form ve input elementleri var mı kontrol ediyoruz
  if (messageForm && userInput && messagesContainer) {
    console.log('Tüm gerekli elementler bulundu, form event listenerı ekleniyor');
    
    // Form submit event listener
    messageForm.addEventListener('submit', async (e) => {
      // Sayfanın yeniden yüklenmesini engelle
      e.preventDefault();
      console.log('Form submit edildi');
      
      // Mesajı input'tan al ve boşlukları temizle
      const message = userInput.value.trim();
      if (!message) {
        console.log('Boş mesaj, işlem iptal');
        return;
      }
      
      console.log('Gönderilen mesaj:', message);
      
      // Kullanıcı mesajını ekrana ekle
      addMessage('user', message);
      
      // Input'u temizle
      userInput.value = '';
      
      try {
        // API'ye istek gönder
        const response = await fetch('/proxy-api/generate', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ prompt: message })
        });
        
        console.log('API yanıt alındı, status:', response.status);
        
        if (!response.ok) {
          throw new Error(`HTTP hata kodu: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('API yanıt verisi:', data);
        
        // Bot yanıtını işle
        if (data.error) {
          console.error('API hatası:', data.error);
          addMessage('bot', `Hata: ${data.error}`);
        } else if (data.text) {
          addMessage('bot', data.text);
        } else {
          console.warn('Beklenmeyen API yanıt formatı:', data);
          addMessage('bot', 'Sunucu yanıt formatı hatası. Lütfen tekrar deneyin.');
        }
        
      } catch (error) {
        console.error('API istek hatası:', error);
        addMessage('bot', `Sunucu hatası: ${error.message}. Lütfen tekrar deneyin.`);
      }
      
      // Sohbeti en alta kaydır
      if (chatContainer) {
        chatContainer.scrollTop = chatContainer.scrollHeight;
      }
    });
    
    console.log('Form event listener eklendi');
  } else {
    console.log('Bazı form elementleri sayfada bulunamadı');
  }
  
  // Mesajı chat penceresine ekle
  function addMessage(type, content) {
    if (!messagesContainer) {
      console.error('Mesaj containerı bulunamadı');
      return;
    }
    
    console.log(`${type} mesajı ekleniyor:`, content.substring(0, 30));
    
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', type === 'user' ? 'user' : 'bot');
    
    const messageContent = document.createElement('div');
    messageContent.classList.add('message-content');
    messageContent.textContent = content;
    
    messageDiv.appendChild(messageContent);
    messagesContainer.appendChild(messageDiv);
    
    // Sohbeti en alta kaydır
    if (chatContainer) {
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }
  }
});
