const express = require('express');
const bodyParser = require('body-parser');
const axios = require('axios');
const path = require('path');
const session = require('express-session');
const cookieParser = require('cookie-parser');
const dotenv = require('dotenv');
const connectDB = require('./config/db');

// .env değişkenlerini yükle
dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;
// API URL'sini 5001 portuna güncelliyoruz (MacOS'ta AirPlay 5000 portunu kullanabilir)
const API_URL = 'http://127.0.0.1:5001';

// MongoDB bağlantısı
connectDB().then(() => {
  console.log('MongoDB veritabanı bağlantısı kuruldu');
}).catch(err => {
  console.error('MongoDB bağlantı hatası:', err);
});

// Middleware
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, 'public')));
app.use(cookieParser());

// Session middleware
app.use(session({
  secret: process.env.JWT_SECRET || 'fikrabot-session-secret',
  resave: true,
  saveUninitialized: true,
  name: 'fikrabot.sid', // Cookie ismi
  cookie: { 
    secure: false,        // Development ortamında güvenli olmayan HTTP bağlantıları için
    maxAge: 30 * 24 * 60 * 60 * 1000, // 30 gün
    httpOnly: true,      // JavaScript erişimini engelle
    sameSite: 'lax',     // Cross-site istekleri için daha uygun
    path: '/'            // Tüm uygulamada geçerli
  }
}));

// Güvenlik ve çerez işleme için ara katman
app.use((req, res, next) => {
  // Session bilgilerini her istekte loglama (debug için)
  console.log('\n=== YENİ İSTEK ===');
  console.log('URL:', req.url);
  console.log('Session ID:', req.sessionID);
  console.log('Session User ID:', req.session?.userId);
  console.log('Cookies:', req.cookies);
  
  // Kimlik doğru ise oturum bilgisini res.locals ile taşı
  res.locals.user = req.session.userId ? { userId: req.session.userId } : null;
  next();
});

// View engine
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// Rotaları içe aktar
const authRoutes = require('./routes/auth');
const dashboardRoutes = require('./routes/dashboard');
const profileRoutes = require('./routes/profile');

// Rotaları kullan
app.use('/', authRoutes);
app.use('/', dashboardRoutes);
app.use('/', profileRoutes);

// Ana sayfa - kimlik doğrulaması olmayan kullanıcıları login sayfasına yönlendir
app.get('/', (req, res) => {
  // Cookie veya session kontrolü
  if (req.session && req.session.userId) {
    return res.redirect('/dashboard');
  }
  if (req.cookies.token) {
    return res.redirect('/dashboard');
  }
  
  // Kimliği doğrulanmamış kullanıcıları login sayfasına yönlendir
  res.redirect('/login');
});

// Metin üretimi için API isteği
app.post('/generate', async (req, res) => {
  try {
    console.log('Frontend API request received:', req.body);
    
    // Accept both prompt and message parameter names
    const userMessage = req.body.prompt || req.body.message;
    
    if (!userMessage) {
      console.error('Error: Message text not sent');
      return res.status(400).json({ error: 'Message text is required' });
    }

    console.log(`Sending to Flask API: ${userMessage}`);
    
    // Make a request to our own server, use it as a proxy (to bypass CORS issues)
    try {
      // Make a request to the /proxy-api endpoint in our own Node.js application, not directly to the Flask API
      const response = await axios.post('/proxy-api/generate', 
        { prompt: userMessage },
        { 
          headers: { 'Content-Type': 'application/json' },
          timeout: 20000 // Add 20 second timeout
        }
      );

      console.log('Flask API response:', response.data);
      
      // Return response as JSON
      return res.json(response.data);
    } catch (apiError) {
      console.error('API request error:', apiError.message);
      // Process API errors in more detail
      if (apiError.response) {
        // Server response but with error code
        console.log('API response status:', apiError.response.status);
        console.log('API response data:', apiError.response.data);
        return res.status(apiError.response.status).json({
          error: 'API error',
          details: apiError.response.data,
          status: apiError.response.status
        });
      } else if (apiError.request) {
        // Request made but no response received
        return res.status(503).json({
          error: 'API did not respond',
          details: 'Service is currently unavailable'
        });
      } else {
        // Error while creating request
        return res.status(500).json({
          error: 'API request could not be initiated',
          details: apiError.message
        });
      }
    }
  } catch (error) {
    console.error('General error:', error);
    res.status(500).json({ 
      error: 'Server error',
      details: error.message 
    });
  }
});

// API proxy endpoint - To overcome CORS issue
app.all('/proxy-api/*', async (req, res) => {
  try {
    const endpoint = req.path.replace('/proxy-api', '');
    console.log(`Proxy isteği: ${req.method} ${endpoint}`);
    console.log('Request body:', req.body);
    
    const response = await axios({
      method: req.method,
      url: `${API_URL}${endpoint}`,
      data: req.body,
      headers: {
        'Content-Type': 'application/json'
      },
      timeout: 30000 // 30 seconds
    });
    
    console.log(`Proxy response: ${response.status}`);
    res.status(response.status).json(response.data);
  } catch (error) {
    console.error('Proxy hatası:', error.message);
    if (error.response) {
      // API response received but there's an error
      console.log('API error:', error.response.status, error.response.data);
      res.status(error.response.status).json(error.response.data);
    } else {
      // Could not connect to API
      res.status(500).json({
        error: 'API servisine bağlanılamadı',
        details: error.message
      });
    }
  }
});

// Check API status
app.get('/check-api', async (req, res) => {
  try {
    const response = await axios.get(`${API_URL}/health`);
    res.json(response.data);
  } catch (error) {
    res.status(500).json({ 
      status: 'error',
      message: 'Could not establish API connection',
      details: error.message
    });
  }
});

// Start the server
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
