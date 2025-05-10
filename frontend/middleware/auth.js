const jwt = require('jsonwebtoken');
const User = require('../models/User');

// JWT gizli anahtarı - production ortamında .env dosyasında saklanmalı
const JWT_SECRET = 'fikrabot-jwt-secret-key';

// JWT token oluştur
const generateToken = (userId) => {
  return jwt.sign({ id: userId }, JWT_SECRET, { expiresIn: '7d' });
};

// Token doğrulama middleware'i
const authenticateToken = (req, res, next) => {
  // Session kontrolü
  if (req.session && req.session.userId) {
    req.user = User.findUserById(req.session.userId);
    return next();
  }
  
  // Cookie'den token al
  const token = req.cookies.token;
  
  if (!token) {
    return res.redirect('/login');
  }
  
  try {
    // Token'ı doğrula
    const decoded = jwt.verify(token, JWT_SECRET);
    
    // Kullanıcıyı bul
    const user = User.findUserById(decoded.id);
    
    if (!user) {
      res.clearCookie('token');
      return res.redirect('/login');
    }
    
    // İsteğe kullanıcı ekle
    req.user = user;
    req.session.userId = user.id;
    
    next();
  } catch (error) {
    res.clearCookie('token');
    return res.redirect('/login');
  }
};

// Oturum açmış kullanıcıları login sayfasından uzaklaştır
const redirectIfAuthenticated = (req, res, next) => {
  if (req.session && req.session.userId) {
    return res.redirect('/dashboard');
  }
  
  const token = req.cookies.token;
  
  if (!token) {
    return next();
  }
  
  try {
    const decoded = jwt.verify(token, JWT_SECRET);
    const user = User.findUserById(decoded.id);
    
    if (user) {
      req.session.userId = user.id;
      return res.redirect('/dashboard');
    }
    
    next();
  } catch (error) {
    res.clearCookie('token');
    next();
  }
};

module.exports = {
  generateToken,
  authenticateToken,
  redirectIfAuthenticated
};
