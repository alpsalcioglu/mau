const express = require('express');
const router = express.Router();
const { protect } = require('../middleware/authMiddleware');
const { updateProfile } = require('../controllers/userController');

// Profil sayfası
router.get('/profile', protect, (req, res) => {
  res.render('profile', { 
    user: req.user,
    error: req.query.error || null,
    success: req.query.success || null
  });
});

// Profil güncelleme
router.post('/profile/update', protect, updateProfile);

module.exports = router;
