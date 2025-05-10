const express = require('express');
const router = express.Router();
const { registerUser, loginUser, logoutUser } = require('../controllers/userController');
const { guest } = require('../middleware/authMiddleware');

// Login page
router.get('/login', guest, (req, res) => {
  res.render('login', { error: req.query.error || null });
});

// Login process
router.post('/login', loginUser);

// Registration page
router.get('/register', guest, (req, res) => {
  res.render('register', { error: req.query.error || null });
});

// Registration process
router.post('/register', registerUser);

// Logout
router.get('/logout', logoutUser);

module.exports = router;
