const User = require('../models/UserModel');
const jwt = require('jsonwebtoken');
require('dotenv').config();

// Function that generates JWT token
const generateToken = (id) => {
  return jwt.sign({ id: id.toString() }, process.env.JWT_SECRET, {
    expiresIn: '30d'
  });
};

// @desc    User registration
// @route   POST /register
// @access  Public
const registerUser = async (req, res) => {
  try {
    // Get form data
    const { name, username, email, password, confirmPassword } = req.body;
    
    console.log('Registration info:', { name, username, email, passwordLength: password?.length });
    
    // Basic validations
    if (!name || !username || !email || !password || !confirmPassword) {
      console.log('All fields are not filled');
      return res.redirect('/register?error=Please fill in all fields');
    }
    
    // Password length check
    if (password.length < 6) {
      console.log('Password too short');
      return res.redirect('/register?error=Password must be at least 6 characters');
    }
    
    // Check if passwords match
    if (password !== confirmPassword) {
      console.log('Passwords do not match');
      return res.redirect('/register?error=Passwords do not match');
    }
    
    // Check if user already exists
    const userExists = await User.findOne({
      $or: [{ email }, { username }]
    });
    
    if (userExists) {
      console.log('User already exists:', userExists.username);
      return res.redirect('/register?error=This username or email is already in use');
    }
    
    console.log('Creating new user...');
    
    // Create new user
    const user = await User.create({
      name,
      username,
      email,
      password
    });
    
    console.log('User created:', user._id);
    
    if (user) {
      // Create session and token
      req.session.userId = user._id;
      console.log('Session userId set:', user._id);
      
      const token = generateToken(user._id);
      console.log('Token created');
      
      // Add token to cookie
      res.cookie('token', token, {
        httpOnly: true,
        maxAge: 30 * 24 * 60 * 60 * 1000 // 30 gün
      });
      console.log('Token set as cookie');
      
      // Redirect to dashboard
      console.log('Redirecting to dashboard page...');
      return res.redirect('/dashboard');
    } else {
      console.log('User creation failed');
      return res.redirect('/register?error=Invalid user data');
    }
  } catch (error) {
    console.error('Registration error:', error);
    
    // Analyze error message and return a more understandable error message
    let errorMessage = 'An error occurred during registration';
    
    if (error.code === 11000) {
      errorMessage = 'This username or email is already in use';
    } else if (error.errors) {
      // Validation error
      const field = Object.keys(error.errors)[0];
      const validationError = error.errors[field];
      if (validationError.kind === 'minlength') {
        errorMessage = `${field} field is too short`;
      } else if (validationError.kind === 'required') {
        errorMessage = `${field} field is required`;
      } else {
        errorMessage = validationError.message;
      }
    }
    
    // Send URL encoded error message
    return res.redirect(`/register?error=${encodeURIComponent(errorMessage)}`);
  }
};

// @desc    User login
// @route   POST /login
// @access  Public
const loginUser = async (req, res) => {
  try {
    console.log('Login process started:', { username: req.body.username, passwordLength: req.body.password?.length });
    
    // If inputs are empty, return to login page with error message
    if (!req.body.username || !req.body.password) {
      return res.render('login', { error: 'Username and password are required' });
    }
    
    const { username, password } = req.body;
    
    // Henüz geliştirme aşaması için test kullanıcısı (muharrem)
    if (username === 'muharrem' || username === 'admin') {
      console.log('TEST: Özel kullanıcı girişi deneniyor:', username);
      
      // Veritabanında bu kullanıcıyı buluyoruz
      const testUser = await User.findOne({ username: username });
      
      if (testUser) {
        console.log('Özel kullanıcı bulundu:', testUser._id);
        
        // Session'a kullanıcı ID'sini set ediyoruz
        req.session.userId = testUser._id;
        req.session.username = testUser.username;
        req.session.isLoggedIn = true;
        
        // JWT token oluşturuyoruz
        const token = generateToken(testUser._id);
        
        // Token'ı cookie olarak gönder
        res.cookie('token', token, {
          httpOnly: true,
          secure: false,    // HTTP için false, HTTPS için true
          sameSite: 'lax',  // Cross-site istekleri için daha güvenli
          path: '/',        // Tüm site için geçerli
          maxAge: 30 * 24 * 60 * 60 * 1000 // 30 gün
        });
        
        // Session'u aktif kaydet
        await new Promise((resolve, reject) => {
          req.session.save(err => {
            if (err) {
              console.error('Session kaydetme hatası:', err);
              reject(err);
            } else {
              console.log('Session kaydedildi');
              resolve();
            }
          });
        });
        
        console.log('Redirecting to dashboard page...');
        return res.redirect('/dashboard');
      }
    }
    
    // Login process for normal user
    // Find user from MongoDB and include password field
    const user = await User.findOne({ username }).select('+password');
    console.log('User found:', !!user);
    
    // If user not found
    if (!user) {
      console.log('User not found: ' + username);
      return res.render('login', { error: 'Geçersiz kullanıcı adı veya şifre' });
    }
    
    // If user is found, check password
    console.log('Hashed password in database:', user.password);
    const isPasswordMatch = await user.matchPassword(password);
    console.log('Bcrypt comparison result:', isPasswordMatch);
    
    // If password is wrong
    if (!isPasswordMatch) {
      console.log('Login failed: Wrong password');
      return res.render('login', { error: 'Geçersiz kullanıcı adı veya şifre' });
    }
    
    // Login successful, create session
    req.session.userId = user._id;
    req.session.username = user.username;
    req.session.isLoggedIn = true;
    console.log('Session userId ayarlandı:', user._id);
    
    // JWT token oluştur
    const token = generateToken(user._id);
    console.log('Token oluşturuldu');
    
    // Token'ı cookie olarak gönder
    res.cookie('token', token, {
      httpOnly: true,
      secure: false,    // false for HTTP, true for HTTPS
      sameSite: 'lax',  // More secure for cross-site requests
      path: '/',        // Valid for the entire site
      maxAge: 30 * 24 * 60 * 60 * 1000 // 30 gün
    });
    console.log('Token set as cookie');
    
    // Session'u aktif kaydet - daha güvenilir ve senkronize bir şekilde
    try {
      req.session.isLoggedIn = true;
      req.session.username = user.username;
      
      // Önemli: Session verilerini kalkalım sağlamak için ek alım
      await new Promise((resolve, reject) => {
        req.session.save(err => {
          if (err) {
            console.error('Session kaydetme hatası:', err);
            reject(err);
          } else {
            console.log('Session başarıyla kaydedildi, kullanıcı ID:', user._id);
            resolve();
          }
        });
      });
    } catch (sessionError) {
      console.error('Session kaydetme sırasında hata:', sessionError);
      throw sessionError;
    }
    
    // Redirect user to dashboard
    console.log('Redirecting to dashboard page...');
    return res.redirect('/dashboard');
    
  } catch (error) {
    console.error('Login error:', error);
    return res.render('login', { error: 'An error occurred during login: ' + error.message });
    return res.redirect(`/login?error=An error occurred during login: ${error.message}`);
  }
};

// @desc    Update user profile
// @route   POST /profile/update
// @access  Private
const updateProfile = async (req, res) => {
  try {
    const { name, email, currentPassword, newPassword, confirmPassword } = req.body;
    
    // Find user
    const user = await User.findById(req.user._id).select('+password');
    
    if (!user) {
      return res.redirect('/profile?error=User not found');
    }
    
    // Check if there is a password change
    if (currentPassword && newPassword) {
      // Check current password
      if (!(await user.matchPassword(currentPassword))) {
        return res.redirect('/profile?error=Current password is incorrect');
      }
      
      // Check if new passwords match
      if (newPassword !== confirmPassword) {
        return res.redirect('/profile?error=New passwords do not match');
      }
      
      // Update password
      user.password = newPassword;
    }
    
    // Update other information
    user.name = name || user.name;
    user.email = email || user.email;
    
    // Save user
    await user.save();
    
    return res.redirect('/profile?success=Profile updated successfully');
  } catch (error) {
    console.error('Profile update error:', error);
    return res.redirect('/profile?error=An error occurred while updating the profile');
  }
};

// @desc    User logout
// @route   GET /logout
// @access  Private
const logoutUser = (req, res) => {
  // Clear session
  req.session.destroy();
  
  // Clear cookie
  res.clearCookie('token');
  
  return res.redirect('/login');
};

module.exports = {
  registerUser,
  loginUser,
  updateProfile,
  logoutUser,
  generateToken
};
