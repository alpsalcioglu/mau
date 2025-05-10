const fs = require('fs');
const path = require('path');
const bcrypt = require('bcryptjs');

// Kullanıcı verilerini saklayacağımız dosya yolu
const dataPath = path.join(__dirname, '../data/users.json');

// Kullanıcı veritabanını yükle veya oluştur
const loadUsers = () => {
  try {
    // Eğer dosya yoksa, boş bir kullanıcı listesi oluştur
    if (!fs.existsSync(dataPath)) {
      fs.writeFileSync(dataPath, JSON.stringify([]));
      return [];
    }
    
    // Dosyayı oku ve JSON olarak parse et
    const data = fs.readFileSync(dataPath, 'utf8');
    return JSON.parse(data);
  } catch (error) {
    console.error('Kullanıcı verileri yüklenirken hata:', error);
    return [];
  }
};

// Kullanıcı veritabanını kaydet
const saveUsers = (users) => {
  try {
    fs.writeFileSync(dataPath, JSON.stringify(users, null, 2));
    return true;
  } catch (error) {
    console.error('Kullanıcı verileri kaydedilirken hata:', error);
    return false;
  }
};

// Kullanıcı oluştur
const createUser = async (userData) => {
  const users = loadUsers();
  
  // Kullanıcı adı veya e-posta zaten kullanılıyor mu?
  const existingUser = users.find(user => 
    user.username === userData.username || user.email === userData.email
  );
  
  if (existingUser) {
    throw new Error('Bu kullanıcı adı veya e-posta zaten kullanımda.');
  }
  
  // Şifreyi hash'le
  const salt = await bcrypt.genSalt(10);
  const hashedPassword = await bcrypt.hash(userData.password, salt);
  
  // Yeni kullanıcı oluştur
  const newUser = {
    id: Date.now().toString(),
    username: userData.username,
    email: userData.email,
    password: hashedPassword,
    name: userData.name || userData.username,
    profilePicture: userData.profilePicture || 'default-avatar.png',
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString()
  };
  
  // Kullanıcıyı kaydet
  users.push(newUser);
  saveUsers(users);
  
  // Şifreyi hariç tut
  const { password, ...userWithoutPassword } = newUser;
  return userWithoutPassword;
};

// Kullanıcı kimlik doğrulama
const authenticateUser = async (username, password) => {
  const users = loadUsers();
  
  // Kullanıcıyı bul
  const user = users.find(user => user.username === username);
  
  if (!user) {
    throw new Error('Kullanıcı adı veya şifre yanlış.');
  }
  
  // Şifreyi karşılaştır
  const isMatch = await bcrypt.compare(password, user.password);
  
  if (!isMatch) {
    throw new Error('Kullanıcı adı veya şifre yanlış.');
  }
  
  // Şifreyi hariç tut
  const { password: _, ...userWithoutPassword } = user;
  return userWithoutPassword;
};

// Kullanıcıyı ID'ye göre bul
const findUserById = (userId) => {
  const users = loadUsers();
  const user = users.find(user => user.id === userId);
  
  if (!user) return null;
  
  // Şifreyi hariç tut
  const { password, ...userWithoutPassword } = user;
  return userWithoutPassword;
};

// Kullanıcı profilini güncelle
const updateUserProfile = async (userId, updateData) => {
  const users = loadUsers();
  const userIndex = users.findIndex(user => user.id === userId);
  
  if (userIndex === -1) {
    throw new Error('Kullanıcı bulunamadı.');
  }
  
  // Güncellenebilecek alanlar
  const allowedUpdates = ['name', 'email', 'profilePicture'];
  
  // Sadece izin verilen alanları güncelle
  for (const field of allowedUpdates) {
    if (updateData[field]) {
      users[userIndex][field] = updateData[field];
    }
  }
  
  // Şifre güncellemesi varsa, hash'le
  if (updateData.password) {
    const salt = await bcrypt.genSalt(10);
    users[userIndex].password = await bcrypt.hash(updateData.password, salt);
  }
  
  users[userIndex].updatedAt = new Date().toISOString();
  
  // Veritabanını güncelle
  saveUsers(users);
  
  // Şifreyi hariç tut
  const { password, ...userWithoutPassword } = users[userIndex];
  return userWithoutPassword;
};

module.exports = {
  createUser,
  authenticateUser,
  findUserById,
  updateUserProfile
};
