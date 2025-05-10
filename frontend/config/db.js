const mongoose = require('mongoose');
require('dotenv').config();

const connectDB = async () => {
  try {
    // DB Bağlantı ayarları
    const options = {
      useNewUrlParser: true,
      useUnifiedTopology: true,
      serverSelectionTimeoutMS: 5000, // Sunucu seçimi için timeout süresi
      socketTimeoutMS: 45000,       // Socket timeout
      family: 4                     // IPv4'e zorla
    };

    const conn = await mongoose.connect(process.env.MONGODB_URI, options);
    
    console.log(`MongoDB bağlantısı başarılı: ${conn.connection.host}`);
    return conn;
  } catch (error) {
    console.error(`MongoDB bağlantı hatası: ${error.message}`);
    console.error('Lütfen .env dosyasındaki MONGODB_URI değişkeninin doğru olduğundan emin olun');
    console.error('Mevcut URI:', process.env.MONGODB_URI);
    // Kritik hata, uygulama çıkışı
    process.exit(1);
  }
};

module.exports = connectDB;
