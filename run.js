// Otomatik backend (Flask) ve frontend (Node.js) başlatıcı
// node run.js ile hem Flask API hem de frontend backend başlatılır

const { spawn } = require('child_process');
const path = require('path');

// Flask API başlat
const flaskProcess = spawn('/Users/alpsalcioglu/miniconda3/bin/python3', ['basit_api.py'], {
    cwd: path.join(__dirname, 'api'),
    stdio: 'inherit',
    shell: true
});

// Frontend backend başlat
const nodeProcess = spawn('node', ['app.js'], {
    cwd: path.join(__dirname, 'frontend'),
    stdio: 'inherit',
    shell: true
});

// Flask veya Node.js kapanırsa ana process de kapansın
defineExitHandler(flaskProcess, 'Flask (basit_api.py)');
defineExitHandler(nodeProcess, 'Node.js (app.js)');

function defineExitHandler(proc, name) {
    proc.on('exit', (code) => {
        console.log(`${name} process exited with code ${code}`);
        process.exit(code);
    });
    proc.on('error', (err) => {
        console.error(`${name} process error:`, err);
        process.exit(1);
    });
}

// Ana process kapatılırsa alt processleri de öldür
process.on('SIGINT', () => {
    flaskProcess.kill('SIGINT');
    nodeProcess.kill('SIGINT');
    process.exit();
});
