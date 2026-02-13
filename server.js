const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const fs = require('fs');
const path = require('path');
const multer = require('multer');
const archiver = require('archiver');
const { kml } = require('@tmcw/togeojson');
const { DOMParser } = require('xmldom');

const { exec } = require('child_process');
const util = require('util');
const execPromise = util.promisify(exec);

const app = express();
const PORT = process.env.PORT || 3001;

// Define directories first
const DATA_DIR = path.join(__dirname, 'data');
const UPLOADS_DIR = path.join(DATA_DIR, 'uploads');
const DATA_FILE = path.join(DATA_DIR, 'drawn_data.json');
const PIPELINE_DIR = path.join(__dirname, 'pipeline');

// Ensure directories exist
if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR);
if (!fs.existsSync(UPLOADS_DIR)) fs.mkdirSync(UPLOADS_DIR);
if (!fs.existsSync(PIPELINE_DIR)) fs.mkdirSync(PIPELINE_DIR);
if (!fs.existsSync(DATA_FILE)) fs.writeFileSync(DATA_FILE, JSON.stringify([]));

app.use(cors());
app.use(bodyParser.json({ limit: '50mb' }));

// Health check endpoint for Railway
app.get('/health', (req, res) => {
    res.status(200).json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Helper function to convert GeoJSON to KML
function geojsonToKml(features, name) {
    let kml = `<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>${name}</name>
    <Style id="defaultStyle">
      <PolyStyle>
        <colorMode>normal</colorMode>
        <fill>0</fill>
        <outline>1</outline>
      </PolyStyle>
    </Style>`;

    features.forEach((feature, index) => {
        const geom = feature.geometry;
        const props = feature.properties || {};
        const featName = props.name || `Feature ${index + 1}`;
        
        kml += `
    <Placemark>
      <name>${featName}</name>
      <styleUrl>#defaultStyle</styleUrl>`;

        if (geom.type === 'Point') {
            kml += `
      <Point>
        <coordinates>${geom.coordinates[0]},${geom.coordinates[1]},0</coordinates>
      </Point>`;
        } else if (geom.type === 'LineString') {
            const coords = geom.coordinates.map(c => `${c[0]},${c[1]},0`).join(' ');
            kml += `
      <LineString>
        <tessellate>1</tessellate>
        <coordinates>${coords}</coordinates>
      </LineString>`;
        } else if (geom.type === 'Polygon') {
            const outerCoords = geom.coordinates[0].map(c => `${c[0]},${c[1]},0`).join(' ');
            kml += `
      <Polygon>
        <tessellate>1</tessellate>
        <outerBoundaryIs>
          <LinearRing>
            <coordinates>${outerCoords}</coordinates>
          </LinearRing>
        </outerBoundaryIs>`;
            
            if (geom.coordinates.length > 1) {
                for (let i = 1; i < geom.coordinates.length; i++) {
                    const innerCoords = geom.coordinates[i].map(c => `${c[0]},${c[1]},0`).join(' ');
                    kml += `
        <innerBoundaryIs>
          <LinearRing>
            <coordinates>${innerCoords}</coordinates>
          </LinearRing>
        </innerBoundaryIs>`;
                }
            }
            kml += `
      </Polygon>`;
        }

        kml += `
    </Placemark>`;
    });

    kml += `
  </Document>
</kml>`;
    return kml;
}

// Helper function to process data with Python script
async function processWithPython(metadata, kmlContent) {
    try {
        const kmlCreationDir = path.join(__dirname, 'kml_creation');
        const inputKmlPath = path.join(kmlCreationDir, 'input.kml');
        const pythonScriptPath = path.join(kmlCreationDir, 'KML_creation.py');
        const pythonExePath = 'python'; 
        
        if (!fs.existsSync(kmlCreationDir)) {
            fs.mkdirSync(kmlCreationDir, { recursive: true });
        }

        fs.writeFileSync(inputKmlPath, kmlContent);
        
        const chainageStart = parseFloat(metadata.chainage) || 0;
        const interval = 5; 
        const laneCount = parseInt(metadata.laneCount) || 4;
        const mergeOffset = parseFloat(metadata.kmlMergeOffset) || 0.100;
        const laneStep = 3.4; 
        const medianOffset = parseFloat(metadata.offsetType) || 2.75;

        const command = `"${pythonExePath}" "${pythonScriptPath}" "${inputKmlPath}" "${PIPELINE_DIR}" ${chainageStart} ${interval} ${laneCount} ${mergeOffset} ${laneStep} ${medianOffset}`;
        const { stdout, stderr } = await execPromise(command);
        
        if (stderr) console.warn(`Python script warning/error: ${stderr}`);
        console.log(`Python script output: ${stdout}`);

        return true;
    } catch (error) {
        console.error('Error in processWithPython:', error);
        throw error;
    }
}

// Helper function to save data to the pipeline folder
async function saveToPipeline(metadata, content, isKmlContent = false) {
    try {
        let kmlContent = isKmlContent ? content : geojsonToKml(content, 'Drawn_Data');
        await processWithPython(metadata, kmlContent);
        return 'Merge_KMLs';
    } catch (error) {
        console.error('Error saving to pipeline:', error);
        return null;
    }
}

// Helper function to trigger pipeline from data file
async function triggerPipelineFromDataFile() {
    try {
        if (!fs.existsSync(DATA_FILE)) return;
        const fileContent = fs.readFileSync(DATA_FILE, 'utf8');
        const data = JSON.parse(fileContent);
        if (data && data.length > 0) {
            const entry = data[0];
            if (entry.metadata && entry.geometry) {
                const kmlContent = geojsonToKml(entry.geometry, 'Drawn_Data');
                await processWithPython(entry.metadata, kmlContent);
            }
        }
    } catch (error) {
        console.error('Error triggering pipeline from data file:', error);
    }
}

// Watch for changes in drawn_data.json
let watchTimeout;
fs.watch(DATA_DIR, (eventType, filename) => {
    if (filename === 'drawn_data.json') {
        if (watchTimeout) clearTimeout(watchTimeout);
        watchTimeout = setTimeout(() => {
            triggerPipelineFromDataFile();
        }, 1000); 
    }
});

// Routes
app.get('/download-folder', (req, res) => {
    const folderPath = req.query.path || '';
    
    try {
        const targetPath = path.resolve(PIPELINE_DIR, folderPath);

        if (!targetPath.startsWith(PIPELINE_DIR)) {
            return res.status(403).json({ success: false, message: 'Access denied' });
        }

        if (!fs.existsSync(targetPath) || !fs.statSync(targetPath).isDirectory()) {
            return res.status(404).json({ success: false, message: 'Folder not found' });
        }

        const folderName = path.basename(targetPath) || 'pipeline';
        res.attachment(`${folderName}.zip`);

        const archive = archiver('zip', { zlib: { level: 9 } });
        archive.on('error', (err) => { throw err; });
        archive.pipe(res);
        archive.directory(targetPath, false);
        archive.finalize();
    } catch (error) {
        console.error('Error zipping folder:', error);
        if (!res.headersSent) {
            res.status(500).json({ success: false, message: 'Error zipping folder' });
        }
    }
});

app.use('/pipeline-files', express.static(PIPELINE_DIR));

app.get('/pipeline-folders', (req, res) => {
    try {
        const subPath = req.query.path || '';
        const currentPath = path.resolve(PIPELINE_DIR, subPath);
        
        if (!currentPath.startsWith(PIPELINE_DIR)) {
            return res.status(403).json({ success: false, message: 'Access denied' });
        }

        if (!fs.existsSync(currentPath)) {
            return res.json({ success: true, items: [], currentPath: subPath });
        }

        const items = fs.readdirSync(currentPath, { withFileTypes: true });
        const contents = items.map(item => {
            const itemPath = path.join(subPath, item.name).replace(/\\/g, '/');
            const stats = fs.statSync(path.join(currentPath, item.name));
            return {
                name: item.name,
                type: item.isDirectory() ? 'folder' : 'file',
                path: itemPath,
                modifiedAt: stats.mtime
            };
        });
        
        contents.sort((a, b) => {
            if (a.type === 'folder' && b.type !== 'folder') return -1;
            if (a.type !== 'folder' && b.type === 'folder') return 1;
            return new Date(b.modifiedAt) - new Date(a.modifiedAt);
        });
        
        res.json({ success: true, items: contents, currentPath: subPath });
    } catch (error) {
        res.status(500).json({ success: false, message: 'Error listing folders' });
    }
});

const storage = multer.diskStorage({
    destination: (req, file, cb) => cb(null, UPLOADS_DIR),
    filename: (req, file, cb) => cb(null, file.originalname)
});
const upload = multer({ storage: storage });

app.post('/upload-kml', upload.single('kmlFile'), async (req, res) => {
    try {
        if (!req.file) return res.status(400).json({ success: false, message: 'No file uploaded' });
        const kmlContent = fs.readFileSync(req.file.path, 'utf8');
        const kmlDom = new DOMParser().parseFromString(kmlContent);
        const geoJson = kml(kmlDom);

        const kmlData = {
            metadata: {
                fileName: req.file.originalname,
                type: 'KML_UPLOAD',
                chainage: req.body.chainage || '',
                offsetType: req.body.offsetType || '',
                laneCount: req.body.laneCount || '',
                kmlMergeOffset: req.body.kmlMergeOffset || ''
            },
            geometry: geoJson.features,
            filePath: req.file.path,
            id: Date.now(),
            timestamp: new Date().toISOString()
        };

        fs.writeFileSync(DATA_FILE, JSON.stringify([kmlData], null, 2));
        await saveToPipeline(kmlData.metadata, kmlContent, true);
        res.json({ success: true, message: 'File uploaded and processed successfully', pipelinePath: 'Merge_KMLs', data: kmlData });
    } catch (error) {
        res.status(500).json({ success: false, message: 'Error uploading file' });
    }
});

app.post('/save', async (req, res) => {
    try {
        const newData = req.body;
        newData.id = Date.now();
        newData.timestamp = new Date().toISOString();
        fs.writeFileSync(DATA_FILE, JSON.stringify([newData], null, 2));
        const pipelinePath = await saveToPipeline(newData.metadata, newData.geometry, false);
        res.json({ success: true, message: 'Data saved successfully', id: newData.id, pipelinePath: pipelinePath });
    } catch (error) {
        res.status(500).json({ success: false, message: 'Error saving data' });
    }
});

app.post('/clear-all', async (req, res) => {
    try {
        fs.writeFileSync(DATA_FILE, JSON.stringify([], null, 2));
        if (fs.existsSync(UPLOADS_DIR)) {
            fs.readdirSync(UPLOADS_DIR).forEach(file => fs.unlinkSync(path.join(UPLOADS_DIR, file)));
        }
        ['LHS_KMLs', 'RHS_KMLs', 'Excels', 'Merge_KMLs'].forEach(sub => {
            const subPath = path.join(PIPELINE_DIR, sub);
            if (fs.existsSync(subPath)) {
                fs.readdirSync(subPath).forEach(item => {
                    const itemPath = path.join(subPath, item);
                    if (fs.statSync(itemPath).isDirectory()) fs.rmSync(itemPath, { recursive: true, force: true });
                    else fs.unlinkSync(itemPath);
                });
            }
        });
        res.json({ success: true, message: 'All data and pipeline files cleared successfully' });
    } catch (error) {
        res.status(500).json({ success: false, message: 'Error clearing all data' });
    }
});

app.get('/data', (req, res) => {
    try {
        res.json(JSON.parse(fs.readFileSync(DATA_FILE, 'utf8')));
    } catch (error) {
        res.status(500).json({ success: false, message: 'Error reading data' });
    }
});

if (process.env.NODE_ENV === 'production') {
    const frontendBuildPath = path.join(__dirname, '../frontend/build');

    // Serve static files
    app.use(express.static(frontendBuildPath));

    // Catch-all route for SPA (React, Vue, etc.)
    app.get('/*', (req, res) => {
        res.sendFile(path.join(frontendBuildPath, 'index.html'));
    });
}


app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server is running on port ${PORT}`);
});
