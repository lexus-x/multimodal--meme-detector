import React, { useState, useRef } from 'react';
import axios from 'axios';
import { Upload, Shield, ShieldAlert, Cpu, Layers, Info, Trash2, Send } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const API_URL = import.meta.env.VITE_API_URL || `http://${window.location.hostname}:8000`;

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [text, setText] = useState('');
  const [fusion, setFusion] = useState('cross_attention');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
      setResult(null);
    }
  };

  const handlePredict = async () => {
    if (!image || !text) {
      setError("Please provide both an image and the meme text.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('image', image);
    formData.append('text', text);
    formData.append('fusion', fusion);

    try {
      const response = await axios.post(`${API_URL}/predict`, formData);
      setResult(response.data);
    } catch (err) {
      console.error(err);
      setError("Failed to process meme. Make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setImage(null);
    setPreview(null);
    setText('');
    setResult(null);
    setError(null);
  };

  return (
    <div className="app-container">
      <header className="header">
        <motion.h1 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          MemeGuardian AI
        </motion.h1>
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
        >
          Advanced Multimodal Offensive Content Detection
        </motion.p>
      </header>

      <main className="main-grid">
        {/* Input Column */}
        <motion.div 
          className="card"
          initial={{ opacity: 0, x: -50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
        >
          <div 
            className="upload-area"
            onClick={() => fileInputRef.current?.click()}
          >
            {preview ? (
              <img src={preview} alt="Meme Preview" />
            ) : (
              <div style={{ color: 'var(--text-muted)' }}>
                <Upload size={48} style={{ marginBottom: '1rem' }} />
                <p>Click to upload or drag meme image</p>
              </div>
            )}
            <input 
              type="file" 
              ref={fileInputRef} 
              style={{ display: 'none' }} 
              accept="image/*"
              onChange={handleFileChange}
            />
            {preview && (
              <button 
                className="feature-tag" 
                style={{ position: 'absolute', top: '10px', right: '10px', background: 'rgba(0,0,0,0.5)', cursor: 'pointer' }}
                onClick={(e) => { e.stopPropagation(); reset(); }}
              >
                <Trash2 size={14} style={{ marginRight: '4px' }} /> Change
              </button>
            )}
          </div>

          <div className="input-group" style={{ marginTop: '1.5rem' }}>
            <label>Meme Text Content</label>
            <textarea 
              rows="3" 
              placeholder="Type the text seen in the meme..."
              value={text}
              onChange={(e) => setText(e.target.value)}
            ></textarea>
          </div>

          <div className="input-group">
            <label>Fusion Architecture</label>
            <select value={fusion} onChange={(e) => setFusion(e.target.value)}>
              <option value="cross_attention">Cross-Modal Attention (Recommended)</option>
              <option value="early">Early Fusion (Baseline)</option>
              <option value="gated">Gated Fusion (ModDrop)</option>
              <option value="bilinear">Bilinear Pooling</option>
            </select>
          </div>

          <button 
            className="btn-primary" 
            onClick={handlePredict}
            disabled={loading || !image || !text}
          >
            {loading ? <div className="loader"></div> : <><Send size={20} /> Analyze Meme</>}
          </button>

          {error && <p style={{ color: 'var(--accent-red)', marginTop: '1rem', textAlign: 'center' }}>{error}</p>}
        </motion.div>

        {/* Result Column */}
        <motion.div 
          className="card"
          initial={{ opacity: 0, x: 50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.4 }}
        >
          <AnimatePresence mode="wait">
            {!result ? (
              <motion.div 
                key="placeholder"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-muted)' }}
              >
                <Shield size={64} style={{ opacity: 0.2, marginBottom: '1rem' }} />
                <p>Upload a meme and text to start analysis</p>
                <div style={{ marginTop: '2rem', display: 'flex', flexWrap: 'wrap', justifyContent: 'center' }}>
                  <span className="feature-tag"><Cpu size={14} style={{ marginRight: '4px' }} /> BiLSTM Text Encoder</span>
                  <span className="feature-tag"><Layers size={14} style={{ marginRight: '4px' }} /> VGG16 Visual Backbone</span>
                  <span className="feature-tag"><Info size={14} style={{ marginRight: '4px' }} /> MultiOFF Dataset</span>
                </div>
              </motion.div>
            ) : (
              <motion.div 
                key="result"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
              >
                <div style={{ textAlign: 'center' }}>
                  <div className={`result-badge ${result.label === 'Offensive' ? 'badge-offensive' : 'badge-clean'}`}>
                    {result.label === 'Offensive' ? <ShieldAlert size={24} style={{ marginRight: '8px', verticalAlign: 'middle' }} /> : <Shield size={24} style={{ marginRight: '8px', verticalAlign: 'middle' }} />}
                    {result.label.toUpperCase()}
                  </div>
                </div>

                <div className="score-row">
                  <div className="score-header">
                    <span>Offensive Probability</span>
                    <span>{(result.offensive_prob * 100).toFixed(1)}%</span>
                  </div>
                  <div className="progress-bar">
                    <motion.div 
                      className="progress-fill progress-offensive"
                      initial={{ width: 0 }}
                      animate={{ width: `${result.offensive_prob * 100}%` }}
                    ></motion.div>
                  </div>
                </div>

                <div className="score-row">
                  <div className="score-header">
                    <span>Clean Probability</span>
                    <span>{(result.non_offensive_prob * 100).toFixed(1)}%</span>
                  </div>
                  <div className="progress-bar">
                    <motion.div 
                      className="progress-fill progress-clean"
                      initial={{ width: 0 }}
                      animate={{ width: `${result.non_offensive_prob * 100}%` }}
                    ></motion.div>
                  </div>
                </div>

                <div style={{ marginTop: '2rem', padding: '1rem', background: 'rgba(255,255,255,0.03)', borderRadius: '1rem', border: '1px solid var(--border)' }}>
                  <h4 style={{ marginBottom: '0.5rem', display: 'flex', alignItems: 'center' }}>
                    <Info size={16} style={{ marginRight: '8px' }} /> Insights
                  </h4>
                  <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)', lineHeight: '1.5' }}>
                    {result.label === 'Offensive' 
                      ? "The model detected potentially harmful patterns in the combination of image features and text sentiment. Higher attention weights were likely assigned to provocative keywords or visual cues."
                      : "The model found no strong evidence of offensive intent. The relationship between the visual elements and the text appears benign based on the training data."}
                  </p>
                </div>

                <div style={{ marginTop: '1.5rem', fontSize: '0.8rem', color: 'var(--text-muted)', textAlign: 'center' }}>
                  Processed using <code>{fusion}</code> fusion strategy
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </main>

      <footer style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
        Built with PyTorch & React • Powered by MemeDetector Engine
      </footer>
    </div>
  );
}

export default App;
