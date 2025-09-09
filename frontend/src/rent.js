import React, { useState } from 'react';
import axios from 'axios';
import './rent.css';

function Rent() {
  const [formData, setFormData] = useState({
    first_name: '',
    last_name: '',
    location: '',
    pickup_datetime: '',
    return_datetime: '',
    vehicle_type: '',
    vehicle_number: ''
  });
  const [photos, setPhotos] = useState({
    front: null,
    back: null,
    left_side: null,
    right_side: null,
    roof: null
  });
  const [previews, setPreviews] = useState({
    front: null,
    back: null,
    left_side: null,
    right_side: null,
    roof: null
  });
  const [reference, setReference] = useState('');
  const [analysisResults, setAnalysisResults] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleFileSelect = (view) => (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
      setPhotos(prev => ({ ...prev, [view]: file }));
      
      // Create preview
      const reader = new FileReader();
      reader.onload = (e) => {
        setPreviews(prev => ({ ...prev, [view]: e.target.result }));
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDragOver = (view) => (e) => {
    e.preventDefault();
    e.currentTarget.classList.add('dragover');
  };

  const handleDragLeave = (view) => (e) => {
    e.currentTarget.classList.remove('dragover');
  };

  const handleDrop = (view) => (e) => {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('image/')) {
      setPhotos(prev => ({ ...prev, [view]: files[0] }));
      
      // Create preview
      const reader = new FileReader();
      reader.onload = (e) => {
        setPreviews(prev => ({ ...prev, [view]: e.target.result }));
      };
      reader.readAsDataURL(files[0]);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    const submitData = new FormData();
    Object.keys(formData).forEach(key => {
      submitData.append(key, formData[key]);
    });
    Object.keys(photos).forEach(view => {
      if (photos[view]) {
        submitData.append(view, photos[view]);
      }
    });

    // Check if at least one photo is uploaded
    const hasPhoto = Object.values(photos).some(photo => photo !== null);
    if (!hasPhoto) {
      setError('At least one vehicle photo must be uploaded');
      setLoading(false);
      return;
    }

    try {
      const response = await axios.post('http://localhost:5000/create_rental', submitData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      
      setReference(response.data.reference);
      setAnalysisResults(response.data.initial_damage_reports);
      console.log('Rental created:', response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Error creating rental');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setFormData({
      first_name: '',
      last_name: '',
      location: '',
      pickup_datetime: '',
      return_datetime: '',
      vehicle_type: '',
      vehicle_number: ''
    });
    setPhotos({
      front: null,
      back: null,
      left_side: null,
      right_side: null,
      roof: null
    });
    setPreviews({
      front: null,
      back: null,
      left_side: null,
      right_side: null,
      roof: null
    });
    setReference('');
    setAnalysisResults(null);
    setError('');
  };

  return (
    <div className="rent-container">
      <div className="rent-card">
        <div className="rent-header">
          <h1>🚗 Create Car Rental</h1>
          <p>Fill out the form and upload up to 5 vehicle photos to start your rental</p>
        </div>
        
        <form onSubmit={handleSubmit} className="rent-form">
          {/* Customer Information */}
          <div className="form-section">
            <h3>Customer Information</h3>
            <div className="form-row">
              <div className="form-group">
                <label>First Name</label>
                <input
                  type="text"
                  name="first_name"
                  value={formData.first_name}
                  onChange={handleInputChange}
                  required
                  placeholder="Enter first name"
                />
              </div>
              <div className="form-group">
                <label>Last Name</label>
                <input
                  type="text"
                  name="last_name"
                  value={formData.last_name}
                  onChange={handleInputChange}
                  required
                  placeholder="Enter last name"
                />
              </div>
            </div>
            <div className="form-group">
              <label>Location</label>
              <input
                type="text"
                name="location"
                value={formData.location}
                onChange={handleInputChange}
                required
                placeholder="Pickup/Return location"
              />
            </div>
          </div>

          {/* Rental Details */}
          <div className="form-section">
            <h3>Rental Details</h3>
            <div className="form-row">
              <div className="form-group">
                <label>Pickup Date & Time</label>
                <input
                  type="datetime-local"
                  name="pickup_datetime"
                  value={formData.pickup_datetime}
                  onChange={handleInputChange}
                  required
                />
              </div>
              <div className="form-group">
                <label>Return Date & Time</label>
                <input
                  type="datetime-local"
                  name="return_datetime"
                  value={formData.return_datetime}
                  onChange={handleInputChange}
                  required
                />
              </div>
            </div>
            <div className="form-row">
              <div className="form-group">
                <label>Vehicle Type</label>
                <select
                  name="vehicle_type"
                  value={formData.vehicle_type}
                  onChange={handleInputChange}
                  required
                >
                  <option value="">Select vehicle type</option>
                  <option value="Economy">BMW</option>
                  <option value="Compact">Tesla</option>
                  <option value="Mid-size">Lamborgini</option>
                  <option value="Full-size">Ford</option>
                  <option value="SUV">Pugo</option>
                  <option value="Premium">opel</option>
                </select>
              </div>
              <div className="form-group">
                <label>Vehicle Number</label>
                <input
                  type="text"
                  name="vehicle_number"
                  value={formData.vehicle_number}
                  onChange={handleInputChange}
                  required
                  placeholder="License plate number"
                />
              </div>
            </div>
          </div>

          {/* Photo Upload */}
          <div className="form-section">
            <h3>Vehicle Photos (Up to 5)</h3>
            <div className="upload-grid">
              {['front', 'back', 'left_side', 'right_side', 'roof'].map(view => (
                <div 
                  key={view}
                  className={`upload-area ${photos[view] ? 'has-file' : ''}`}
                  onDragOver={handleDragOver(view)}
                  onDragLeave={handleDragLeave(view)}
                  onDrop={handleDrop(view)}
                  onClick={() => document.getElementById(`photoInput-${view}`).click()}
                >
                  {previews[view] ? (
                    <div className="preview-container">
                      <img src={previews[view]} alt={`${view} preview`} className="preview-image" />
                      <div className="preview-overlay">
                        <p>Click or drag to change {view} photo</p>
                      </div>
                    </div>
                  ) : (
                    <>
                      <div className="upload-icon">📷</div>
                      <div className="upload-text">
                        <p>{view.replace('_', ' ').toUpperCase()}</p>
                        <p className="upload-subtext">Drag and drop or click to upload</p>
                      </div>
                    </>
                  )}
                  <input
                    type="file"
                    id={`photoInput-${view}`}
                    accept="image/*"
                    onChange={handleFileSelect(view)}
                    style={{ display: 'none' }}
                  />
                </div>
              ))}
            </div>
          </div>

          <div className="form-actions">
            <button type="button" onClick={resetForm} className="btn-secondary">
              Reset Form
            </button>
            <button 
              type="submit" 
              disabled={loading || !Object.values(photos).some(photo => photo !== null)} 
              className="btn-primary"
            >
              {loading ? (
                <>
                  <div className="spinner"></div>
                  Creating Rental...
                </>
              ) : (
                '🚀 Create Rental'
              )}
            </button>
          </div>
        </form>
        
        {error && (
          <div className="error-message">
            <strong>Error:</strong> {error}
          </div>
        )}
        
        {reference && (
          <div className="success-message">
            <h3>✅ Rental Created Successfully!</h3>
            <p>Your rental reference: <strong className="reference-code">{reference}</strong></p>
            <p className="reference-note">Please save this reference number for returning the vehicle.</p>
          </div>
        )}
        
        {/* Analysis Results */}
        {analysisResults && (
          <div className="analysis-section">
            <h3>🔍 Initial Condition Analysis</h3>
            <div className="analysis-grid">
              {Object.entries(analysisResults).map(([view, result]) => (
                <div key={view} className="analysis-card">
                  <div className="analysis-header">
                    <h4>{view.replace('_', ' ').toUpperCase()}</h4>
                    <div className={`status-badge ${result.has_damage ? 'damage' : 'clean'}`}>
                      {result.has_damage ? '⚠️ Damage Detected' : '✅ Clean Condition'}
                    </div>
                  </div>
                  
                  <div className="analysis-stats">
                    <div className="stat-item">
                      <span className="stat-value">{result.prediction_count}</span>
                      <span className="stat-label">Detections</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-value">{result.damages?.length || 0}</span>
                      <span className="stat-label">Issues Found</span>
                    </div>
                  </div>
                  
                  {result.damages && result.damages.length > 0 && (
                    <div className="damages-list">
                      <h5>Detected Issues:</h5>
                      <div className="damages-grid">
                        {result.damages.map((damage, index) => (
                          <div key={index} className="damage-item">
                            <div className="damage-type">{damage.class}</div>
                            <div className="damage-confidence">
                              {(damage.confidence * 100).toFixed(1)}% confidence
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  <div className="images-section">
                    <div className="images-grid">
                      {result.original_image && (
                        <div className="image-container">
                          <h5>Original Image</h5>
                          <img 
                            src={`http://localhost:5000${result.original_image}`} 
                            alt={`${view} original condition`}
                            className="result-image"
                          />
                        </div>
                      )}
                      {result.annotated_image && (
                        <div className="image-container">
                          <h5>Analysis Results</h5>
                          <img 
                            src={`http://localhost:5000${result.annotated_image}`} 
                            alt={`${view} analysis with detections`}
                            className={`result-image ${result.has_damage ? 'damage-border' : 'clean-border'}`}
                          />
                        </div>
                      )}
                    </div>
                  </div>
                  
                  <div className="analysis-note">
                    <p><strong>Note:</strong> {result.note}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default Rent;