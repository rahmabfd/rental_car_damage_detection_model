import React, { useState } from 'react';
import axios from 'axios';
import './return.css';

function Return() {
  const [step, setStep] = useState(1); // 1: Find Rental, 2: Upload Photos, 3: Results
  const [rentalData, setRentalData] = useState({
    reference: '',
    first_name: '',
    last_name: ''
  });
  const [rentalInfo, setRentalInfo] = useState(null);
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
  const [comparison, setComparison] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setRentalData(prev => ({
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

  const fetchRental = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const response = await axios.post('http://localhost:5000/fetch_rental', rentalData);
      setRentalInfo(response.data);
      setStep(2);
    } catch (err) {
      setError(err.response?.data?.error || 'Error fetching rental');
    } finally {
      setLoading(false);
    }
  };

  const submitReturn = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    const formData = new FormData();
    Object.keys(photos).forEach(view => {
      if (photos[view]) {
        formData.append(view, photos[view]);
      }
    });

    // Check if at least one photo is uploaded
    const hasPhoto = Object.values(photos).some(photo => photo !== null);
    if (!hasPhoto) {
      setError('At least one return photo must be uploaded');
      setLoading(false);
      return;
    }

    try {
      const response = await axios.post(`http://localhost:5000/submit_return/${rentalData.reference}`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      
      setComparison(response.data.report);
      setStep(3);
    } catch (err) {
      setError(err.response?.data?.error || 'Error submitting return');
    } finally {
      setLoading(false);
    }
  };

  const resetProcess = () => {
    setStep(1);
    setRentalData({ reference: '', first_name: '', last_name: '' });
    setRentalInfo(null);
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
    setComparison(null);
    setError('');
  };

  return (
    <div className="return-container">
      <div className="return-card">
        <div className="return-header">
          <h1>🔄 Return Vehicle</h1>
          <div className="step-indicator">
            <div className={`step ${step >= 1 ? 'active' : ''}`}>1</div>
            <div className={`step ${step >= 2 ? 'active' : ''}`}>2</div>
            <div className={`step ${step >= 3 ? 'active' : ''}`}>3</div>
          </div>
        </div>

        {error && (
          <div className="error-message">
            <strong>Error:</strong> {error}
          </div>
        )}

        {/* Step 1: Find Rental */}
        {step === 1 && (
          <div className="step-content">
            <div className="step-header">
              <h2>📋 Find Your Rental</h2>
              <p>Enter your rental details to proceed with the return</p>
            </div>

            <form onSubmit={fetchRental} className="rental-form">
              <div className="form-group">
                <label>Rental Reference</label>
                <input
                  type="text"
                  name="reference"
                  value={rentalData.reference}
                  onChange={handleInputChange}
                  required
                  placeholder="Enter rental reference (e.g., ABC12345)"
                  className="reference-input"
                />
              </div>
              
              <div className="form-row">
                <div className="form-group">
                  <label>First Name</label>
                  <input
                    type="text"
                    name="first_name"
                    value={rentalData.first_name}
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
                    value={rentalData.last_name}
                    onChange={handleInputChange}
                    required
                    placeholder="Enter last name"
                  />
                </div>
              </div>

              <button type="submit" disabled={loading} className="btn-primary">
                {loading ? (
                  <>
                    <div className="spinner"></div>
                    Searching...
                  </>
                ) : (
                  '🔍 Find Rental'
                )}
              </button>
            </form>
          </div>
        )}

        {/* Step 2: Upload Return Photos */}
        {step === 2 && rentalInfo && (
          <div className="step-content">
            <div className="step-header">
              <h2>📸 Upload Return Photos</h2>
              <p>Upload up to 5 photos of the vehicle's current condition</p>
            </div>

            {/* Rental Info Summary */}
            <div className="rental-summary">
              <h3>Rental Information</h3>
              <div className="info-grid">
                <div className="info-item">
                  <span className="info-label">Reference:</span>
                  <span className="info-value">{rentalInfo.reference}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">Customer:</span>
                  <span className="info-value">{rentalInfo.first_name} {rentalInfo.last_name}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">Vehicle:</span>
                  <span className="info-value">{rentalInfo.vehicle_type} - {rentalInfo.vehicle_number}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">Location:</span>
                  <span className="info-value">{rentalInfo.location}</span>
                </div>
              </div>
            </div>

            <form onSubmit={submitReturn} className="return-form">
              <div className="upload-grid">
                {['front', 'back', 'left_side', 'right_side', 'roof'].map(view => (
                  <div 
                    key={view}
                    className={`upload-area ${photos[view] ? 'has-file' : ''}`}
                    onDragOver={handleDragOver(view)}
                    onDragLeave={handleDragLeave(view)}
                    onDrop={handleDrop(view)}
                    onClick={() => document.getElementById(`returnPhotoInput-${view}`).click()}
                  >
                    {previews[view] ? (
                      <div className="preview-container">
                        <img src={previews[view]} alt={`${view} return preview`} className="preview-image" />
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
                      id={`returnPhotoInput-${view}`}
                      accept="image/*"
                      onChange={handleFileSelect(view)}
                      style={{ display: 'none' }}
                    />
                  </div>
                ))}
              </div>

              <div className="form-actions">
                <button type="button" onClick={() => setStep(1)} className="btn-secondary">
                  ← Back
                </button>
                <button 
                  type="submit" 
                  disabled={loading || !Object.values(photos).some(photo => photo !== null)} 
                  className="btn-primary"
                >
                  {loading ? (
                    <>
                      <div className="spinner"></div>
                      Analyzing...
                    </>
                  ) : (
                    '🔍 Process Return'
                  )}
                </button>
              </div>
            </form>
          </div>
        )}

        {/* Step 3: Comparison Results */}
        {step === 3 && comparison && (
          <div className="step-content">
            <div className="step-header">
              <h2>📊 Return Analysis Complete</h2>
              <div className={`damage-status ${comparison.summary.has_any_new_damage ? 'damage' : 'clean'}`}>
                {comparison.summary.has_any_new_damage ? '⚠️ New Damage Detected' : '✅ No New Damage Found'}
              </div>
            </div>

            {/* Damage Summary */}
            <div className="damage-summary">
              <div className="summary-stats">
                <div className="stat-card">
                  <div className="stat-number">{comparison.summary.total_views_compared}</div>
                  <div className="stat-label">Views Compared</div>
                </div>
                <div className="stat-card">
                  <div className="stat-number">{comparison.summary.total_new_damages}</div>
                  <div className="stat-label">New Damages</div>
                </div>
                <div className="stat-card">
                  <div className="stat-number">{comparison.summary.damage_severity}</div>
                  <div className="stat-label">Damage Severity</div>
                </div>
              </div>
              <p className="summary-note">{comparison.summary.note}</p>
            </div>

            {/* Comparison Results by View */}
            <div className="comparison-section">
              <h3>📷 Comparison by View</h3>
              <div className="comparison-grid">
                {Object.entries(comparison.views_comparison).map(([view, data]) => (
                  <div key={view} className="comparison-card">
                    <div className="comparison-header">
                      <h4>{view.replace('_', ' ').toUpperCase()}</h4>
                      <div className={`status-badge ${data.comparison_status === 'new_damage_detected' ? 'damage' : data.comparison_status === 'no_new_damage' ? 'clean' : 'missing'}`}>
                        {data.comparison_status === 'new_damage_detected' ? '⚠️ New Damage' :
                         data.comparison_status === 'no_new_damage' ? '✅ No New Damage' :
                         data.comparison_status === 'initial_missing' ? '⚠️ Initial Missing' :
                         data.comparison_status === 'return_missing' ? '⚠️ Return Missing' : '⚠️ No Images'}
                      </div>
                    </div>
                    
                    <div className="comparison-stats">
                      <div className="stat-item">
                        <span className="stat-windows">{data.damage_count.initial}</span>
                        <span className="stat-label">Initial Issues</span>
                      </div>
                      <div className="stat-item">
                        <span className="stat-value">{data.damage_count.return}</span>
                        <span className="stat-label">Return Issues</span>
                      </div>
                      <div className="stat-item">
                        <span className="stat-value">{data.damage_count.new}</span>
                        <span className="stat-label">New Damages</span>
                      </div>
                    </div>
                    
                    {data.new_damages.length > 0 && (
                      <div className="new-damages">
                        <h5>New Damages:</h5>
                        <div className="damages-grid">
                          {data.new_damages.map((damage, index) => (
                            <div key={index} className="damage-item new-damage">
                              <div className="damage-type">{damage.class}</div>
                              <div className="damage-confidence">
                                {(damage.confidence * 100).toFixed(1)}% confidence
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    
                    <div className="images-comparison">
                      {data.initial_annotated_image && (
                        <div className="image-container">
                          <h5>Initial Condition</h5>
                          <img 
                            src={`http://localhost:5000${data.initial_annotated_image}`}
                            alt={`${view} initial condition`}
                            className="comparison-image initial"
                          />
                          <div className="image-info">
                            <span>{data.damage_count.initial} issues found</span>
                          </div>
                        </div>
                      )}
                      {data.return_annotated_image && (
                        <div className="image-container">
                          <h5>Return Condition</h5>
                          <img 
                            src={`http://localhost:5000${data.return_annotated_image}`}
                            alt={`${view} return condition`}
                            className={`comparison-image return ${data.has_new_damage ? 'damage-border' : 'clean-border'}`}
                          />
                          <div className="image-info">
                            <span>{data.damage_count.return} issues found</span>
                          </div>
                        </div>
                      )}
                    </div>
                    
                    <div className="comparison-note">
                      <p><strong>Note:</strong> {data.note}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="return-actions">
              <button onClick={resetProcess} className="btn-secondary">
                🔄 Process Another Return
              </button>
              <button 
                onClick={() => window.print()} 
                className="btn-primary"
              >
                🖨️ Print Report
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default Return;