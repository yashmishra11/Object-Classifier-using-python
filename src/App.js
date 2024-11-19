import React, { useState } from 'react';
import './App.css';

const App = () => {
    const [image, setImage] = useState(null);
    const [preview, setPreview] = useState(null);
    const [loading, setLoading] = useState(false);
    const [vggPredictions, setVggPredictions] = useState([]);
    const [gradCamPath, setGradCamPath] = useState(null); // Grad-CAM Path
    const [info, setInfo] = useState({});
    const [errorMessage, setErrorMessage] = useState('');

    const handleImageChange = (e) => {
        const file = e.target.files[0];
        setImage(file);
        setPreview(URL.createObjectURL(file));
    };

    const handlePredict = async () => {
        if (!image) return;

        setLoading(true);
        const formData = new FormData();
        formData.append('image', image);

        try {
            const response = await fetch('http://localhost:5000/predict', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const result = await response.json();

            // Handling response structure
            setVggPredictions(result.predictions || []);
            setGradCamPath(result.grad_cam_path || null); // Save Grad-CAM path
            setInfo({
                label: result.top_label,
                summary: result.summary,
                wiki_url: result.wiki_url,
            });
        } catch (error) {
            console.error('Error fetching data:', error);
            setErrorMessage("Sorry, we couldn't process the image. Please try again.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="app-container">
            <div className="content">
                <h1>-Object Classifier-</h1>
                <input type="file" onChange={handleImageChange} />
                {preview && <img src={preview} alt="Image Preview" style={{ maxWidth: '300px' }} />}

                {loading ? (
                    <p>Analyzing image...</p>
                ) : (
                    <button onClick={handlePredict}>Submit</button>
                )}

                {vggPredictions.length > 0 && (
                    <div>
                        <h2>VGG16 Predictions:</h2>
                        {vggPredictions.map((pred, index) => (
                            <div key={index} style={{ marginBottom: '8px' }}>
                                <strong>{pred.label}</strong>
                                <div
                                    style={{
                                        backgroundColor: '#ddd',
                                        width: '100%',
                                        height: '10px',
                                        borderRadius: '4px',
                                        overflow: 'hidden',
                                    }}
                                >
                                    <div
                                        style={{
                                            backgroundColor: '#4caf50',
                                            width: `${pred.confidence * 100}%`, // Correct string interpolation
                                            height: '100%',
                                        }}
                                    ></div>
                                </div><br></br>
                                <span>{Math.round(pred.confidence * 100)}% confidence</span>
                            </div>
                        ))}
                    </div>
                )}

                {gradCamPath && (
                    <div> <br></br>
                        <h2>Grad-CAM Heatmap:</h2>
                        <img
                            src={`http://localhost:5000/${gradCamPath}`} // Ensure correct path
                            alt="Grad-CAM Heatmap"
                            style={{ maxWidth: '300px' }}
                        />
                    </div>
                )}

                {errorMessage && <p style={{ color: 'red' }}>{errorMessage}</p>}

                {info.label && (
                    <div>
                        <h3>Top Prediction: {info.label}</h3>
                        <p>{info.summary}</p>
                        {info.wiki_url && (
                            <a href={info.wiki_url} target="_blank" rel="noopener noreferrer">
                                Read more on Wikipedia
                            </a>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

export default App;
