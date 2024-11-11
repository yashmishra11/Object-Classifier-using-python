import React, { useState } from 'react';

function App() {
    const [image, setImage] = useState(null);
    const [predictions, setPredictions] = useState([]);
    const [info, setInfo] = useState(null);

    const handleImageUpload = (event) => {
        setImage(event.target.files[0]);
    };

    const handlePredict = async () => {
        if (!image) return;

        const formData = new FormData();
        formData.append('image', image);

        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        setPredictions(result.predictions);
        setInfo({
            label: result.top_label,
            summary: result.summary,
            wiki_url: result.wiki_url
        });
    };

    return (
        <div style={{ textAlign: 'center' }}>
            <h1>Object Identification with VGG16</h1>
            <input type="file" onChange={handleImageUpload} />
            <button onClick={handlePredict}>Predict</button>

            {predictions.length > 0 && (
                <div>
                    <h2>Predictions:</h2>
                    <ul>
                        {predictions.map((pred, idx) => (
                            <li key={idx}>{pred.label}: {Math.round(pred.confidence * 100)}% confidence</li>
                        ))}
                    </ul>
                </div>
            )}

            {info && (
                <div>
                    <h3>More about {info.label}:</h3>
                    <p>{info.summary}</p>
                    {info.wiki_url && <a href={info.wiki_url} target="_blank" rel="noopener noreferrer">Read more on Wikipedia</a>}
                </div>
            )}
        </div>
    );
}

export default App;