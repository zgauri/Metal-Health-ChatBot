const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');

const app = express();
const PORT = 5000;

// Middleware
app.use(bodyParser.json());
app.use(cors());

// Dummy logic to categorize user messages
function categorizeMessage(message) {
    message = message.toLowerCase();

    if (message.includes('suicide') || message.includes('kill myself') || message.includes('want to die')) {
        return { category: 'High Risk - Suicidal' };
    } else if (message.includes('depressed') || message.includes('anxious') || message.includes('hopeless')) {
        return { category: 'Moderate Risk - Signs of Depression' };
    } else if (message.includes('sad') || message.includes('stress') || message.includes('lonely')) {
        return { category: 'Low Risk - Manageable Emotional Range' };
    } else {
        return { category: 'General - No Significant Risk Detected' };
    }
}

// API endpoint
app.post('/chat', (req, res) => {
    const { message } = req.body;

    if (!message) {
        return res.status(400).json({ error: 'Message is required' });
    }

    const response = categorizeMessage(message);
    res.json(response);
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running on http://127.0.0.1:${PORT}`);
});