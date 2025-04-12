# Brain Tumor Detection Application

A web application for detecting brain tumors in medical images using AI.

## Features

- User authentication
- Image upload and processing
- AI-powered brain tumor detection
- Image history tracking
- Starred images functionality
- Dual storage (MongoDB and local repository)

## Prerequisites

- Node.js (v14 or higher)
- Python (v3.8 or higher)
- MongoDB Atlas account
- Google Gemini API key

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - Unix/MacOS:
     ```bash
     source venv/bin/activate
     ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Create a `.env` file:
   - Copy `.env.example` to `.env`
   - Fill in your actual values for:
     - SECRET_KEY
     - MONGO_URI
     - GEMINI_API_KEY

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

## Running the Application

1. Start the backend server:
   ```bash
   cd backend
   python app.py
   ```

2. Start the frontend development server:
   ```bash
   cd frontend
   npm start
   ```

3. Access the application at `http://localhost:3000`

## Project Structure

```
.
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   ├── .env
│   ├── .env.example
│   ├── uploads/
│   └── starred_images/
├── frontend/
│   ├── src/
│   ├── public/
│   └── package.json
└── README.md
```

## Security Notes

- Never commit `.env` files to version control
- Keep your API keys and secrets secure
- Use environment variables for sensitive data

## License

[Your chosen license] 