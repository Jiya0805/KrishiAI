# KrishiAI Setup Guide

## 🚀 Complete Feature List

### ✅ Backend Features
- **ML Models Integration**: Uses actual trained models from `backend/ml_models/`
- **SHAP Explanations**: Visual feature importance for crop predictions
- **Profit Calculator**: Calculate farming profitability
- **Gemini AI Integration**: Cultivation roadmap generation
- **AI Chatbot**: Farming advice via Gemini API
- **PDF Generation**: Download prediction reports
- **REST API**: Complete Django REST framework setup

### ✅ Frontend Features
- **Firebase Authentication**: Email/password + Google login
- **Protected Routes**: Login required to access features
- **Enhanced Crop Prediction**: 4 options after prediction
- **SHAP Visualizations**: Feature importance charts
- **Floating AI Chatbot**: Bottom-right chat interface
- **Modern UI**: Tailwind CSS responsive design

## 🔧 Setup Instructions

### 1. Backend Setup

```bash
cd backend
pip install -r requirements.txt
cd core
python manage.py migrate
```

### 2. Environment Variables

Create `backend/core/.env`:
```
GEMINI_API_KEY=your-gemini-api-key-here
```

Get your Gemini API key from: https://makersuite.google.com/app/apikey

### 3. Frontend Setup

```bash
cd frontend/app
npm install
```

Create `frontend/app/.env`:
```
REACT_APP_FIREBASE_API_KEY=your-firebase-api-key
REACT_APP_FIREBASE_AUTH_DOMAIN=your-project.firebaseapp.com
REACT_APP_FIREBASE_PROJECT_ID=your-project-id
REACT_APP_FIREBASE_STORAGE_BUCKET=your-project.appspot.com
REACT_APP_FIREBASE_MESSAGING_SENDER_ID=123456789
REACT_APP_FIREBASE_APP_ID=your-app-id
REACT_APP_GEMINI_API_KEY=your-gemini-api-key-here
```

### 4. Firebase Configuration

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Create a new project
3. Enable Authentication:
   - Go to Authentication > Sign-in method
   - Enable Email/Password
   - Enable Google
4. Get your config from Project Settings > General > Your apps
5. Replace values in `.env` file

### 5. Run the Application

**Backend:**
```bash
cd backend/core
python manage.py runserver
```

**Frontend:**
```bash
cd frontend/app
npm start
```

## 🎯 Application Flow

1. **Login/Signup**: Firebase authentication required
2. **Main Dashboard**: Access to all ML features
3. **Crop Prediction**: 
   - Input soil parameters
   - Get SHAP explanations
   - 4 options: Profit calc, Roadmap, PDF, Back
4. **Disease/Pest Detection**: Upload images for analysis
5. **AI Chatbot**: Always available for farming advice

## 📊 Crop Prediction Enhanced Flow

After getting a crop recommendation:

### 1. Profit Calculation 💰
- Input: Area, cost per acre, yield per acre, market price
- Output: Total profit, profit margin, detailed breakdown

### 2. Cultivation Roadmap 🗺️
- Input: Location, season
- Output: AI-generated step-by-step cultivation guide

### 3. Save & Download 📄
- Generates PDF report with prediction details
- Includes SHAP explanations and input parameters

### 4. Back ⬅️
- Return to crop recommendation form

## 🤖 AI Chatbot Features

- **Floating Interface**: Bottom-right corner
- **Farming Context**: Specialized for agricultural advice
- **Real-time Chat**: Powered by Gemini AI
- **Always Available**: Accessible from any page

## 🔒 Authentication Features

- **Email/Password**: Standard authentication
- **Google Sign-in**: One-click login
- **Protected Routes**: Must login to access features
- **Session Management**: Automatic login state handling

## 📱 Responsive Design

- **Mobile-first**: Works on all device sizes
- **Modern UI**: Clean, professional interface
- **Tailwind CSS**: Utility-first styling
- **Interactive Elements**: Hover effects, loading states

## 🛠️ Technical Stack

**Backend:**
- Django 4.2.7
- Django REST Framework
- TensorFlow 2.15.0
- SHAP 0.43.0
- Google Generative AI
- ReportLab (PDF generation)

**Frontend:**
- React 19.1.1
- Firebase 10.7.1
- Tailwind CSS 3.3.6
- Axios 1.6.2

## 🚨 Important Notes

1. **Firebase Config**: Must configure with your actual Firebase project
2. **Gemini API**: Need valid API key for AI features
3. **ML Models**: Ensure trained models exist in `backend/ml_models/`
4. **CORS**: Already configured for localhost:3000
5. **Authentication**: Required for all main features

## 🎉 Ready to Use!

Your KrishiAI application is now fully configured with:
- ✅ Firebase authentication
- ✅ ML model predictions with SHAP explanations
- ✅ Profit calculations
- ✅ AI-powered cultivation roadmaps
- ✅ PDF report generation
- ✅ Floating AI chatbot
- ✅ Modern responsive UI

Access the app at:
- **Backend**: http://localhost:8000
- **Frontend**: http://localhost:3000
