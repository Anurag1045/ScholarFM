# ScholarFM — Deployment Guide

## Stack
- Frontend → Vercel (free)
- Backend  → Railway (free $5 credit/mo)
- Auth     → Firebase (free tier)

---

## Step 1 — Firebase Setup (Auth)

1. Go to https://console.firebase.google.com
2. Click "Add project" → name it "scholafm" → Continue
3. Disable Google Analytics (optional) → Create project
4. Left sidebar → "Authentication" → Get Started → Enable "Email/Password" + "Google"
5. Left sidebar → Project Settings (gear icon) → scroll to "Your apps"
6. Click "</>" (Web) → name it "scholafm-web" → Register app
7. Copy the firebaseConfig object — looks like:
   ```js
   const firebaseConfig = {
     apiKey: "AIza...",
     authDomain: "scholafm.firebaseapp.com",
     projectId: "scholafm",
     ...
   };
   ```
8. Paste it into BOTH:
   - frontend/login.html  (replace the placeholder config)
   - frontend/app.html    (replace the placeholder config)

---

## Step 2 — Backend on Railway

1. Go to https://railway.app → Sign up with GitHub
2. New Project → Deploy from GitHub repo → select this repo
3. Set root directory to: `backend`
4. Add environment variable:
   - Key: ANTHROPIC_API_KEY
   - Value: your sk-ant-... key
5. Railway auto-detects railway.toml and deploys
6. Copy your Railway URL (looks like: https://scholafm-production.up.railway.app)
7. Open frontend/app.html and replace:
   ```
   'https://YOUR_RAILWAY_URL.railway.app'
   ```
   with your actual Railway URL

---

## Step 3 — Frontend on Vercel

1. Go to https://vercel.com → Sign up with GitHub
2. New Project → Import your repo
3. Set root directory to: `frontend`
4. Framework Preset: Other
5. Click Deploy

Your site will be live at: https://your-project.vercel.app

---

## Step 4 — Add your Vercel domain to Firebase Auth

1. Firebase Console → Authentication → Settings → Authorized domains
2. Add your Vercel URL (e.g. scholafm.vercel.app)

---

## Local Development

```bash
# Backend
cd backend
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn app:app --reload --port 8000

# Frontend — just open in browser
open frontend/index.html
```

The frontend auto-detects localhost and points to http://localhost:8000.
