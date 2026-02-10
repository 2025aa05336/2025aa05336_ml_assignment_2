# GitHub Upload Guide - ML Classification Assignment

## 📋 **Prerequisites**
- GitHub account (create at https://github.com if you don't have one)
- Git installed on your computer
- Your project files ready (which you already have!)

## 🚀 **Step-by-Step Upload Process**

### **Step 1: Create GitHub Repository**
1. Go to https://github.com
2. Click the **"+"** icon in top right corner
3. Select **"New repository"**
4. Repository settings:
   - **Repository name**: `ml-classification-assignment2` (or any name you prefer)
   - **Description**: `Machine Learning Classification Models Comparison - BITS Pilani Assignment 2`
   - **Visibility**: ✅ **PUBLIC** (required for Streamlit Cloud)
   - **Initialize**: ❌ Don't add README, .gitignore, or license (we have our files)
5. Click **"Create repository"**

### **Step 2: Prepare Your Local Project**
Open Terminal/Command Prompt and navigate to your project folder:
```bash
cd /Users/bhawanip/Downloads/AIML_materials/Rem_assignments/IML
```

### **Step 3: Initialize Git Repository**
```bash
# Initialize git repository
git init

# Add all project files
git add .

# Create initial commit
git commit -m "Initial commit: ML Classification Assignment 2"
```

### **Step 4: Connect to GitHub**
Replace `YOUR_USERNAME` with your actual GitHub username:
```bash
# Add remote repository (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/ml-classification-assignment2.git

# Push files to GitHub
git push -u origin main
```

### **Step 5: Verify Upload**
1. Go to your GitHub repository URL
2. Verify all files are present:
   - ✅ `app.py`
   - ✅ `requirements.txt`
   - ✅ `README.md`
   - ✅ `model/` folder
   - ✅ `test_models.py`

## 🌐 **Deploy to Streamlit Cloud**

### **Step 1: Access Streamlit Cloud**
1. Go to https://streamlit.io/cloud
2. Sign in with your GitHub account

### **Step 2: Create New App**
1. Click **"New app"**
2. Connect your GitHub account if prompted
3. Select your repository: `ml-classification-assignment2`
4. Branch: `main`
5. Main file path: `app.py`
6. App URL: Choose a custom name or use default

### **Step 3: Deploy**
1. Click **"Deploy!"**
2. Wait for deployment (usually 2-5 minutes)
3. Your app will be live at: `https://YOUR_APP_NAME.streamlit.app`

## 📝 **For Assignment Submission**

After successful deployment, you'll have:

1. **GitHub Repository Link**: 
   `https://github.com/YOUR_USERNAME/ml-classification-assignment2`

2. **Live Streamlit App Link**: 
   `https://YOUR_APP_NAME.streamlit.app`

## ⚠️ **Troubleshooting Tips**

### **If Git Commands Don't Work:**
1. Install Git: https://git-scm.com/downloads
2. Configure Git:
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```

### **If GitHub Push Fails:**
1. Check if repository name matches exactly
2. Use GitHub Personal Access Token instead of password
3. Or use GitHub Desktop app for easier management

### **If Streamlit Deployment Fails:**
1. Check if repository is PUBLIC
2. Verify `requirements.txt` is present
3. Make sure `app.py` is in root directory

## 🎯 **Quick Commands Summary**
```bash
# Navigate to project
cd /Users/bhawanip/Downloads/AIML_materials/Rem_assignments/IML

# Git commands
git init
git add .
git commit -m "Initial commit: ML Assignment 2"
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
git push -u origin main
```

That's it! Your project will be live and accessible to anyone with the links.
