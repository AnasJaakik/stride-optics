# Deploying StrideOptics to Railway

Railway is an excellent choice for deploying StrideOptics because it handles large dependencies (OpenCV, MediaPipe) and long-running processes well.

## Quick Deploy Steps

### 1. Install Railway CLI
```bash
npm i -g @railway/cli
```

### 2. Login to Railway
```bash
railway login
```

### 3. Initialize Railway Project
From your project directory:
```bash
railway init
```

### 4. Deploy
```bash
railway up
```

Railway will automatically:
- Detect Python
- Install dependencies from `requirements.txt`
- Run the app using the `Procfile`

## Environment Variables

Set these in Railway Dashboard → Variables:

1. **SECRET_KEY**: Generate a strong random string
   ```bash
   python -c "import secrets; print(secrets.token_hex(32))"
   ```

2. **FLASK_DEBUG**: Set to `False` for production

3. **PORT**: Railway automatically sets this (don't override)

## Database Setup

### Option 1: Use Railway PostgreSQL (Recommended)
1. In Railway dashboard, click "New" → "Database" → "Add PostgreSQL"
2. Railway will provide a `DATABASE_URL` environment variable
3. Update `backend/config.py` to use PostgreSQL:
   ```python
   DATABASE_URL = os.environ.get('DATABASE_URL', f'sqlite:///{DATABASE_PATH}')
   ```

### Option 2: Keep SQLite (for testing)
- SQLite will work but data won't persist between deployments
- Files are stored in Railway's filesystem

## File Storage

For production, consider using:
- **Railway Volume**: Persistent storage for uploads
- **Cloud Storage**: AWS S3, Cloudinary, or similar

## Monitoring

- Check logs in Railway dashboard
- Monitor resource usage (CPU, Memory)
- Set up alerts if needed

## Custom Domain

1. In Railway dashboard → Settings → Domains
2. Add your custom domain
3. Railway will provide DNS instructions

## Pricing

Railway offers:
- **Free tier**: $5 credit/month (good for testing)
- **Pay-as-you-go**: After free credit is used
- **Pro plan**: For production workloads

## Troubleshooting

### Build fails
- Check that all dependencies are in `requirements.txt`
- Ensure Python 3.9 is specified (Railway auto-detects)

### App crashes
- Check logs in Railway dashboard
- Verify environment variables are set
- Ensure PORT is not manually set (Railway handles this)

### Large deployment
- Railway handles large packages better than Vercel
- OpenCV and MediaPipe should work fine

## Local Testing with Railway

Test Railway setup locally:
```bash
railway run python backend/app.py
```

## Next Steps

1. Deploy to Railway
2. Set environment variables
3. Configure database (PostgreSQL recommended)
4. Test the deployment
5. Share the Railway URL with colleagues!

Your app will be available at: `https://your-project-name.up.railway.app`

