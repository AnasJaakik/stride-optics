# Quick Railway Deployment Guide

## Step 1: Install Railway CLI
```bash
npm install -g @railway/cli
```

## Step 2: Login
```bash
railway login
```
This will open your browser to authenticate.

## Step 3: Initialize Project
```bash
railway init
```
- Choose "Create a new project"
- Name it: `stride-optics` (or your choice)

## Step 4: Deploy
```bash
railway up
```

Railway will:
- Detect Python automatically
- Install all dependencies from `requirements.txt`
- Start your app using the `Procfile`

## Step 5: Set Environment Variables

In Railway Dashboard → Variables, add:

```
SECRET_KEY=your-secret-key-here
FLASK_DEBUG=False
```

Generate a secret key:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

## Step 6: Get Your URL

After deployment, Railway will provide a URL like:
`https://stride-optics-production.up.railway.app`

You can also get it with:
```bash
railway domain
```

## Optional: Add PostgreSQL Database

1. In Railway dashboard, click "New" → "Database" → "Add PostgreSQL"
2. Railway automatically sets `DATABASE_URL` environment variable
3. Your app will automatically use PostgreSQL instead of SQLite

## Share with Colleagues

Just share the Railway URL! No need for them to install anything.

## Pricing

- **Free**: $5 credit/month (good for demos and testing)
- **Pay-as-you-go**: After free credit
- Perfect for showing to colleagues without cost concerns

## Troubleshooting

- Check logs: `railway logs`
- View in dashboard: Railway Dashboard → Deployments → Logs
- Restart service: Railway Dashboard → Settings → Restart

For detailed info, see `README_RAILWAY.md`

