# Quick Deploy Guide for StrideOptics

## Deploy to Vercel

### Step 1: Install Vercel CLI
```bash
npm install -g vercel
```

### Step 2: Login
```bash
vercel login
```

### Step 3: Deploy
```bash
vercel
```

Follow the prompts. For first deployment:
- Set up and deploy? **Yes**
- Link to existing project? **No**
- Project name: **strideoptics** (or your choice)
- Directory: **./**

### Step 4: Set Environment Variables

In Vercel Dashboard → Settings → Environment Variables, add:

```
SECRET_KEY=your-strong-random-secret-key-here
FLASK_DEBUG=False
```

### Step 5: Deploy to Production
```bash
vercel --prod
```

## Important Notes

⚠️ **Vercel Limitations:**
- Function timeout: 300 seconds max (Pro plan)
- File storage: Only `/tmp` (ephemeral)
- Database: SQLite in `/tmp` is not persistent

**Recommended for Production:**
1. Use Vercel Postgres for database:
   ```bash
   vercel postgres create
   ```
2. Use cloud storage for videos (Vercel Blob, S3, etc.)
3. Consider background jobs for long video processing

## Local Testing

Test Vercel setup locally:
```bash
vercel dev
```

## Troubleshooting

- Check function logs in Vercel dashboard
- Ensure all dependencies are in `requirements.txt`
- Verify Python version is 3.9
- Check environment variables are set

For detailed information, see `README_VERCEL.md`

