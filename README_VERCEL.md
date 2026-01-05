# Deploying StrideOptics to Vercel

This guide will help you deploy your StrideOptics application to Vercel.

## Prerequisites

1. A Vercel account (sign up at [vercel.com](https://vercel.com))
2. Vercel CLI installed: `npm i -g vercel`
3. Git repository (recommended)

## Important Considerations

⚠️ **Note**: Vercel serverless functions have limitations:
- **Execution Time**: Maximum 300 seconds (5 minutes) for Pro plan, 10 seconds for Hobby
- **Memory**: Limited to 3GB
- **File Storage**: `/tmp` directory only (ephemeral, cleared between invocations)
- **Database**: SQLite in `/tmp` will be lost between deployments. Consider using:
  - Vercel Postgres (recommended)
  - Supabase
  - PlanetScale
  - Other cloud databases

## Deployment Steps

### 1. Install Vercel CLI (if not already installed)
```bash
npm i -g vercel
```

### 2. Login to Vercel
```bash
vercel login
```

### 3. Deploy to Vercel
From the project root directory:
```bash
vercel
```

Follow the prompts:
- Set up and deploy? **Yes**
- Which scope? (Select your account/team)
- Link to existing project? **No** (for first deployment)
- What's your project's name? **strideoptics** (or your preferred name)
- In which directory is your code located? **./** (current directory)

### 4. Set Environment Variables

After deployment, set environment variables in Vercel dashboard:

1. Go to your project on [vercel.com](https://vercel.com)
2. Navigate to **Settings** → **Environment Variables**
3. Add the following:

```
SECRET_KEY=your-secret-key-here (generate a strong random string)
FLASK_DEBUG=False
```

### 5. Production Deployment

For production:
```bash
vercel --prod
```

## Configuration Files

### `vercel.json`
- Configures Python 3.9 runtime
- Sets max function duration to 300 seconds
- Routes static files to `/backend/static/`
- Routes all other requests to the Flask app

### `api/index.py`
- Serverless function handler
- Wraps Flask app for Vercel compatibility
- Sets Vercel environment flag

### `.vercelignore`
- Excludes unnecessary files from deployment
- Reduces deployment size

## Limitations & Workarounds

### 1. Video Processing Time
**Problem**: Video analysis may take longer than function timeout.

**Solutions**:
- Use Vercel Pro plan (300s timeout)
- Implement background job processing with:
  - Vercel Cron Jobs
  - External queue service (Redis, RabbitMQ)
  - Separate processing service

### 2. File Storage
**Problem**: `/tmp` is ephemeral and cleared between invocations.

**Solutions**:
- Use cloud storage:
  - Vercel Blob Storage
  - AWS S3
  - Cloudinary
  - Upload directly to storage, process from there

### 3. Database Persistence
**Problem**: SQLite in `/tmp` is not persistent.

**Solutions**:
- **Recommended**: Use Vercel Postgres
  ```bash
  vercel postgres create
  ```
- Or use other cloud databases (Supabase, PlanetScale, etc.)
- Update `config.py` to use PostgreSQL connection string

### 4. Large Dependencies
**Problem**: OpenCV and MediaPipe are large packages.

**Solutions**:
- Vercel will handle this, but deployment may take longer
- Consider using Docker for very large dependencies (Vercel supports Docker)

## Updating Database for Production

If using PostgreSQL (recommended):

1. Create Vercel Postgres database:
   ```bash
   vercel postgres create strideoptics-db
   ```

2. Update `backend/config.py`:
   ```python
   DATABASE_URL = os.environ.get('POSTGRES_URL', 'sqlite:///...')
   ```

3. Update `backend/models.py` to ensure PostgreSQL compatibility

## Monitoring & Debugging

- Check function logs in Vercel dashboard
- Use Vercel's function logs for debugging
- Monitor function execution time and memory usage

## Alternative Deployment Options

If Vercel limitations are too restrictive, consider:

1. **Railway** - Better for long-running processes
2. **Render** - Good Flask support, persistent storage
3. **Fly.io** - Docker-based, flexible
4. **Heroku** - Traditional PaaS (paid plans)
5. **AWS/GCP/Azure** - Full control, more setup required

## Local Development

To test Vercel setup locally:
```bash
vercel dev
```

This runs a local server that mimics Vercel's environment.

## Troubleshooting

### Import Errors
- Ensure all dependencies are in `requirements.txt`
- Check Python version matches (3.9)

### Function Timeout
- Reduce video processing time
- Process videos in chunks
- Use background jobs

### Memory Issues
- Optimize video processing
- Reduce batch sizes
- Consider upgrading Vercel plan

## Next Steps

1. Deploy to Vercel
2. Set up environment variables
3. Configure database (PostgreSQL recommended)
4. Set up cloud storage for videos
5. Test deployment thoroughly
6. Monitor performance and adjust as needed

