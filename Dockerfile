# Pull clean runtime base image
FROM python:3.10-slim

# Set internal processing directory
WORKDIR /app

# Isolate dependency caching to prevent unneeded image bloating
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bring everything else over into container file space
COPY . .

# Expose port mapping configuration for Render deployment pipelines [cite: 58]
EXPOSE 5000

# Fire production-grade WSGI worker stack instead of internal development debugger
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "app:app"]