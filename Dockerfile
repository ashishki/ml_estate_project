# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies from requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# (Optional) Install additional packages for testing
RUN pip install pytest pytest-cov

# Expose port if your project will run a web service (optional)
EXPOSE 8080

# Set environment variable to avoid buffering issues
ENV PYTHONUNBUFFERED=1

# Run tests by default when the container starts
CMD ["pytest", "--maxfail=1", "--disable-warnings", "-v"]
