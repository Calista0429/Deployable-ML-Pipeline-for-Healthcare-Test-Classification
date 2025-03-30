FROM python:3.9

# Set working directory
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
# Copy all files
COPY ./app/ /code/app/

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Expose the FastAPI port
EXPOSE 8000

# Start the FastAPI app using uvicorn
CMD ["fastapi", "run", "app/main.py", "--port", "8000"]
