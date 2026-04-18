# Use the official Python 3.11 slim image as a base
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /workspace

# Install the openai library needed for vLLM communication
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir openai

# install sympy
RUN pip install --no-cache-dir sympy numpy

# Add any other dependencies required by the task execution environment here
# e.g., RUN pip install --no-cache-dir numpy pandas etc.

# Optional: You could copy the sandbox_eval_helper.py here instead of mounting,
# but mounting allows easier updates without rebuilding the image.

# Default command can be empty or set if needed, but we override it anyway.