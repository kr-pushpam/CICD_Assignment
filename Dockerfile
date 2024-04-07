	# Use an official Python runtime as a parent image
	FROM python:3.8
	
	# Set the working directory in the container
	WORKDIR /app
	
	# Copy the Python scripts and data directory into the container
	COPY train.py test.py /app/
	COPY data /app/data/
	
	# Install pandas and scikit-learn using pip
	RUN pip install --no-cache-dir pandas scikit-learn numpy
	
	# Run train.py to train the model and save it as model.pkl
	RUN python train.py
	
	# Command to run test.py when the container starts
CMD ["python", "test.py"]
