# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.9

# # Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# expose port
EXPOSE 5000

# Install production dependencies.
RUN python3 -m pip install --upgrade pip
# RUN pip3 install "uvicorn[standard]" gunicorn
# Install production dependencies.
RUN pip3 install -r requirements.txt

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# CMD exec gunicorn --bind :$PORT main:app --workers 1 --timeout 1200 --preload
CMD exec gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind :$PORT --timeout 1200 

# FROM python:3.9

# RUN pip install virtualenv
# ENV VIRTUAL_ENV=/venv
# RUN virtualenv venv -p python3
# ENV PATH="VIRTUAL_ENV/bin:$PATH"

# WORKDIR /app
# ADD . /app

# # install dependencies
# RUN pip install -r requirements.txt

# # expose port
# EXPOSE 5000

# # run application
# CMD ["python", "main.py"]
