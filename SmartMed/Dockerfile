FROM python:3.9

# set environment variables
ENV APP_HOME=/app
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# set work directory
WORKDIR $APP_HOME

# Install system dependencies needed for OpenCV (libGL.so.1)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0

# update pip, install dependencies
RUN pip install --upgrade pip 
COPY ./requirements.txt $APP_HOME
RUN pip install -r requirements.txt

# copy app folder
COPY . $APP_HOME

# run python commands to prepare the database and collect static files
RUN python manage.py makemigrations
RUN python manage.py migrate
RUN python manage.py collectstatic --noinput --clear

# Expose the port
EXPOSE 8000


