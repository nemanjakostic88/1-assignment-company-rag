# 0. Important Note
My colleague Milos and I worked together on this task. This is just a dockerized version. Considering that my colleague Milos has already recorded a video, I will not make it and upload the same so as not to waste my time and yours.
# 1. Setup Environment
## Setup virtual environment
docker-compose build --no-cache
## Setup Python environment
Copy .env.sample to .env file in the root directory and set variables
# 2. Enter data
docker-compose run --rm app ingestion.py
# 3. Ask questions
docker-compose run --rm app generation.py
# 4. See Improvement from filtering
docker-compose run --rm app precision_delta.py


