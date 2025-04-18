# ZenStatement
This repository contains the code for Zenstatement assignment solved by Anirudh Punetha

## Setup
The requirements.txt contains the libraries I have used in my personal system, which is windows. I have tried to optimize it for all OS.
Use 'pip install -r requirements.txt' to install all the libraries required to run this code.
The api.py is the backend file and can be run using a simple 'python api.py'. You can also use the uvicorn command given in the dockerfile if you need multiple workers.
The gradio.py is the UI which I created for a demo video and can be run using 'python gradio_ui.py'

## Credentials
Please note that I have removed the AWS KEYS and OPENAI Key to prevent misuse of my personal account. If you are an evaluator please contact me and  I will share my keys with you over a private channel. You are also free to use your own keys.

## Dockerfile
I have tested the Dockerfile in a linux system and it works fine. You can use it to host your backend api.
For building the image use 'docker image build -t zenstatement:v1 .'
For running the container use 'docker container run -d -p 8000:8000 zenstatement:v1'
The container will only host the backend, the UI should still be run using the python command in setup.

## Approach
I have created 5 endpoints in my FastAPI file, which is api.py
1. /api/v1/zen/health is the endpoint for healthcheck.
2. /api/v1/zen/upload is used to upload files to local dir.
3. /api/v1/zen/preprocess is used to fill nan values and only keep the “Not Found Sys B” category. It will also upload the csv in the end to S3 bucket.
4. /api/v1/zen/resolve is used to resolve queries when you have uploaded the resolution file again from step1. It will create an openAI agent and classify the comment for each row as resolved or unresolved. If it is unresolved it will give the next steps as well. Each row is uploaded to S3 bucket based on resolution status. You can see the definition of agent and the prompt in api.py from line 60-80.
5. /api/v1/zen/cluster is used to gather all the resolved comments and cluster it using DBSCAN. First all comments are converted to embeddings using sentence transformers and then fit using a DBSCAN model. The noise is removed and clustered records are uploaded to S3 bucket.

## Improvements
Currently each row is processed sequentially. We can use a threadpool to process multiple comments at once.
Code needs to be split into smaller modules which can make it more readable.
I had to save some dataframes locally to display on the UI. If you use it via postman it is more efficient as no extra memory is used to store those temp dataframes for display.

