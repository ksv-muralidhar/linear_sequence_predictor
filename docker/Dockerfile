FROM python:3.7-slim
WORKDIR /webapp
COPY . .
RUN rm -r ./docker
RUN pip install --no-cache-dir -r requirements.txt
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0", "--port=5002"]
EXPOSE 5002