# COLLAGE MAKER

## FOR DEVELOPERS

### Start the App Locally
Install the python packages specified in the `requirements.txt` by running the following command:

```
pip install -r requirements.txt
```

Then, run the following command to execute the app:

```
gunicorn -b 0.0.0.0:8080 app.app:server --reload --timeout 120
```

The `--reload` tag at the end for live reloading on changes.

Navigate to: `http://localhost:8080/`

### Dockerize the app
Run:

```
docker build -t collage-maker .
docker run -p 8080:80 collage-maker
```