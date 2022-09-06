# COLLAGE MAKER

Create a photo mosaic from a reference photo using other pictures.

![Collage maker](gifs/collage_maker.gif)

You can try the app in the following url: https://collage-creator.herokuapp.com/

`Note: Heroku is a free host with CPU and request time limitations, i'm sorry for any inconvenience this may cause`

## HOW IT WORKS

The app is made using the python library Dash, which allows creating a front-end from a python back-end.

When the pictures are uploaded the predominant color of each of them is computed by applying a K-means algorithm to each.

Once the predominant color of each photo is calculated, another K-means computes N clusters in the RGB space that collect the pictures with similar colors, in order to generate a palette of N colors.

Finally, the distance in RGB space of each pixel of the reference image to each cluster centroid is computed and a photo of the closest cluster is replaced by the corresponding pixel. 

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