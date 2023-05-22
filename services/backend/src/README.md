# Phase AI App: Back-end Service

All back-end services, components, and resources for the Phase AI app can be found under `/services/backend/src/`

## Required Libs

The main back-end services for the Phase AI app run on [FastAPI](https://fastapi.tiangolo.com/lo/), with [Uvicorn](https://www.uvicorn.org/) as the underlying web server.

First install them via

```bash
$ pip3 install uvicorn
$ pip3 install fastapi
```

## Starting the Back-end Service

You can then start the back-end service.

```bash
$ cd /services/backend/src
$ uvicorn main:app --reload --port 5000
```

Once the server is up, navigate to

* `http://127.0.0.1:5000`
* `http://127.0.0.1:5000/ping`

## API Docs

Swagger API documentation for the various endpoints on the Phase AI app back-end service can be found at `http://127.0.0.1:5000/docs`.