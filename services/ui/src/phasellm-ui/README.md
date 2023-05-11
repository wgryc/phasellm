# Phase AI App: Front-end Service

All front-end services, components, and resources for the Phase AI app can be found under `/services/ui/src/phasellm-ui`

## Required Libs

The Phase AI app is developed using a variety of utilities which are run on [Node](https://nodejs.org/) and [npm](https://www.npmjs.com/):

On OS X, you can install both via [brew](https://formulae.brew.sh/formula/node) as follows:

```bash
$ brew install node
$ node -v
$ npm -v
```

### PrimeVue UI Components

The Phase AI app is built on [PrimeVue](https://primevue.org/) UI components which you can install as follows:

```bash
$ cd services/ui/src/phasellm-ui
$ npm install primevue
$ npm install -g @vue/cli
$ npm install vue-router
$ npm install axios
$ npm install -g serve
```

## Serving the Front-end

First, build (compile) the front-end:

```bash
$ npm run build
```

To view the production-ready (compiled, minified) front-end, run

```bash
$ serve -s dist
```

after which the front-end can be viewed at `http://localhost:3000`

## Local Dev with hot-reloading

```
$ npm run serve
```

The dev version of the front-end is then available at `http://localhost:8080`

## UI Project Setup
```
$ npm install
```

### Linting
```
$ npm run lint
```
