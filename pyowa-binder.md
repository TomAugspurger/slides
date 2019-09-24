---
title: Introduction to Binder
author: Tom Augspurger
affiliation: Anaconda
date: 2019-05-19
---

### Binder

* Share custom interactive computing environments
* Documentation
* Scientific publishing
* Teaching

### Example: Black holes!

- <https://github.com/minrk/ligo-binder>

### Components

* BinderHub = repo2docker + jupyterhub
* `repo2docker`: Convert a repository to a docker image
* jupyterhub: Multi-user server for Jupyter notebooks (and other things)
* <https://mybinder.org>: A public BinderHub deployment
 
### repo2docker

* Convert a repository to a Docker container
* Specify the environment
  * `environment.yml` / `requirements.txt`
  * `Project.toml` (Julia)
  * `install.R` (R)
  * ... or a Dockerfile
* Choose a UI
  * JupyterLab
  * nteract
  * [RStudio](http://mybinder.org/v2/gh/binder-examples/r/master?urlpath=rstudio)

### JupyterHub

![](https://jupyterhub.readthedocs.io/en/stable/_images/jhub-fluxogram.jpeg)

### JupyterHub

<https://andersonbanihirwe.dev/talks/dask-jupyter-scipy-2019.html>

### BinderHub

* Kubernetes application for creating and serving environments
* Builds repo images
* Serves users

### mybinder: A public BinderHub deployment

* <https://mybinder.org>

### More examples: documentation

* <https://examples.dask.org>
* <https://binder.pangeo.org>
