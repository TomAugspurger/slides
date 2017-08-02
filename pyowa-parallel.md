---
title: Parallel Programming in Python
author: Tom Augspurger
affiliation: Continuum Analytics
date: 2017-07-27
---

### Hi

I'm Tom, I work for Continuum Analytics

---

You may know us from projects like

<div class="left">
- Anaconda
- Bokeh
- Numba
- Dask
</div>

### Topics

- Super high-level overview of Python's parallelism story
- Specifically processing data in parallel
- In depth on Dask and distributed

### Some Warnings

- Parallel computing is difficult, and you may not need it
- Distributed computing is even more difficult, you probably don't need it

### Threads vs. Processes

First choice to make when deciding *how* to parallelize

### The Global Interpreter Lock

    /* file: ceval.c */
    /* This is the GIL */
    static PyThread_type_lock interpreter_lock = 0;

---

- Only one thread in your python process can run *Python* at once
- See [http://www.dabeaz.com/GIL/](http://www.dabeaz.com/GIL/) for more

### Threads

- Context switching between two CPU bound threads *worsen* performance
- All blocking IO functions in the standard library release the GIL
- Shared memory, so two threads can access the same data
 
### MultiProcessing

- Your python process is forked, making an independent copy
- Any communication must be *serialized*
- Sidesteps the GIL

### Threads vs. Processes

- Use `threading` for IO bound tasks
- Use `multiprocessing` for CPU bound tasks...

<div class="fragment">
---

..unless you're using the scientific Python stack

</div>

### Parallelism from the stdlib

Demo of [`concurrent.futures`](https://docs.python.org/3/library/concurrent.futures.html)


### Parallelism for PyData

- Libraries like NumPy, scikit-learn, and pandas contain a lot of C code
- They release the GIL when not nescessary

---

    # file: pandas/_libs/hashing.pyx
    result = np.empty(n, dtype=np.uint64)
    with nogil:  #  Isn't Cython neat?
        for i in range(n):
            result[i] = low_level_siphash(<uint8_t *>vecs[i],
                                          lens[i], kb)

---

### Parallelism for PyData

<div class="fragment">
  NumPy, pandas, scikit-learn don't provide parallelism natively
</div>

<div class="fragment">
But they allow for parallelism by releasing the GIL
</div>

### 

<img src="figures/dask_horizontal_white.svg"/>

Enable parallel and distributed computing for the PyData stack

### Dask's Values

- Flexible execution (thread, processes, distributed)
- Familiar API
- Low overhead

### Dask Demo

### Dask's Values

- Flexible execution --- Schedulers execute a DAG (task graph)
- Familiar API --- collections to build a DAG
- Low overhead --- pure python

---

<img src="figures/collections-schedulers.png">

### Distributed Scheduler

- Scale out to a cluster of computers (1,000s of nodes)
- Or run locally on your laptop

### Distributed Demo

### Summary

- `concurrent.futures` for simple data parallelism
- `dask` and `distributed` for more complex, dynamic workflows

### Thanks

- Slides: http://tomaugspurger.github.io/
- Twitter: @TomAugspurger
- Questions?
