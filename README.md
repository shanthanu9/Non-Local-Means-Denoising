# NLM: Non Local Means

## Repo Tour

`nlm.jl` contains the NLM implementation codes. I have used Julia language to implement
the algorithms.

`nlm.jl` is actually a Pluto notebook [similar to Jupyter notebook]. Refer
[https://github.com/fonsp/Pluto.jl](https://github.com/fonsp/Pluto.jl) for
instructions to open and run it in Pluto.

Also you will see several images in this repo. `nosiy_mosaic.png` has all noisy
images. `mosaic-5.png`, etc have the respective denoised images. `plot.png` has
the PSNR plots.

**NOTE**: To take advantage of multi-threaded code, launch julia in your terminal
with multiple threads.