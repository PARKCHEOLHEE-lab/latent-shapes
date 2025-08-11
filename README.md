# latent-shapes


This project is a naive implementation of the paper [Free-form Floor Plan Design using Differentiable Voronoi Diagram](https://www.dropbox.com/scl/fi/culi7j1v14r9ax98rfmd6/2024_pg24_floorplan.pdf?rlkey=s5xwncuybrtsj5vyphhn61u0h&e=3&dl=0). The paper is based on the <b>differentiable Voronoi diagram</b>, but this repository uses `Shapely` and `Pytorch`. Specifically, PyTorch's autograd functionality for <b>numerical differentiation</b> is combined with Shapely's geometric operations to compute gradients. Also, the initialization method to assign room cells is different. I used the KMeans to converge the result faster than random initialization.
<mark>The detailed process for this project is archived [__here__](https://parkcheolhee-lab.github.io/floor-plan-generation-with-voronoi-diagram/).</mark>


<br>

<div align="center">
    <img src="latent_shapes/assets/latent-shapes-demo.gif" width="70%">　　
</div>
<p align="center" color="gray">
  <i>
  Design Exploration with Interpolator
  </i>
</p>
