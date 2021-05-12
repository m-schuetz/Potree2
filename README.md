# About

* Playing around with WebGPU and learning how to rewrite [Potree](https://github.com/potree/potree/) with it.
* Tested with Chrome Version 92.
* Online demo: https://potree.org/temporary/potree2_ca13/
	<!-- * Achieves 144fps rendering 15.6M points in 3'106 nodes on an RTX 2080 Ti. -->
<!-- * [Experimental LAS viewer](http://potree.org/temporary/lasviewer_2021.01.14/lasviewer.html) -->

# References

## own

* [Rendering point clouds with compute shaders](https://github.com/m-schuetz/compute_rasterizer) <br>
In some cases, compute shaders can render point clouds multiply times faster than native primitives like GL_POINTS or "point-list". Extensive evaluation subject to future work. 
* [Fast Out-of-Core Octree Generation for Massive Point Clouds](https://www.cg.tuwien.ac.at/research/publications/2020/SCHUETZ-2020-MPC/)<br>
Generating LOD structures can be slow and time consuming. Using hierarchical counting sort, we can efficiently organize the input into small batches that can be processed in parallel. 
* [Potree: Rendering Large Point Clouds in Web Browsers](https://www.cg.tuwien.ac.at/research/publications/2016/SCHUETZ-2016-POT/) <br>
Rendering billions of points in web browsers using an octree acceleration structure. 


## more

* [QSplat](http://graphics.stanford.edu/papers/qsplat/)<br>
First approach to render large point clouds (or meshes transformed to point clouds) in real-time using a bounding-sphere hierarchy.
* [Layered Point Clouds](http://publications.crs4.it/pubdocs/2004/GM04c/spbg04-lpc.pdf)<br>
Basis of most commonly used point cloud acceleration structures, including Potree, Entwine, Arena4D, etc. The key feature is a GPU-friendly data structure that stores subsamples comprising thousands of points in each node. 
* [High-Quality Surface Splatting on Today’s GPUs](https://www.graphics.rwth-aachen.de/media/papers/splatting1.pdf) <br>
Anti-Aliasing for point clouds by blending overlapping fragments within a certain range together. Results are similar to mip mapping or anisotropic filtering for textured meshes, which don't work for point clouds because they are colored by vertex rather than texture. 
* ["A GPGPU-based Pipeline for Accelerated Rendering of Point Clouds."](https://core.ac.uk/download/pdf/295554194.pdf)<br>
Compute shader rendering using OpenCL.
* ["View-warped Multi-view Soft Shadowing for Local Area Lights"](http://jcgt.org/published/0007/03/01/)<br>
Uses compute shaders for point-based depth map rendering.

# Further Credits and Resources

* [WebGPU Samples](http://austin-eng.com/webgpu-samples/?wgsl=1#animometer). Github: [austinEng/webgpu-samples](https://github.com/austinEng/webgpu-samples)
* [WebGPU x-mas card](http://trierlab.com/VClab/webtek/xmas/)

# LICENSE

This prototype of Potree 2 uses the AGPL license: https://www.gnu.org/licenses/agpl-3.0.en.html. 