# Latent Shapes [<img src="latent_shapes/assets/external-link.svg" width="25" height="25" alt="open the demo in new tab" />](https://parkcheolhee-lab.github.io/latent-shapes/)

<div align="justify">
  The Latent Shapes project explores two main conceptions:
  <strong>1) Latent vectors with geometric shape ─</strong> exploring what happens if we give the latent vectors a geometric structure, and how that might help us better understand or control the results;
  <strong>2) Interactive manipulation using the latent vector ─</strong> making it possible for people to easily change and play with the latent vectors, so they can see and understand the effects right away and create new shapes or designs interactively.
  <br><br>
  The detailed process of this project is archived <a href="https://parkcheolhee-lab.github.io/latent-shapes-experiment/">here</a>.
</div>

<br>

<div align="center" text-align="justify">
  <img src="latent_shapes/assets/latent-shapes-demo.gif" alt="Interactive latent-shape editing and reconstruction demo" width="100%">　　
</div>

<br><br>

# Demo

### In the browser

<div align="justify">
  The demo runs entirely in your browser — nothing to install and no server: <strong><a href="https://parkcheolhee-lab.github.io/latent-shapes/">try it here</a></strong>.
  The trained decoder is exported to ONNX and executed client-side by <code>onnxruntime-web</code>, on WebGPU when the browser supports it and on WASM otherwise.
  Reconstruction samples the SDF coarse-to-fine — an <a href="https://en.wikipedia.org/wiki/Octree">octree</a>-style refinement that subdivides only the cells near the surface — and sharpens the mesh progressively while the rest computes.
</div>

<br>

### Running locally

<div align="justify">
  The local demo can run on both CPU and GPU. However, CPU inference is significantly slower. For reference, using RTX 3060 Laptop GPU at 80 for resolution, mesh reconstruction took approximately 1500ms. To run the demo, execute <code>cd latent_shapes/demo && python app.py</code>.
  The environment setup is described in <a href="/.devcontainer/README.md">.devcontainer</a>.
</div>

<br><br>


# Limitations and Future Works

<div align="justify">
  The most critical limitation is that the latent shape vertices do not understand shapes at the semantic part level.
  Each of the 98 vertices is a purely geometric handle over a nearby region of space — no vertex corresponds to a part a person would name, such as the backrest, an armrest, or a leg.
  As a result, editing a specific part is indirect: you have to discover which vertices influence it by trial and error, and a single drag can also deform neighboring parts you did not intend to touch.
  <br><br>
  A promising future work is making the latent shape part-aware — for example, aligning vertices (or groups of them) with semantic part annotations such as <a href="https://partnet.cs.stanford.edu/">PartNet</a>, or learning the grouping directly, so that grabbing "the armrest" moves exactly the vertices that encode it.
  This would turn the interaction from space-level dragging into part-level editing.
</div>
