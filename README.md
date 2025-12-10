<p align="center">

  <h1 align="center">Intrinsic Image Fusion for Multi-View 3D Material Reconstruction</h1>
  <p align="center">
    <a href="https://peter-kocsis.github.io/">Peter Kocsis</a>
    ·
    <a href="https://lukashoel.github.io/">Lukas Höllein</a>
    ·
    <a href="https://niessnerlab.org/members/matthias_niessner/profile.html">Matthias Nießner</a>
  </p>
  <h2 align="center">ArXiv 2025</h2>
  <h3 align="center"><a href="TODO">Paper</a> | <a href="https://peter-kocsis.github.io/IntrinsicImageFusion/">Project Page</a> </h3>
  <div align="center"></div>
</p>

<p align="center">
  <a href="">
    <img src="./docs/static/teaser/teaser.jpg" alt="Logo" width="95%">
  </a>
</p>

<p align="center">
We introduce Intrinsic Image Fusion, a method that reconstructs high-quality physically based materials from multi-view images.
Material reconstruction is highly underconstrained and typically relies on analysis-by-synthesis, which requires expensive and noisy path tracing. 
To better constrain the optimization, we incorporate single-view priors into the reconstruction process. 
We leverage a diffusion-based material estimator that produces multiple, but often inconsistent, candidate decompositions per view.
To reduce the inconsistency, we fit an explicit low-dimensional parametric function to the predictions.
We then propose a robust optimization framework using soft per-view prediction selection together with confidence-based soft multi-view inlier set to fuse the most consistent predictions of the most confident views into a consistent parametric material space. 
Finally, we use inverse path tracing to optimize for the low-dimensional parameters. 
Our results outperform state-of-the-art methods in material disentanglement on both synthetic and real scenes, producing sharp and clean reconstructions suitable for high-quality relighting.
</p>
<br>
