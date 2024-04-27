pip install thirdparty/gaussian_splatting/submodules/gaussian_rasterization_ch9

# Install for Gaussian Rasterization (Ch3) - Ours-Lite
pip install thirdparty/gaussian_splatting/submodules/gaussian_rasterization_ch3

# Grey image output
pip install thirdparty/gaussian_splatting/submodules/gaussian_rasterization_ch1

# Install for Forward Full - Ours-Full (speed up testing, mlp fused, no sigmoid)
pip install thirdparty/gaussian_splatting/submodules/forward_full

# Install for Forward Lite - Ours-Lite (speed up testing)
pip install thirdparty/gaussian_splatting/submodules/forward_lite


# Install for Forward Lite - Lite Single Channel output (speed up testing)
pip install thirdparty/gaussian_splatting/submodules/forward_lite_single

# Install simpleknn
pip install thirdparty/gaussian_splatting/submodules/simple-knn

# Install MMCV
cd thirdparty/mmcv && pip install -e . && cd ../../
