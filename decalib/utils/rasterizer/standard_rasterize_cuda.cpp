#include <torch/torch.h>
#include <vector>
#include <iostream>

std::vector<at::Tensor> forward_rasterize_cuda(
        at::Tensor face_vertices,
        at::Tensor depth_buffer,
        at::Tensor triangle_buffer,
        at::Tensor baryw_buffer,
        int h,
        int w);

std::vector<at::Tensor> standard_rasterize(
        at::Tensor face_vertices,
        at::Tensor depth_buffer,
        at::Tensor triangle_buffer,
        at::Tensor baryw_buffer,
        int height, int width
        ) {
    return forward_rasterize_cuda(face_vertices, depth_buffer, triangle_buffer, baryw_buffer, height, width);
}

std::vector<at::Tensor> forward_rasterize_colors_cuda(
        at::Tensor face_vertices,
        at::Tensor face_colors,
        at::Tensor depth_buffer,
        at::Tensor triangle_buffer,
        at::Tensor images,
        int h,
        int w);

std::vector<at::Tensor> standard_rasterize_colors(
        at::Tensor face_vertices,
        at::Tensor face_colors,
        at::Tensor depth_buffer,
        at::Tensor triangle_buffer,
        at::Tensor images,
        int height, int width
        ) {
    return forward_rasterize_colors_cuda(face_vertices, face_colors, depth_buffer, triangle_buffer, images, height, width);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("standard_rasterize", &standard_rasterize, "RASTERIZE (CUDA)");
    m.def("standard_rasterize_colors", &standard_rasterize_colors, "RASTERIZE COLORS (CUDA)");
}

// TODO: backward