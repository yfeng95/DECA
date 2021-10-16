// Ref: https://github.com/daniilidis-group/neural_renderer/blob/master/neural_renderer/cuda/rasterize_cuda_kernel.cu
// https://github.com/YadiraF/face3d/blob/master/face3d/mesh/cython/mesh_core.cpp

#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace{
__device__ __forceinline__ float atomicMin(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
            __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}
__device__ __forceinline__ double atomicMin(double* address, double val)
{
    unsigned long long int* address_as_i = (unsigned long long int*) address;
    unsigned long long int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
            __double_as_longlong(fminf(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}

template <typename scalar_t>
__device__ __forceinline__ bool check_face_frontside(const scalar_t *face) {
    return (face[7] - face[1]) * (face[3] - face[0]) < (face[4] - face[1]) * (face[6] - face[0]);
}


template <typename scalar_t> struct point
{
    public:
    scalar_t x;
    scalar_t y;

    __host__ __device__ scalar_t dot(point<scalar_t> p)
    {
        return this->x * p.x + this->y * p.y;
    };

    __host__ __device__  point<scalar_t> operator-(point<scalar_t>& p)
    {
        point<scalar_t> np;
        np.x = this->x - p.x;
        np.y = this->y - p.y;
        return np;
    };

    __host__ __device__  point<scalar_t> operator+(point<scalar_t>& p)
    {
        point<scalar_t> np;
        np.x = this->x + p.x;
        np.y = this->y + p.y;
        return np;
    };

    __host__ __device__  point<scalar_t> operator*(scalar_t s)
    {
        point<scalar_t> np;
        np.x = s * this->x;
        np.y = s * this->y;
        return np;
    };
};          

template <typename scalar_t>
__device__ __forceinline__ bool check_pixel_inside(const scalar_t *w) {
    return w[0] <= 1 && w[0] >= 0 && w[1] <= 1 && w[1] >= 0 && w[2] <= 1 && w[2] >= 0;
}

template <typename scalar_t>
__device__ __forceinline__ void barycentric_weight(scalar_t *w,  point<scalar_t> p, point<scalar_t> p0,  point<scalar_t> p1,  point<scalar_t> p2) {
    
    // vectors
    point<scalar_t> v0, v1, v2;
    scalar_t s = p.dot(p);
    v0 = p2 - p0; 
    v1 = p1 - p0; 
    v2 = p - p0; 

    // dot products
    scalar_t dot00 = v0.dot(v0); //v0.x * v0.x + v0.y * v0.y //np.dot(v0.T, v0)
    scalar_t dot01 = v0.dot(v1); //v0.x * v1.x + v0.y * v1.y //np.dot(v0.T, v1)
    scalar_t dot02 = v0.dot(v2); //v0.x * v2.x + v0.y * v2.y //np.dot(v0.T, v2)
    scalar_t dot11 = v1.dot(v1); //v1.x * v1.x + v1.y * v1.y //np.dot(v1.T, v1)
    scalar_t dot12 = v1.dot(v2); //v1.x * v2.x + v1.y * v2.y//np.dot(v1.T, v2)

    // barycentric coordinates
    scalar_t inverDeno;
    if(dot00*dot11 - dot01*dot01 == 0)
        inverDeno = 0;
    else
        inverDeno = 1/(dot00*dot11 - dot01*dot01);

    scalar_t u = (dot11*dot02 - dot01*dot12)*inverDeno;
    scalar_t v = (dot00*dot12 - dot01*dot02)*inverDeno;

    // weight
    w[0] = 1 - u - v;
    w[1] = v;
    w[2] = u;
}

// Ref: https://www.scratchapixel.com/lessons/3d-basic-rendering/rasterization-practical-implementation/overview-rasterization-algorithm
template <typename scalar_t>
__global__ void forward_rasterize_cuda_kernel(
        const scalar_t* __restrict__ face_vertices, //[bz, nf, 3, 3]
        scalar_t*  depth_buffer,
        int*  triangle_buffer,
        scalar_t*  baryw_buffer,        
        int batch_size, int h, int w, 
        int ntri) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * ntri) {
        return;
    }
    int bn = i/ntri;
    const scalar_t* face = &face_vertices[i * 9];
    scalar_t bw[3];
    point<scalar_t> p0, p1, p2, p;

    p0.x = face[0]; p0.y=face[1];
    p1.x = face[3]; p1.y=face[4];
    p2.x = face[6]; p2.y=face[7];
    
    int x_min = max((int)ceil(min(p0.x, min(p1.x, p2.x))), 0);
    int x_max = min((int)floor(max(p0.x, max(p1.x, p2.x))), w - 1);
    int y_min = max((int)ceil(min(p0.y, min(p1.y, p2.y))), 0);
    int y_max = min((int)floor(max(p0.y, max(p1.y, p2.y))), h - 1);

    for(int y = y_min; y <= y_max; y++) //h
    {
        for(int x = x_min; x <= x_max; x++) //w
        {
            p.x = x; p.y = y;
            barycentric_weight(bw, p, p0, p1, p2);
            // if(((bw[2] >= 0) && (bw[1] >= 0) && (bw[0]>0)) && check_face_frontside(face))
            if((bw[2] >= 0) && (bw[1] >= 0) && (bw[0]>0))
            {
                // perspective correct: https://www.scratchapixel.com/lessons/3d-basic-rendering/rasterization-practical-implementation/perspective-correct-interpolation-vertex-attributes
                scalar_t zp = 1. / (bw[0] / face[2] + bw[1] / face[5] + bw[2] / face[8]);
                // printf("%f %f %f \n", (float)zp, (float)face[2], (float)bw[2]);
                atomicMin(&depth_buffer[bn*h*w + y*w + x],  zp);
                if(depth_buffer[bn*h*w + y*w + x] == zp)
                {
                    triangle_buffer[bn*h*w + y*w + x] = (int)(i%ntri);
                    for(int k=0; k<3; k++){
                        baryw_buffer[bn*h*w*3 + y*w*3 + x*3 + k] = bw[k];
                    }
                }
            }
        }
    }

}

template <typename scalar_t>
__global__ void forward_rasterize_colors_cuda_kernel(
        const scalar_t* __restrict__ face_vertices, //[bz, nf, 3, 3]
        const scalar_t* __restrict__ face_colors, //[bz, nf, 3, 3]
        scalar_t*  depth_buffer,
        int*  triangle_buffer,
        scalar_t*  images,        
        int batch_size, int h, int w, 
        int ntri) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * ntri) {
        return;
    }
    int bn = i/ntri;
    const scalar_t* face = &face_vertices[i * 9];
    const scalar_t* color = &face_colors[i * 9];
    scalar_t bw[3];
    point<scalar_t> p0, p1, p2, p;

    p0.x = face[0]; p0.y=face[1];
    p1.x = face[3]; p1.y=face[4];
    p2.x = face[6]; p2.y=face[7];
    scalar_t cl[3][3];
    for (int num = 0; num < 3; num++) {
        for (int dim = 0; dim < 3; dim++) {
            cl[num][dim] = color[3 * num + dim]; //[3p,3rgb]
        }
    }
    int x_min = max((int)ceil(min(p0.x, min(p1.x, p2.x))), 0);
    int x_max = min((int)floor(max(p0.x, max(p1.x, p2.x))), w - 1);
    int y_min = max((int)ceil(min(p0.y, min(p1.y, p2.y))), 0);
    int y_max = min((int)floor(max(p0.y, max(p1.y, p2.y))), h - 1);

    for(int y = y_min; y <= y_max; y++) //h
    {
        for(int x = x_min; x <= x_max; x++) //w
        {
            p.x = x; p.y = y;
            barycentric_weight(bw, p, p0, p1, p2);
            if(((bw[2] >= 0) && (bw[1] >= 0) && (bw[0]>0)) && check_face_frontside(face))
            // if((bw[2] >= 0) && (bw[1] >= 0) && (bw[0]>0))
            {
                scalar_t zp = 1. / (bw[0] / face[2] + bw[1] / face[5] + bw[2] / face[8]);

                atomicMin(&depth_buffer[bn*h*w + y*w + x],  zp);
                if(depth_buffer[bn*h*w + y*w + x] == zp)
                {
                    triangle_buffer[bn*h*w + y*w + x] = (int)(i%ntri);
                    for(int k=0; k<3; k++){
                        // baryw_buffer[bn*h*w*3 + y*w*3 + x*3 + k] = bw[k];
                        images[bn*h*w*3 + y*w*3 + x*3 + k] = bw[0]*cl[0][k] + bw[1]*cl[1][k] + bw[2]*cl[2][k];
                    }
                    // buffers[bn*h*w*2 + y*w*2 + x*2 + 1] = p_depth;
                }
            }
        }
    }

}
    
}

std::vector<at::Tensor> forward_rasterize_cuda(
    at::Tensor face_vertices,
    at::Tensor depth_buffer,
    at::Tensor triangle_buffer,
    at::Tensor baryw_buffer,
    int h,
    int w){

    const auto batch_size = face_vertices.size(0);
    const auto ntri = face_vertices.size(1);

    // print(channel_size)
    const int threads = 512;
    const dim3 blocks_1 ((batch_size * ntri - 1) / threads +1);

    AT_DISPATCH_FLOATING_TYPES(face_vertices.type(), "forward_rasterize_cuda1", ([&] {
      forward_rasterize_cuda_kernel<scalar_t><<<blocks_1, threads>>>(
        face_vertices.data<scalar_t>(),
        depth_buffer.data<scalar_t>(),
        triangle_buffer.data<int>(),
        baryw_buffer.data<scalar_t>(),
        batch_size, h, w,
        ntri);
      }));

    // better to do it twice  (or there will be balck spots in the rendering)
    AT_DISPATCH_FLOATING_TYPES(face_vertices.type(), "forward_rasterize_cuda2", ([&] {
        forward_rasterize_cuda_kernel<scalar_t><<<blocks_1, threads>>>(
        face_vertices.data<scalar_t>(),
        depth_buffer.data<scalar_t>(),
        triangle_buffer.data<int>(),
        baryw_buffer.data<scalar_t>(),
        batch_size, h, w,
        ntri);
        }));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in forward_rasterize_cuda_kernel: %s\n", cudaGetErrorString(err));

    return {depth_buffer, triangle_buffer, baryw_buffer};
}


std::vector<at::Tensor> forward_rasterize_colors_cuda(
    at::Tensor face_vertices,
    at::Tensor face_colors,
    at::Tensor depth_buffer,
    at::Tensor triangle_buffer,
    at::Tensor images,
    int h,
    int w){

    const auto batch_size = face_vertices.size(0);
    const auto ntri = face_vertices.size(1);

    // print(channel_size)
    const int threads = 512;
    const dim3 blocks_1 ((batch_size * ntri - 1) / threads +1);
    //initial 

    AT_DISPATCH_FLOATING_TYPES(face_vertices.type(), "forward_rasterize_colors_cuda", ([&] {
      forward_rasterize_colors_cuda_kernel<scalar_t><<<blocks_1, threads>>>(
        face_vertices.data<scalar_t>(),
        face_colors.data<scalar_t>(),
        depth_buffer.data<scalar_t>(),
        triangle_buffer.data<int>(),
        images.data<scalar_t>(),
        batch_size, h, w,
        ntri);
      }));
    // better to do it twice 
    // AT_DISPATCH_FLOATING_TYPES(face_vertices.type(), "forward_rasterize_colors_cuda", ([&] {
    //     forward_rasterize_colors_cuda_kernel<scalar_t><<<blocks_1, threads>>>(
    //       face_vertices.data<scalar_t>(),
    //       face_colors.data<scalar_t>(),
    //       depth_buffer.data<scalar_t>(),
    //       triangle_buffer.data<int>(),
    //       images.data<scalar_t>(),
    //       batch_size, h, w,
    //       ntri);
    //     }));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in forward_rasterize_cuda_kernel: %s\n", cudaGetErrorString(err));

    return {depth_buffer, triangle_buffer, images};
}




