## Install
from standard_rasterize_cuda import standard_rasterize
        # from .rasterizer.standard_rasterize_cuda import standard_rasterize
        
in this folder, run
```python setup.py build_ext -i ```

then remember to set --rasterizer_type=standard when runing demos :)  

## Alg
https://www.scratchapixel.com/lessons/3d-basic-rendering/rasterization-practical-implementation

## Speed Comparison
runtime for raterization only  
In PIXIE, number of faces in SMPLX: 20908   

for image size = 1024  
pytorch3d: 0.031s  
standard: 0.01s  

for image size = 224  
pytorch3d: 0.0035s  
standard: 0.0014s  
  
why standard rasterizer is faster than pytorch3d?  
Ref: https://github.com/facebookresearch/pytorch3d/blob/master/pytorch3d/csrc/rasterize_meshes/rasterize_meshes.cu  
pytorch3d: for each pixel in image space (each pixel is parallel in cuda), loop through the faces, check if this pixel is in the projection bounding box of the face, then sorting faces according to z, record the face id of closest K faces.   
standard rasterization: for each face in mesh (each face is parallel in cuda), loop through pixels in the projection bounding box (normally a very samll number), compare z, record face id of that pixel   

