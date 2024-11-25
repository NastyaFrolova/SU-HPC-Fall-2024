#ifndef CAMERAOLD_H
#define CAMERAOLD_H

#include "vec3.h"
#include "ray.h"

class camera {
    public:
        __device__ camera() {
            lower_left_corner = vec3(-2.0, -1.0, -0.3);
            horizontal = vec3(5, 0.0, 0.0);
            vertical = vec3(0.0, 2.0, 0.0);
            origin = vec3(0.0, 1.0, 3.0);
        }
        __device__ ray get_ray(float u, float v) { return ray(origin, lower_left_corner + u*horizontal + v*vertical - origin); }

        vec3 origin;
        vec3 lower_left_corner;
        vec3 horizontal;
        vec3 vertical;
};


#endif