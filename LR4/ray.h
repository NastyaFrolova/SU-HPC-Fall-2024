#ifndef RAY_H
#define RAY_H

#include "vec3.h"

// Класс луча, как функция p(t) = A + t*B, которая дает трехмерное положение. A - это начало луча, а B - направление луча. t - параметр для перемещения по линии
class ray
{
    public:
        __device__ ray() {}
        __device__ ray(const vec3& a, const vec3& b) { A = a; B = b; }
        __device__ vec3 origin() const       { return A; }
        __device__ vec3 direction() const    { return B; }
        __device__ vec3 point_at_parameter(float t) const { return A + t*B; }

        vec3 A;
        vec3 B;
};

#endif