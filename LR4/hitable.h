#ifndef HITABLE_H
#define HITABLE_H

#include "vec3.h"
#include "ray.h"

// Заключает в себе функцию hit, которая принимает луч и засчитывается только в случае попадания в определнный интервал от tmin до tmax

struct hit_record
{
    float t;
    vec3 p;
    vec3 normal;
};

class hitable  {
    public:
        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
};

#endif