#include <curand_kernel.h>
#include <cstdio>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include "EasyBMP.h"
#include <vector_types.h>
#include <vector_functions.h>
#include <cmath>
#include <stdio.h>
#include <chrono>

#define RECURSION_DEPTH 5 // Максимальная глубина рекурсии - 5

struct vec3 {
    float x=0, y=0, z=0;
          float& operator[](const int i)       { return i==0 ? x : (1==i ? y : z); }
    const float& operator[](const int i) const { return i==0 ? x : (1==i ? y : z); }
    vec3  operator*(const float v) const { return {x*v, y*v, z*v};       }
    float operator*(const vec3& v) const { return x*v.x + y*v.y + z*v.z; }
    vec3  operator+(const vec3& v) const { return {x+v.x, y+v.y, z+v.z}; }
    vec3  operator-(const vec3& v) const { return {x-v.x, y-v.y, z-v.z}; }
    vec3  operator-()              const { return {-x, -y, -z};          }
    float norm() const { return std::sqrt(x*x+y*y+z*z); }
    vec3 normalized() const { return (*this)*(1.f/norm()); }
};

struct sphere
{
    vec3 center;
    float radius;
};

struct ray
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

vec3 reflect(const vec3 &A, const vec3 &B) {
    return A - B*2.f*(A*B);
}

/*__global__ void TraceRay(ray, int depth)
{
    if(depth > RECURSION_DEPTH)
    return 0 ;
    find closest ray object/intersection;
    if(intersection exists)
    {
        for each light source in the scene
        {
            if(light source is visible)
            {
            illumination += light contribution;
            }
        }
        if(surface is reflective)
        {
            illumination += TraceRay(reflected ray, depth+ 1 ) ;
        }
        if(surface is transparent)
        {
            illumination += TraceRay(refracted ray, depth+ 1 ) ;
        }
        return illumination modulated according to the surface properties;
    }
    else return EnvironmentMap(ray) ;
    }
for each pixel
{
    compute ray starting point and direction;
    illumination = TraceRay(ray, 0 ) ;
    pixel color = illumination tone mapped to displayable range;
}*/



int main() {
    int WIDTH = 1920;
    int HEIGHT = 1080;
    int N_spheres = 10;
    int ns = 100;
    int tx = 8;
    int ty = 8;

    BMP AnImage;
    AnImage.SetSize(WIDTH, HEIGHT) ;
    for (int i = 0 ; i < WIDTH; i++ )
        for (int j = 0 ; j < HEIGHT; j++ )
        {
            RGBApixel pixel;
            pixel . Red = pR[j * WIDTH + i ] ;
            pixel . Green = pG[j * WIDTH + i ] ;
            pixel . Blue = pB[j * WIDTH + i ] ;
            pixel . Alpha = 0 ;
            AnImage.SetPixel(i , j , pixel) ;
        }
    AnImage.WriteToFile("out.bmp") ;
}
