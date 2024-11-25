#include <curand_kernel.h>
#include <cstdio>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include "EasyBMP.h"
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "camera.h"
#include "hitable.h"
#include "hitable_list.h"
#include <vector_types.h>
#include <vector_functions.h>
#include <cmath>
#include <stdio.h>
#include <chrono>

// Проверка на ошибки возникающие в CUDA
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
//Функция для вывода кода ошибки
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

#define RECURSION_DEPTH 5 // Максимальная глубина рекурсии - 5

// Используется метод отклонения
// Выбирается случайная точка в единичном кубе. Для генерации случайный чисел используется библиотека curand
// Используем отклонение и если точка находится за пределами сфер то пробуем снова
#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ vec3 random_in_unit_sphere(curandState *local_rand_state) {
    vec3 p;
    do {
        p = 2.0f*RANDVEC3 - vec3(1,1,1);
    } while (p.squared_length() >= 1.0f);
    return p;
}

// Функция определяющая цвет полученной при столкновении луча и объекта
// Луч направляется на пискели и определяется цвет, который виден в направлении этих лучей
__device__ vec3 color(const ray& r, hitable **world, curandState *local_rand_state) {
    ray cur_ray = r;
    float cur_attenuation = 1.0f;
    for (int i = 0; i < RECURSION_DEPTH; i++) {
        hit_record rec;
        // Некоторые отраженные лучи попадают на объект, от которого они отражаются, но не точно в 0 или в любую другую плавающую точку, которую дает нам пересечение сфер
        // необходимо игнорировать попадания, очень близкие к нулю
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) { // Таким образом, если True, то повторяется
            vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
            cur_attenuation *= 0.5f;
            cur_ray = ray(rec.p, target - rec.p);
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.8, 0.1, -1.); // Цвет сфер
            return cur_attenuation * c;
        }
    }
    return vec3(0.0,0.0,0.0); // Превышена глубина рекурсии
}

// Случайное заполнение пространства
// Используется для отделения времени случайной инициализации от времени, необходимого для рендеринга
__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    // Каждая нить получает одно и то же начальное число, разный порядковый номер, без смещения
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

// Рендеринг
__global__ void render(vec3* fb, int max_x, int max_y, int ns, camera** cam, hitable** world, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0, 0, 0);
    for (int s = 0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v);
        col += color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    // Для тени
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

// Создание мира
__global__ void create_world(hitable** d_list, hitable** d_world, camera** d_camera, int nx, int ny, int N_spheres) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Создание набора сфер
        d_list[0] = new sphere(vec3(0, 0, -1), 0.5);
        d_list[1] = new sphere(vec3(0,-100.5,-1), 100);
        d_list[2] = new sphere(vec3(1,0,-1), 0.5);
        d_list[3] = new sphere(vec3(-1,0,-0.5), 0.6);
        d_list[4] = new sphere(vec3(2.5, 0, 3.5), 0.7);
        d_list[5] = new sphere(vec3(-2.5, 0, 0), 0.5);
        d_list[6] = new sphere(vec3(1, -0.2, 2), 0.4);
        d_list[7] = new sphere(vec3(-1, 0, -4), 0.7);
        d_list[8] = new sphere(vec3(2, 0, -1), 0.5);
        d_list[9] = new sphere(vec3(3, 0, -3), 0.5);
        d_list[10] = new sphere(vec3(3, 0, 2), 0.6);
        // Сохраняем наши лучи
        *d_world = new hitable_list(d_list, N_spheres+1);
        // Настраиваем камеру, угол обзора и расположение
        *d_camera   = new camera(vec3(-5,6,1),
                                 vec3(0,0,-1),
                                 vec3(0,1,0),
                                 48.0,
                                 float(nx)/float(ny));
    }
}

// Удаление всех элементов
__global__ void free_world(hitable **d_list, hitable **d_world, camera **d_camera) {
    for (int i = 0; i < 11; i++) {
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

// В моем случае используются матовые объекты

int main() {
    int nx = 1920;
    int ny = 1080;
    int N_spheres = 10;
    int ns = 100;
    int tx = 8;
    int ty = 8;

    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx * ny; //Определяем размер изображения
    size_t fb_size = num_pixels * sizeof(vec3); // Определяем размер буфера

    // Выделяем буфер изображения, которое потом будет выгружено
    vec3* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

    // Генерируем случайное состояние
    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));

    // Создаем мир с определенным числом лучей и положением камеры
    hitable** d_list;
    checkCudaErrors(cudaMalloc((void**)&d_list, 11 * sizeof(hitable*)));
    hitable** d_world;
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hitable*)));
    camera** d_camera;
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));
    create_world << <1, 1 >> > (d_list, d_world, d_camera, nx, ny, N_spheres);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Создаем переменные для измерения времени, затраченного на рендерринг
    clock_t start, stop;
    start = clock();
    // Определяем число блоков и нитей
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    render_init << <blocks, threads >> > (nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render << <blocks, threads >> > (fb, nx, ny, ns, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";


    BMP image;
    image.SetSize(nx, ny);

    // Выгружаем изображение из буфера в формат, пригодный для BMP
    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j * nx + i;
            RGBApixel pixel;
            pixel.Red = int(255.99 * fb[pixel_index].r());
            pixel.Green = int(255.99 * fb[pixel_index].g());
            pixel.Blue = int(255.99 * fb[pixel_index].b());
            pixel.Alpha = 0;
            image.SetPixel(i, j, pixel);


        }
    }
    image.WriteToFile("out.bmp");

    // Очищаем мир и память
    checkCudaErrors(cudaDeviceSynchronize());
    free_world << <1, 1 >> > (d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(fb));

    cudaDeviceReset();
}
