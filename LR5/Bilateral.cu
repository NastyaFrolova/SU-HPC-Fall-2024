#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include "EasyBMP.h"
#include <stdio.h>
#include <omp.h>
#include <chrono>

void saveImage(float* image, int width, int height, bool method) {
	BMP Output;
	Output.SetSize(width, height);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			RGBApixel pixel;
			pixel.Red = image[i * width + j];
			pixel.Green = image[i * width + j];
			pixel.Blue = image[i * width + j];
            pixel.Alpha = 0 ;
			Output.SetPixel(j, i, pixel);
		}
	}
	if (method) {
		Output.WriteToFile("GPUout.bmp");
        std::cout << "The image from GPU has been saved." << std::endl;
    }
	else {
		Output.WriteToFile("CPUout.bmp");
        std::cout << "The image from CPU has been saved." << std::endl;
    }

}


// Функция для расчета амплитуды Гаусса на CPU
float g(int x, int y, int x0, int y0, float sigmad) {
    return exp(- (pow((x - x0),2) - pow((y - y0),2))/pow(sigmad,2));
}


// Функция для расчета новой интенсивности без нормировочных коэффициентов на CPU
float r(int a, int a0, float sigmar) {
    return exp(- pow((a - a0),2) /pow(sigmar,2));
}

//Функция для фильтра Гаусса на CPU
void Bilateral_CPU(float* input, float* output, int width, int height, float sigmad, float sigmar) {
    omp_set_num_threads(8); //Устанавливаем число потоков для параллельных вычислений
    #pragma omp parallel /*Уставновка структурного блока, вычисление которого происходит параллельно*/
        #pragma omp master //Выполнение только главным потоком
        {
            for (int  i = 0; i < height; i++) {
                for (int  j = 0; j < width; j++) {
                    float k = 0;
                    float part_of_h = 0;
                    for (int  m = i-1; m <= i+2; m++) {
                        for (int  n = j-1; n <= j+2; n++) {
                            if (n >= 0 && n < width && m >= 0 && m < height ) { // Проверка граничных условий
                                k += g(m, n, i, j, sigmad) * r(input[m*width + n], input[i*width + j], sigmad); // Нормирующая константа для предотвращения увеличения интенсивности
                                part_of_h += input[m*width + n] * g(m, n, i, j, sigmad) * r(input[m*width + n], input[i*width + j], sigmar); // Часть нового значения интенсивности
                            }
                        }
                    }
                    output[i*width+j] = part_of_h/k; // Новое значение интенсивности пикселей
                }
            }
        }
}

// Функция для расчета амплитуды Гаусса на GPU
__device__ float g_gpu(int x, int y, int x0, int y0, float sigmad) {
    return exp(- (pow((x - x0),2) - pow((y - y0),2))/pow(sigmad,2));
}


// Функция для расчета новой интенсивности без нормировочных коэффициентов на GPU
__device__ float r_gpu(int a, int a0, float sigmar) {
    return exp(- pow((a - a0),2) /pow(sigmar,2));
}

//Функция для фильтра Гаусса на GPU
__global__ void Bilateral_GPU(float* input, float* output, int width, int height, float sigmad, float sigmar) {
    int rows = blockIdx.y*blockDim.y+threadIdx.y; //Число строк для нитей в гриде
    int cols = blockIdx.x*blockDim.x+threadIdx.x; //Число столбцов для нитей в гриде

    // Проверка граничных условий
    if((rows >= 0) && (rows < height) && (cols >= 0) && (cols < width))
    {
        float k = 0;
        float part_of_h = 0;
        for (int  m = rows -1; m <= rows+1; m++) {
            for (int  n = cols-1; n <= cols+1; n++) {
                k += g_gpu(m, n, rows, cols, sigmad) * r_gpu(input[m*width + n], input[rows*width + cols], sigmad);  // Нормирующая константа для предотвращения увеличения интенсивности
                part_of_h += input[m*width + n] * g_gpu(m, n, rows, cols, sigmad) * r_gpu(input[m*width + n], input[rows*width + cols], sigmar); // Часть нового значения интенсивности
            }
        }
        output[rows*width+cols] = part_of_h/k; // Новое значение интенсивности пикселей
    }
}

int main() {
    BMP AnImage;
    AnImage.ReadFromFile("Image.bmp");
    int height = AnImage.TellHeight();
	int width = AnImage.TellWidth();

    float sigmad = 10000; // Параметр, определяющий амплитуду функции Гаусса
    float sigmar = 10000; // Константа ранг-фильтра

    float* imageArray = (float*)calloc(height * width, sizeof(float));
	float* outputCPU = (float*)calloc(height * width, sizeof(float));
	float* outputGPU = (float*)calloc(height * width, sizeof(float));
    float* outputDevice;
    float* inputDevice;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            imageArray[y * width + x] = AnImage.GetPixel(x, y).Red;
        }
    }

    auto start1 = std::chrono::steady_clock::now();

    Bilateral_CPU(imageArray, outputCPU, width, height, sigmad, sigmar);

    auto end1 = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration<float, std::chrono::seconds::period> (end1 - start1); //Расчет времени вычислений для CPU
    std::cout << "The time: " << elapsed.count() <<" seconds on CPU.\n";

    cudaEvent_t begin, stop;

    dim3 block_dim(32, 32);
    dim3 grid_dim((width + block_dim.x - 1) / block_dim.x,
			(height + block_dim.y - 1) / block_dim.y);

    cudaMalloc(&inputDevice, height * width * sizeof(float));

    cudaMemcpy(inputDevice, imageArray, width * height * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&outputDevice, height * width * sizeof(float));

    cudaEventCreate(&begin);
    cudaEventCreate(&stop);

    cudaEventRecord(begin);

	Bilateral_GPU << <grid_dim, block_dim >> > (inputDevice, outputDevice, width, height, sigmad, sigmar);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time = 0;

    cudaEventElapsedTime(&gpu_time, begin, stop);

    cudaMemcpy(outputGPU, outputDevice, height * width * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "The time: " << gpu_time/1000 <<" seconds on GPU.\n"; //Для перевода в секунды

    saveImage(outputGPU, width, height, true);
	saveImage(outputCPU, width, height, false);

    cudaEventDestroy(begin);
    cudaEventDestroy(stop);
    cudaFree(inputDevice);
    cudaFree(outputDevice);
}
