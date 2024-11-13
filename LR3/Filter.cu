#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include "EasyBMP.h"
#include <stdio.h>
#include <omp.h>
#include <chrono>

void saveImage(unsigned char* image, int width, int height, bool method) {
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

//Функция для медианного фильтра на CPU
void Noise_Removal_CPU(unsigned char* input, unsigned char* output, int width, int height) {
    unsigned char filter_window[9] = {0,0,0,0,0,0,0,0,0}; //Создание окна фильтрации
    omp_set_num_threads(8); //Устанавливаем число потоков для параллельных вычислений
    #pragma omp parallel /*Уставновка структурного блока, вычисление которого происходит параллельно*/
        #pragma omp master //Выполнение только главным потоком
        {
            for (int rows = 0; rows < height; rows++) {
                for (int cols = 0; cols < width; cols++) {
                     if((rows==0) || (cols==0) || (rows==height-1) || (cols==width-1)) {
                        output[rows*width+cols] = 0; //Граничные условия
                     } else {
                        for (int x = 0; x < 3; x++) {
                            for (int y = 0; y < 3; y++){
                                filter_window[x*3+y] = input[(rows+x-1)*width+(cols+y-1)];   //Расчет окна фильтрации
                            }
                        }
                     }
                    //Пузырьковая сортировка массива
                    for (int i = 0; i < 9; i++) {
                        for (int j = i + 1; j < 9; j++) {
                            if (filter_window[i] > filter_window[j]) {
                                //Меняем значения
                                char tmp = filter_window[i];
                                filter_window[i] = filter_window[j];
                                filter_window[j] = tmp;
                            }
                        }
                    }
                    output[rows*width+cols] = filter_window[4];   //Выходной элемент - средний элемент (медиана)
                }
            }
        }
}

//Функция для медианного фильтра на GPU
__global__ void Noise_Removal_GPU(unsigned char* input, unsigned char* output, int width, int height) {
    int rows = blockIdx.y*blockDim.y+threadIdx.y; //Число строк для нитей в гриде
    int cols = blockIdx.x*blockDim.x+threadIdx.x; //Число столбцов для нитей в гриде
    unsigned char filter_window[9] = {0,0,0,0,0,0,0,0,0}; //Создание окна фильтрации

    if((rows==0) || (cols==0) || (rows==height-1) || (cols==width-1))
    {
        output[rows*width+cols] = 0; //Граничные условия
    } else {
        for (int x = 0; x < 3; x++) {
			for (int y = 0; y < 3; y++){
				filter_window[x*3+y] = input[(rows+x-1)*width+(cols+y-1)];   //Расчет окна фильтрации
			}
		}
        //Пузырьковая сортировка массива
		for (int i = 0; i < 9; i++) {
			for (int j = i + 1; j < 9; j++) {
				if (filter_window[i] > filter_window[j]) {
					//Меняем значения
					char tmp = filter_window[i];
					filter_window[i] = filter_window[j];
					filter_window[j] = tmp;
				}
			}
		}
		output[rows*width+cols] = filter_window[4];   //Выходной элемент - средний элемент (медиана)
    }
}

int main() {
    BMP AnImage;
    AnImage.ReadFromFile("Image4.bmp");
    int height = AnImage.TellHeight();
	int width = AnImage.TellWidth();

    unsigned char* imageArray = (unsigned char*)calloc(height * width, sizeof(unsigned char));
	unsigned char* outputCPU = (unsigned char*)calloc(height * width, sizeof(unsigned char));
	unsigned char* outputGPU = (unsigned char*)calloc(height * width, sizeof(unsigned char));
    unsigned char* outputDevice;
    unsigned char* inputDevice;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            imageArray[y * width + x] = AnImage.GetPixel(x, y).Red;
        }
    }

    auto start1 = std::chrono::steady_clock::now();

    Noise_Removal_CPU(imageArray, outputCPU, width, height);

    auto end1 = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration<float, std::chrono::seconds::period> (end1 - start1); //Расчет времени вычислений для CPU
    std::cout << "The time: " << elapsed.count() <<" seconds on CPU.\n";

    cudaEvent_t begin, stop;

    dim3 block_dim(32, 32);
    dim3 grid_dim((width + block_dim.x - 1) / block_dim.x,
			(height + block_dim.y - 1) / block_dim.y);

    cudaMalloc(&inputDevice, height * width * sizeof(unsigned char));

    cudaMemcpy(inputDevice, imageArray, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    cudaMalloc(&outputDevice, height * width * sizeof(unsigned char));

    cudaEventCreate(&begin);
    cudaEventCreate(&stop);

    cudaEventRecord(begin);

	Noise_Removal_GPU << <grid_dim, block_dim >> > (inputDevice, outputDevice, width, height);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time = 0;

    cudaEventElapsedTime(&gpu_time, begin, stop);

    cudaMemcpy(outputGPU, outputDevice, height * width * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    std::cout << "The time: " << gpu_time/1000 <<" seconds on GPU.\n"; //Для перевода в секунды

    saveImage(outputGPU, width, height, true);
	saveImage(outputCPU, width, height, false);

    cudaEventDestroy(begin);
    cudaEventDestroy(stop);
    cudaFree(inputDevice);
    cudaFree(outputDevice);
}
