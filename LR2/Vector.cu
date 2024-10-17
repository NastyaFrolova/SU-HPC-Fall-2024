#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <omp.h>
#include <chrono>

//Функция для сложения элементов векторов на CPU
int vector_CPU(const std::vector<int>& vec, int size, std::ofstream& out) {
    int sum = 0;
    auto start1 = std::chrono::steady_clock::now();
    omp_set_num_threads(8); //Устанавливаем число потоков для параллельных вычислений
    #pragma omp parallel /*Уставновка структурного блока, вычисление которого происходит параллельно*/
        #pragma omp master //Выполнение только главным потоком
        {
            for (int i = 0; i < size; i++)
            {
                sum += vec[i];
            }
        }
    auto end1 = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration<float, std::chrono::microseconds::period> (end1 - start1); //Расчет времени вычислений для CPU
    std::cout << "The time: " << elapsed.count() <<" microseconds on CPU.\n";
    out << elapsed.count();
    /*printf_s("%d\n", omp_get_max_threads( ));*/
    return sum;
}

//Функция для сложения элементов векторов на GPU
__global__ void vector_GPU(int* vec, int* sum, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        sum[idx] = vec[idx];
    }
}

int main() {
    std::ofstream out;
    out.open("Output.txt");
    if (out.is_open())
    {
        for (int vec_size = 1000; vec_size <= 1000000; vec_size+=1000) {
            std::cout << "Vector size: " << vec_size << ".\n";
            std::vector<int> vec(vec_size);
            std::fill_n(vec.begin(), vec.size(), 5);

            int result_CPU = vector_CPU(vec, vec.size(), out);
            std::cout << "Summ of elements: " <<result_CPU <<" on CPU.\n";
            out << " " << result_CPU;
            int* vec_GPU, * result_GPU;
            cudaMalloc(&vec_GPU,  vec.size() * sizeof(int));
            cudaMalloc(&result_GPU,  vec.size() * sizeof(int));

            cudaMemcpy(vec_GPU, vec.data(), vec_size * sizeof(int), cudaMemcpyHostToDevice);

            cudaEvent_t begin, stop;
            cudaEventCreate(&begin);
            cudaEventCreate(&stop);

            cudaEventRecord(begin);
            vector_GPU<<<1000,1000>>>(vec_GPU, result_GPU, vec.size());
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            //Создаем переменную, в которой потом будет находится время перемножения на GPU
            float gpu_time = 0;

            //Получаем время между двумя событиями
            cudaEventElapsedTime(&gpu_time, begin, stop);

            std::cout << "The time: " << gpu_time*1000 <<" microseconds on GPU.\n"; //Для перевода в секунды
            out << " " << gpu_time*1000;
            std::vector<int> res(vec_size);
            cudaMemcpy(res.data(), result_GPU, vec.size() * sizeof(int), cudaMemcpyDeviceToHost);
            int sum = 0;
            for (int i = 0; i < res.size(); i++) {
                sum += res[i];
            }
            std::cout << "Summ of elements: " << sum <<" on GPU.\n";
            out << " " << sum;
            cudaFree(vec_GPU);
            cudaFree(result_GPU);

            cudaEventDestroy(begin);
            cudaEventDestroy(stop);
            out << std::endl;
        }
    }
    out.close();
    return 0;
}