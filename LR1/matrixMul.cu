#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <omp.h>
#include <chrono>

//Функция для перемножения матриц на CPU
void matrix_Mul_CPU(int rowsA, int colsA, int colsB, const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C) {
    auto start1 = std::chrono::steady_clock::now();
    omp_set_num_threads(8); //Устанавливаем число потоков для параллельных вычислений
    #pragma omp parallel /*Уставновка структурного блока, вычисление которого происходит параллельно*/
        #pragma omp master //Выполнение только главным потоком
        {
            for (int i = 0; i < rowsA; i++)
            {
                for (int j = 0; j < colsB; j++)
                {
                    C[i*colsB + j] = 0;
                    for (int k = 0; k < colsA; k++)
                    {
                        C[i*colsB + j] += A[i*colsA + k] * B[k*colsB + j];
                    }
                }
            }
        }
    auto end1 = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration<float, std::chrono::seconds::period> (end1 - start1); //Расчет времени вычислений для CPU
    std::cout << "The time: " << elapsed.count() <<" seconds on CPU.\n";
    /*printf_s("%d\n", omp_get_max_threads( ));*/
}

//Функция для перемножения матриц на GPU
__global__ void matrix_Mul_GPU(int rowsA, int colsA, int colsB, double *A, double *B, double *C) {
    int rows = blockIdx.y*blockDim.y+threadIdx.y; //Число строк для нитей в гриде
    int cols = blockIdx.x*blockDim.x+threadIdx.x; //Число столбцов для нитей в гриде
    if (rows < rowsA && cols < colsB) { //Проверка на соответствие общего количества строк и столбцов с значениями для матриц, чтобы не выйти за пределы
         C[rows * colsB + cols] = 0;
        for (int i = 0; i < colsA; i++) {
            C[rows * colsB + cols] += A[rows * rowsA + i] * B[i * colsB + cols];
        }
    }
}

int main() {
    //Так как необходимо произвести перемножение от 100 до 2000, используем for с шагом 100
    for (int dimm = 100; dimm <= 2000; dimm+=100) {
        std::cout << "Dimension of matrix: " << dimm << ".\n"; //Выводим размерность матрицы

        //Определяем число строк и столбцов в матрицах
        int m = dimm;
        int n = dimm;
        int k = dimm;

        //Создаем матрицы
        std::vector<double> A(m*n);
        std::vector<double> B(n*k);
        std::vector<double> C(m*k);

        //Заполняем матрицы A и B случайными числами
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                A[i*n + j] = rand();
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                B[i*k + j] = rand();
            }
        }

        //Выполняем перемножение на CPU
        matrix_Mul_CPU(m, n, k, A, B, C);

        //Размер блока потоков
        dim3 block_dim(32, 32);

        //Определяем размеры грида, чтобы размер грида в нитях позволял вместить в себя матрицу целиком
        dim3 grid_dim((k + block_dim.x - 1) / block_dim.x, (m + block_dim.y - 1) / block_dim.y);

        //Создаем матрицы для вычисления на GPU и выделяем для них память
        double* da, * db, * dc;
        cudaMalloc(&da, m * n * sizeof(double));
        cudaMalloc(&db, n * k * sizeof(double));
        cudaMalloc(&dc, m * k * sizeof(double));

        //Переносим значения матриц для дальнейших расчетов с хоста на девайс
        cudaMemcpy(da, A.data(), m * n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(db, B.data(), n * k * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dc, C.data(), m * k * sizeof(double), cudaMemcpyHostToDevice);

        // Создание обработчика событий CUDA
        cudaEvent_t begin, stop;
        cudaEventCreate(&begin);
        cudaEventCreate(&stop);

        cudaEventRecord(begin);

        //Выполняем перемножение на GPU
        matrix_Mul_GPU<<<grid_dim,block_dim>>>(m, n, k, da, db, dc);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        //Создаем переменную, в которой потом будет находится время перемножения на GPU
        float gpu_time = 0;

        //Получаем время между двумя событиями
        cudaEventElapsedTime(&gpu_time, begin, stop);

        std::cout << "The time: " << gpu_time/1000 <<" seconds on GPU.\n"; //Для перевода в секунды

        //Переносим значения матрицы С с девайса на хост
        std::vector<double> hc(m * n, 0.0);
        cudaMemcpy(hc.data(), dc, m * k * sizeof(double), cudaMemcpyDeviceToHost);

        //Оценка эквивалентности результатов вычисления матрицы на CPU и GPU
        printf("Checking:\n");
        bool check = true;
        for (int i = 0; i < m * k; ++i) {
            if (C[i] != hc[i]) {
                check = false;
                break;
            }
        }
        if (check) {
            printf("SUCCESS\n");
        }
        else {
            printf("FAILURE\n");
        }

        //Освобождение ресурсов на GPU
        cudaFree(da);
        cudaFree(db);
        cudaFree(dc);

        cudaEventDestroy(begin);
        cudaEventDestroy(stop);
    }
    return 0;
}