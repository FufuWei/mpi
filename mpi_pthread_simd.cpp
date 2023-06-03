#include <pthread.h>
#include <omp.h>
#include <iostream>
#include <cmath>
#include <arm_neon.h>
#include <semaphore.h>
#include <stdio.h>
#include <sys/time.h>
#include <algorithm>
#include <mpi.h>
#define ROW 1024
#define TASK 8
#define INTERVAL 10000
using namespace std;
float matrix[ROW][ROW];
float revmat[ROW][ROW];
typedef long long ll;
typedef struct {
    int k;
    int t_id;
}threadParam_t;

sem_t sem_leader;
sem_t sem_Divsion[32];
sem_t sem_Elimination[32];
pthread_barrier_t division;
pthread_barrier_t elemation;
int NUM_THREADS = 8;
int remain = ROW;
pthread_mutex_t remainLock;

void init()
{
    for (int i = 0; i < ROW; i++)
    {
        for (int j = 0; j < i; j++)
        {
            matrix[i][j] = 0;
        }
        for (int j = i; j < ROW; j++)
        {
            matrix[i][j] = rand() % 10000 + 1;
        }
    }
    for (int i = 0; i < 2000; i++)
    {
        int row1 = rand() % ROW;
        int row2 = rand() % ROW;
        int judge = rand() % 2;
        if (judge == 1)
        {
            for (int j = 0; j < ROW; j++)
            {
                matrix[row1][j] += matrix[row2][j] * (rand() % 100);
            }
        }
        else
        {
            for (int j = 0; j < ROW; j++)
            {
                matrix[row1][j] -= matrix[row2][j] * (rand() % 100);
            }
        }
    }
}

void mpi_omp_SIMD()
{
    double start_time = 0;
    double end_time;
    int m_size = 0;
    int m_rank = 0;
    int i = 0;
    int j = 0;
    int k = 0;
    float32x4_t diver, divee, mult1, mult2, sub1;
    MPI_Status status;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &m_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    int r1 = (ROW - ROW % m_size) / m_size * m_rank;
    int r2 = (ROW - ROW % m_size) / m_size * (m_rank + 1);
    if (ROW - r2 < (ROW - ROW % m_size) / m_size)
    {
        r2 = ROW;
    }
    if (m_rank == 0)
    {
        init();
    }
    start_time = MPI_Wtime();
#pragma omp parallel num_threads(NUM_THREADS), shared(matrix), private(i,j,k,diver,divee,mult1,mult2,sub1,m_size,m_rank)
    for (k = 0; k < ROW; ++k)
    {
        if (k >= r1 && k <= r2)
        {
            diver = vld1q_dup_f32(&matrix[k][k]);
#pragma omp single
            for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)
            {
                matrix[k][j] = matrix[k][j] / matrix[k][k];
            }
            for (; j < ROW; j += 4)
            {
                divee = vld1q_f32(&matrix[k][j]);
                divee = vdivq_f32(divee, diver);
                vst1q_f32(&matrix[k][j], divee);
            }
#pragma omp barrier
            matrix[k][k] = 1.0;
            for (j = 0; j < m_size; ++j)
            {
                MPI_Send(&matrix[k][0], ROW, MPI_FLOAT, j, 1, MPI_COMM_WORLD);
            }
        }
        else
        {
            MPI_Recv(&matrix[k][0], ROW, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &status);
        }
#pragma omp for schedule(dynamic)
        for (i = max(r1, k + 1); i < r2; ++i)
        {
            mult1 = vld1q_dup_f32(&matrix[i][k]);
            for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)
            {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            for (; j < ROW; j += 4)
            {
                sub1 = vld1q_f32(&matrix[i][j]);
                mult2 = vld1q_f32(&matrix[k][j]);
                mult2 = vmulq_f32(mult1, mult2);
                sub1 = vsubq_f32(sub1, mult2);
                vst1q_f32(&matrix[i][j], sub1);
            }
            matrix[i][k] = 0.0;
        }
#pragma omp barrier
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (m_rank == 0)
    {
        end_time = MPI_Wtime();
        cout << "mpi_omp_SIMD:" << (end_time - start_time) * 1000 << "ms" << endl;
    }
    MPI_Finalize();
}

void gettime(void (*func)())
{
    timeval tv_begin, tv_end;
    int counter(0);
    double time = 0;
    while (INTERVAL > time)
    {
        init();
        gettimeofday(&tv_begin, 0);
        func();
        gettimeofday(&tv_end, 0);
        counter++;
        time += ((ll)tv_end.tv_sec - (ll)tv_begin.tv_sec) * 1000.0 + ((ll)tv_end.tv_usec - (ll)tv_begin.tv_usec) / 1000.0;
    }
    cout << time / counter << "ms" << '\n';
}

void gettime(void (*func)(void* (*threadFunc)(void*)), void* (*threadFunc)(void*))
{
    timeval tv_begin, tv_end;
    int counter(0);
    double time = 0;
    while (INTERVAL > time)
    {
        init();
        gettimeofday(&tv_begin, 0);
        func(threadFunc);
        gettimeofday(&tv_end, 0);
        counter++;
        time += ((ll)tv_end.tv_sec - (ll)tv_begin.tv_sec) * 1000.0 + ((ll)tv_end.tv_usec - (ll)tv_begin.tv_usec) / 1000.0;
    }
    cout << time / counter << "ms" << '\n';
}

int main()
{
    cout << "mpi_omp_simd: ";
    //gettime(plain); 
    mpi_omp_SIMD();
    return 0;
}