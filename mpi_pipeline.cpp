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

void mpi_pipeline()
{
    double start_time = 0;
    double end_time;
    int m_size = 0;
    int m_rank = 0;
    int i = 0;
    int j = 0;
    int k = 0;
    int r = 0;
    int addr = 0;
    float(*space)[ROW] = NULL;
    float local[ROW];
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &m_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    int* thread_count = new int[m_size];
    int* s_count = new int[m_size + 1];
    fill(thread_count, thread_count + ROW % m_size, (int)ceil((float)ROW / m_size) * ROW);
    fill(thread_count + ROW % m_size, thread_count + m_size, ROW / m_size * ROW);
    for (i = 0; i < m_size; i++)
    {
        s_count[i] = addr;
        addr += thread_count[i];
    }
    s_count[m_size] = addr;
    space = new float[thread_count[m_rank] / ROW][ROW];
    if (m_rank == 0)
    {
        init();
        start_time = MPI_Wtime();
    }
    MPI_Scatterv(matrix, thread_count, s_count, MPI_FLOAT, space, thread_count[m_rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
    for (k = 0; k < s_count[m_rank] / ROW; k++)
    {
        MPI_Recv(local, ROW, MPI_FLOAT, m_rank - 1, k, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (m_rank != m_size - 1)
        {
            MPI_Send(local, ROW, MPI_FLOAT, m_rank + 1, k, MPI_COMM_WORLD);
        }
        for (i = 0; i < thread_count[m_rank] / ROW; i++)
        {
            for (j = k + 1; j < ROW; j++)
            {
                space[i][j] -= space[i][k] * local[j];
            }
            space[i][k] = 0;
        }
    }
    for (k = s_count[m_rank] / ROW; k < s_count[m_rank + 1] / ROW; k++)
    {
        int myRow = k - s_count[m_rank] / ROW;
        for (j = k + 1; j < ROW; j++)
        {
            space[myRow][j] /= space[myRow][k];
        }
        space[myRow][k] = 1.0;
        if (m_rank != m_size - 1)
        {
            MPI_Send(space[myRow], ROW, MPI_FLOAT, m_rank + 1, k, MPI_COMM_WORLD);
        }
        for (r = myRow + 1; r < thread_count[m_rank] / ROW; r++)
        {
            for (j = k + 1; j < ROW; j++)
            {
                space[r][j] -= space[r][k] * space[myRow][j];
            }
            space[r][k] = 0;
        }
    }
    MPI_Gatherv(space, thread_count[m_rank], MPI_FLOAT, matrix, thread_count, s_count, MPI_FLOAT, 0, MPI_COMM_WORLD);
    if (m_rank == 0)
    {
        end_time = MPI_Wtime();
        cout << "mpi_pipeline:" << (end_time - start_time) * 1000 << "ms" << endl;
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
    cout << "mpi_pipeline: ";
    //gettime(plain); 
    mpi_pipeline();
    return 0;
}