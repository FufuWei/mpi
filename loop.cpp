#include <iostream>
#include <mpi.h>
#include <cmath>
#include <sys/time.11h>

#define _TEST

using namespace std;

// 涉及到的变量
int N;//矩阵规模
const int L = 100;
int LOOP = 1;
float** origin;//原始数据
float** matrix = nullptr;//矩阵数据(会在运算过程中变化)


void init_data();//初始化原始数据
void init_matrix();//为计算矩阵赋值
double MPI_cycle();
void print_matrix();//打印结果矩阵

void test(int);//测试函数

void print_result(double);//打印结果
// 初始化数据，这里初始为了一个对称矩阵
void init_data()
{
    origin = new float* [N];
    matrix = new float* [N];
    auto* tmp = new float[N * N];
    for (int i = 0; i < N; i++)
    {
        origin[i] = new float[N];
        matrix[i] = tmp + i * N;
    }
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matrix[i][j] = 0;
            origin[i][j] = 0;
        }
    }
    //为上三角赋值
    for (int i = 0; i < N; i++)
    {
        for (int j = i; j < N; j++)
        {
            origin[i][j] = rand() * 1.0 / RAND_MAX * L;
        }
    }
    //把初始矩阵设置为对称矩阵
    for (int i = 0; i < N - 1; i++)
    {
        for (int j = i + 1; j < N; j++)
        {
            for (int k = 0; k < N; k++)
            {
                origin[j][k] += origin[i][k];
            }
        }
    }
}

// 用初始矩阵来初始化待计算矩阵，每个计算开始前都要重新赋值，保证运算数据是一样的
void init_matrix()
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matrix[i][j] = origin[i][j];
        }
    }
}
int main() {
    MPI_Init(nullptr, nullptr);
#ifdef _TEST
    LOOP = 1;
    test(100);
    test(500);
    test(1000);
    test(1500);
    test(2000);
#endif
#ifdef _PRINT
    test(10);
#endif
    MPI_Finalize();
    return 0;
}



// MPI_cycle 循环分配
double MPI_cycle() {
    double start_time, end_time;

    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // 只有是0号进程，才进行初始化工作
    if (rank == 0) {
        init_matrix();
    }
    start_time = MPI_Wtime();
    int task_num = rank < N% size ? N / size + 1 : N / size;
    // 0号进程负责任务的初始分发工作
    auto* buff = new float[task_num * N];
    if (rank == 0) {
        for (int p = 1; p < size; p++) {
            for (int i = p; i < N; i += size) {
                for (int j = 0; j < N; j++) {
                    buff[i / size * N + j] = matrix[i][j];
                }
            }
            int count = p < N% size ? N / size + 1 : N / size;
            MPI_Send(buff, count * N, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
        }
    }
    // 非0号进程负责任务的接收工作
    else {
        MPI_Recv(&matrix[rank][0], task_num * N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < task_num; i++) {
            for (int j = 0; j < N; j++) {
                matrix[rank + i * size][j] = matrix[rank + i][j];
            }
        }
    }
    // 做消元运算
    for (int k = 0; k < N; k++) {
        // 如果除法操作是本进程负责的任务，并将除法结果广播
        if (k % size == rank) {
            for (int j = k + 1; j < N; j++) {
                matrix[k][j] /= matrix[k][k];
            }
            matrix[k][k] = 1;
            for (int p = 0; p < size; p++) {
                if (p != rank) {
                    MPI_Send(&matrix[k][0], N, MPI_FLOAT, p, 1, MPI_COMM_WORLD);
                }
            }
        }
        // 其余进程接收除法行的结果
        else {
            MPI_Recv(&matrix[k][0], N, MPI_FLOAT, k % size, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // 进行消元操作
        int begin = N / size * size + rank < N ? N / size * size + rank : N / size * size + rank - size;
        for (int i = begin; i > k; i -= size) {
            for (int j = k + 1; j < N; j++) {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
    end_time = MPI_Wtime();
    return (end_time - start_time) * 1000;
}



// 显示结果矩阵
void print_matrix() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << matrix[i][j];
        }
        cout << endl;
    }
}

// 测试函数
void test(int n) {
    N = n;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        cout << N;
    }
    struct timeval start {};
    struct timeval end {};
    double time = 0;
    init_data();
    //cycle分配
    time = 0;
    for (int i = 0; i < LOOP; i++)
    {
        time += MPI_cycle();
    }
    if (rank == 0)
    {
        cout << "," << time / LOOP;
        print_result(time);
    }
    cout << endl;
}

// 结果显示
void print_result(double time) {
#ifdef _TEST
#endif
#ifdef _PRINT
    print_matrix();
#endif
}
