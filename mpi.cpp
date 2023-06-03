/*#include<stdio.h>
#include<mpi.h>
#include<stdlib.h>
#include<math.h>
#define M 5

int main(int argc, char* argv[]) {
    int myid, numprocs, namelen, cpuname;
    MPI_Status status;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    int i, j, k, n, temp;
    int a[M][M], b[M];
    if (!myid) {

        for (i = 0; i < M - 1; i++) {
            printf("please input  the %d row :\n", i + 1);
            for (j = 0; j < M; j++) {

                scanf_s("%d", &a[i][j]);
            }
           // printf("\n");
        }
        MPI_Send(a, M * M, MPI_INT, 1, 99, MPI_COMM_WORLD);


    }

    else {

        MPI_Recv(a, M * M, MPI_INT, myid - 1, 99, MPI_COMM_WORLD, &status);
        printf("经过第%d次消元后的结果是\n", myid);
        for (j = myid; j < M - 1; j++) {
            temp = a[j][myid - 1] / a[myid - 1][myid - 1];
            for (k = 0; k < M; k++) {
                a[j][k] = a[j][k] - temp * a[myid - 1][k];
            }
        }


        for (i = 0; i < M - 1; i++) {
            for (j = 0; j < M; j++) {
                printf("%d ", a[i][j]);
            }
            printf("\n");
        }
        printf("\n");
        if (myid < M - 2) {
            MPI_Send(a, M * M, MPI_INT, myid + 1, 99, MPI_COMM_WORLD);
        }
        if (myid == M - 2) {
            for (i = M - 2; i + 1; i--) {
                temp = a[i][M - 1];
                for (j = M - 2; j > i; j--) {
                    temp = temp - a[i][j] * b[j];
                }
                b[i] = temp / a[i][i];
            }
            for (i = 0; i < M - 1; i++) {
                printf("x%d = %d\n", i + 1, b[i]);
            }
        }
    }

    MPI_Finalize();
}*/
#include<iostream>
#include<cmath>
#include<ctime>
#include <mpi.h>
//#include"Matrix.h"
class Matrix
{
public:

	int row;
	int column;

	double* memoryPool;
	double** p;

public:
	Matrix(int scale = 1) :row(scale), column(scale)
	{
		int dim = scale;
		int num = scale * scale;
		memoryPool = new double[num] {0};
		p = new double* [dim];
		for (int i = 0; i < dim; ++i)
			p[i] = memoryPool + i * scale;
	}
	Matrix(int _row, int _column) :row(_row), column(_column)
	{
		int num = row * column;
		memoryPool = new double[num] {0};
		p = new double* [row];
		for (int i = 0; i < row; ++i)
			p[i] = memoryPool + i * column;
	}
	~Matrix()
	{
		if (memoryPool) { delete[]memoryPool; }
		if (p) { delete[]p; }
	}
	double& operator()(int i, int j)const { return p[i][j]; }
	/*friend ostream& operator<<(ostream& out, const Matrix& obj)
	{
		for (int i = 0; i < obj.row; ++i)
		{
			for (int j = 0; j < obj.column; ++j)
				out << obj(i, j) << '\t';
			cout << endl;
		}
		return out;
	}*/
	Matrix(Matrix&& other)
	{
		row = other.row;
		column = other.column;
		memoryPool = other.memoryPool;
		p = other.p;
		other.memoryPool = nullptr;
		other.p = nullptr;
	}
	Matrix(const Matrix& obj)
	{
		//cout << "赋值构造函数" << endl;
		row = obj.row;
		column = obj.column;
		memoryPool = new double[row * column];
		p = new double* [row];
		for (int i = 0; i < row; ++i)
			p[i] = memoryPool + i * column;
		for (int i = 0; i < row; ++i)
			for (int j = 0; j < column; ++j)
				p[i][j] = obj(i, j);


	}
	Matrix& operator=(const Matrix& obj)
	{
		if (row != obj.row || column != obj.column)
		{
			if (memoryPool)delete[]memoryPool;
			if (p)delete[]p;
			row = obj.row;
			column = obj.column;
			memoryPool = new double[row * column];
			p = new double* [row];
			for (int i = 0; i < row; ++i)
				p[i] = memoryPool + i * column;
		}
		for (int i = 0; i < row; ++i)
			for (int j = 0; j < column; ++j)
				p[i][j] = obj(i, j);
		return *this;
	}
	Matrix& operator=(Matrix&& obj)
	{
		if (memoryPool) { delete[]memoryPool; }
		if (p) { delete[]p; }
		row = obj.row;
		column = obj.column;
		memoryPool = obj.memoryPool;
		p = obj.p;
		obj.memoryPool = nullptr;
		obj.p = nullptr;
		return *this;
	}
	void ranCreate(int val = 200)
	{
		srand(time(NULL));
		for (int i = 0; i < row; ++i)
			for (int j = 0; j < column; ++j)
				p[i][j] = (rand() % val) / 100.0;
	}
	Matrix operator*(const Matrix& obj)
	{
		Matrix tmp(row, obj.column);
		for (int i = 0; i < row; ++i)
			for (int j = 0; j < obj.column; ++j)
			{
				for (int k = 0; k < column; ++k)
					tmp(i, j) += p[i][k] * obj(k, j);
			}
		return tmp;
	}
};

#include"mpi.h"

using namespace std;

int myid, numprocs, masterNode;//进程标识号与进程总数，设为全局变量可在任意个函数中访问
int partRow, up, down;//各进程分配到的行数，相邻的上下两个进程
int dim = 1500;//问题规模
MPI_Comm partComm;//不能整除的部分在该通信域做一次scatter/gather即可
double start, finish;

void seqGaussSolver(Matrix A, Matrix b, Matrix& x)//串行 行主元高斯消去法
{
	int N = A.row, picked = 0;
	int* loc = new int[N] {0};
	for (int i = 0; i < N; ++i)
		loc[i] = i;
	for (int k = 0; k < N; ++k)
	{
		//找行主元
		double max = 0;
		for (int j = k; j < N; ++j)
		{
			if (abs(A(k, loc[j])) > max)
			{
				max = abs(A(k, loc[j]));
				picked = j;
			}
		}
		//picked为当前行k的主元，但当前行记录的主元为loc[k]
		int tmp = loc[k];
		loc[k] = loc[picked];
		loc[picked] = tmp;

		//以枢纽为基准开始消元
		for (int i = k + 1; i < N; ++i)
		{
			double t = A(i, loc[k]) / A(k, loc[k]);
			for (int j = k; j < N; ++j)
				A(i, loc[j]) -= t * A(k, loc[j]);
			b(i, 0) -= t * b(k, 0);
		}
	}
	for (int i = N - 1; i >= 0; --i)
	{
		x(loc[i], 0) = b(i, 0) / A(i, loc[i]);
		for (int j = 0; j < i; ++j)
			b(j, 0) -= x(loc[i], 0) * A(j, loc[i]);
	}
	//cout << x << endl;
	delete[]loc;

}
void parallelInit()
{
	partRow = dim / numprocs;
	int remainder = dim % numprocs;
	if (myid < remainder)++partRow;

	//进程成环
	up = myid - 1, down = myid + 1;
	if (myid == 0)up = numprocs - 1;
	if (myid == numprocs - 1)down = 0;

	//创建新的通信域，方便不能整除时的播撒数据
	MPI_Group worldGroup;
	MPI_Comm_group(MPI_COMM_WORLD, &worldGroup);
	MPI_Group partGroup;

	int* groupRank = new int[remainder];
	for (int i = 0; i < remainder; ++i)
		groupRank[i] = i;

	MPI_Group_incl(worldGroup, remainder, groupRank, &partGroup);
	MPI_Comm_create(MPI_COMM_WORLD, partGroup, &partComm);
	delete[]groupRank;
}
void distributeTask(const Matrix& lAb, Matrix& Ab)
{

	int counts = dim / numprocs;
	for (int i = 0; i < counts; ++i)
		MPI_Scatter(&lAb(i * numprocs, 0), dim + 1, MPI_DOUBLE, &Ab(i, 0), dim + 1, MPI_DOUBLE, masterNode, MPI_COMM_WORLD);
	if (counts != partRow)
		MPI_Scatter(&lAb(counts * numprocs, 0), dim + 1, MPI_DOUBLE, &Ab(partRow - 1, 0), dim + 1, MPI_DOUBLE, masterNode, partComm);


}
void gatherResult(const Matrix& partResult, Matrix& result)
{
	int counts = dim / numprocs;
	for (int i = 0; i < counts; ++i)
		MPI_Gather(&partResult(i, 0), 1, MPI_DOUBLE, &result(i * numprocs, 0), 1, MPI_DOUBLE, masterNode, MPI_COMM_WORLD);
	if (counts != partRow)
		MPI_Gather(&partResult(partRow - 1, 0), 1, MPI_DOUBLE, &result(counts * numprocs, 0), 1, MPI_DOUBLE, masterNode, partComm);
}
void parallelGaussSolver(const Matrix& _Ab, Matrix& partResult, Matrix& result)
{
	Matrix Ab = _Ab;
	MPI_Request request;
	MPI_Status status;
	int* loc = new int[dim + 1]{ 0 };
	for (int i = 0; i < dim + 1; ++i)
		loc[i] = i;
	int picked = 0;

	Matrix sta(1, dim + 1);//定义基准行元素

	double s1, s2, dur = 0.0;
	double rec1, rec2, dur2 = 0;

	//群集通信函数实现并行
	int p = 0;
	for (int k = 0; k < dim; ++k)
	{
		int source = k % numprocs;
		if (myid == source)
		{
			double max = 0.0;
			for (int j = k; j < dim; ++j)
			{
				if (abs(Ab(p, loc[j])) > max)
				{
					max = abs(Ab(p, loc[j]));
					picked = j;
				}
			}

			for (int j = 0; j < dim + 1; ++j)
				sta(0, j) = Ab(p, j);
			p++;
		}

		if (myid == masterNode)
			rec1 = MPI_Wtime();
		MPI_Bcast(&sta(0, 0), dim + 1, MPI_DOUBLE, source, MPI_COMM_WORLD);
		MPI_Bcast(&picked, 1, MPI_DOUBLE, source, MPI_COMM_WORLD);
		if (myid == masterNode)
		{
			rec2 = MPI_Wtime();
			dur2 += rec2 - rec1;
		}
		int tmp = loc[k];
		loc[k] = loc[picked];
		loc[picked] = tmp;
		if (myid == masterNode)
			s1 = MPI_Wtime();
		for (int i = p; i < partRow; ++i)
		{
			double t = Ab(i, loc[k]) / sta(0, loc[k]);
			for (int j = k; j < dim + 1; ++j)
				Ab(i, loc[j]) -= t * sta(0, loc[j]);
		}
		if (myid == masterNode)
		{
			s2 = MPI_Wtime();
			dur += s2 - s1;
		}
	}

	//非阻塞通信实现异步计算
	/*int p = 0;
	for (int k = 0; k < dim; ++k)
	{
		int source = k % numprocs;
		if (myid == source)
		{
			double max = 0.0;
			for (int j = k; j < dim; ++j)
			{
				if (abs(Ab(p, loc[j])) > max)
				{
					max = abs(Ab(p, loc[j]));
					picked = j;
				}
			}
			MPI_Isend(&Ab(p, 0), dim + 1, MPI_DOUBLE, down, picked, MPI_COMM_WORLD, &request);

			for (int j = 0; j < dim + 1; ++j)
				sta(0, j) = Ab(p, j);

			p++;
		}
		else
		{
			MPI_Irecv(&sta(0, 0), dim + 1, MPI_DOUBLE, up, MPI_ANY_TAG, MPI_COMM_WORLD, &request);
			if (myid == masterNode)
				rec1 = MPI_Wtime();
			MPI_Wait(&request, &status);

			if (myid == masterNode)
			{
				rec2 = MPI_Wtime();
				dur2 += rec2 - rec1;
			}
			picked = status.MPI_TAG;
			if (down != source)
				MPI_Isend(&sta(0, 0), dim + 1, MPI_DOUBLE, down, picked, MPI_COMM_WORLD, &request);
		}

		if (myid == masterNode)
			s1 = MPI_Wtime();
		int tmp = loc[k];
		loc[k] = loc[picked];
		loc[picked] = tmp;


		for (int i = p; i < partRow; ++i)
		{

			double t = Ab(i, loc[k]) / sta(0, loc[k]);

			for (int j = k; j < dim + 1; ++j)
				Ab(i, loc[j]) -= t * sta(0, loc[j]);
		}

		if (myid == masterNode)
		{
			s2 = MPI_Wtime();
			dur += s2 - s1;
		}
	}*/
	if (myid == masterNode)
	{
		cout << "计算时间：" << dur << endl;
		cout << "通信时间：" << dur2 << endl;
	}

	//并行回代
	int rp = partRow - 1;
	double val = 0;
	for (int k = dim - 1; k >= 0; --k)
	{
		int source = k % numprocs;
		if (myid == source)
		{
			partResult(rp, 0) = Ab(rp, dim) / Ab(rp, loc[k]);
			val = partResult(rp--, 0);
		}
		MPI_Bcast(&val, 1, MPI_DOUBLE, source, MPI_COMM_WORLD);
		for (int j = rp; j >= 0; --j)
			Ab(j, dim) -= val * Ab(j, loc[k]);
	}

	gatherResult(partResult, result);


	//解向量的元素顺序重排
	if (myid == masterNode)
	{
		Matrix rtmp = result;
		for (int i = 0; i < dim; ++i)
			result(loc[i], 0) = rtmp(i, 0);
		//cout << result << endl;
	}
	delete[]loc;
}

int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);//获得进程数
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);//获得当前进程标识号0,1,2,3,....,numprocs - 1

	masterNode = 0;

	Matrix lA, lb, lAb, result;//分别是系数矩阵，等号右端常数向量，增广矩阵，解向量
	if (myid == masterNode)
	{
		/*cout << "请输入问题规模，系数矩阵及右端常数向量:" << endl;
		cin >> dim;*/
		lA = Matrix(dim, dim);
		lb = Matrix(dim, 1);
		lAb = Matrix(dim, dim + 1);
		result = Matrix(dim, 1);
		srand((unsigned)time(NULL));
		for (int i = 0; i < dim; i++)
		{
			for (int j = 0; j < dim; j++)
			{

				lA.p[i][j] = (float)(rand() % 10000) / 100.00;
				//scanf("%lf", &A[i][j]);
			}
		}
		//lA.MatrixInit("data_A"), lb.MatrixInit("data_b");//矩阵文件输入
		//lA.ranCreate(),lb.ranCreate(); //自动生成矩阵

		for (int i = 0; i < dim; i++)
		{

			lb.p[i][0] = (float)(rand() % 10000) / 100.00;
			//scanf("%lf", &b[i]);
		}

		for (int i = 0; i < dim; i++)
		{
			for (int j = 0; j < dim; j++)
			{

				lAb.p[i][j] = lA.p[i][j];
				//scanf("%lf", &A[i][j]);
			}
			lAb.p[i][dim] = lb.p[i][0];
		}
		//lAb = lA.merge(lb, COLUMN);//A，b合并得增广矩阵
		if (numprocs == 1)
		{
			start = MPI_Wtime();
			seqGaussSolver(lA, lb, result);
			finish = MPI_Wtime();
			cout << finish - start << endl;
			MPI_Finalize();
			//system("pause");
			return 0;
		}
	}
	MPI_Bcast(&dim, 1, MPI_INT, masterNode, MPI_COMM_WORLD);//将问题规模广播至各个进程
	parallelInit();//并行初始化工

	Matrix Ab(partRow, dim + 1), partResult(partRow, 1);
	distributeTask(lAb, Ab);//将增广矩阵lAb交叉散射到每个进程的Ab中
	start = MPI_Wtime();
	parallelGaussSolver(Ab, partResult, result);
	finish = MPI_Wtime();
	if (myid == masterNode)
	{

		cout << finish - start << endl;
		//cout << result << endl;
	}


	MPI_Finalize();
	return 0;
}