//使用高斯顺序消元法求解线性方程组

#include<iostream>
#include "math.h"
using namespace std;

double** A, * b, * x;
unsigned int RANK = 4;
unsigned int makematrix()
{
	unsigned int r, c;

	printf("请输入矩阵行列数，用空格隔开：");
	scanf("%d %d", &r, &c);

	A = (double**)malloc(sizeof(double*) * r);//创建一个指针数组，把指针数组的地址赋值给a ,*r是乘以r的意思
	for (int i = 0; i < r; i++)
		A[i] = (double*)malloc(sizeof(double) * c);//给第二维分配空间
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++)
			A[i][j] = 0.0;
	}

	b = (double*)malloc(sizeof(double) * r);
	for (int i = 0; i < r; i++)
	{
		b[i] = 0.0;
	}
	x = (double*)malloc(sizeof(double) * c);
	for (int i = 0; i < c; i++)
	{
		x[i] = 0.0;
	}

	return r;//一般都是输入方阵，返回行数也阔以
}

void getmatrix(void)//输入矩阵并呈现
{
	//printf("按行从左到右依次生成系数矩阵A，不同元素用空格隔开\n");
	srand((unsigned)time(NULL));
	for (int i = 0; i < RANK; i++)
	{
		for (int j = 0; j < RANK; j++)
		{
			
			A[i][j] = (float)(rand() % 10000) / 100.00;
			//scanf("%lf", &A[i][j]);
		}
	}
	/*printf("系数矩阵如下\n");
	for (int i = 0; i < RANK; i++)
	{
		for (int j = 0; j < RANK; j++)
		{
			printf("%g\t", A[i][j]);
		}
		printf("\n");
	}*/
	//printf("请按从上到下依次输入常数列b，不同元素用空格隔开\n");
	for (int i = 0; i < RANK; i++)
	{
		
		b[i] = (float)(rand() % 10000) / 100.00;
		//scanf("%lf", &b[i]);
	}
	/*printf("常数列如下\n");
	for (int i = 0; i < RANK; i++)
	{
		printf("%g\t", b[i]);
	}printf("\n");*/
}

void Gauss_calculation(void)//Gauss消去法解线性方程组
{
	double get_A = 0.0;
	//printf("利用以上A与b组成的增广阵进行高斯消去法计算方程组\n");
	for (int i = 1; i < RANK; i++)
	{
		for (int j = i; j < RANK; j++)
		{
			get_A = A[j][i - 1] / A[i - 1][i - 1];
			b[j] = b[j] - get_A * b[i - 1];
			for (int k = i - 1; k < RANK; k++)
			{
				A[j][k] = A[j][k] - get_A * A[i - 1][k];
			}
		}
	}
	/printf("顺序消元后的上三角系数增广矩阵如下\n");
	for (int i = 0; i < RANK; i++)
	{
		for (int j = 0; j < RANK; j++)
		{
			printf("%g\t", A[i][j]);
		}
		printf("    %g", b[i]);
		printf("\n");
	}
	printf("利用回代法求解上三角方程组，解得：\n");

	for (int i = 0; i < RANK; i++)
	{
		double get_x = 0.0;
		for (int j = 0; j < RANK; j++)
		{
			get_x = get_x + A[RANK - 1 - i][j] * x[j];//把左边全部加起来了，下面需要多减去一次Xn*Ann
		}
		x[RANK - 1 - i] = (b[RANK - 1 - i] - get_x + A[RANK - 1 - i][RANK - 1 - i] * x[RANK - 1 - i]) / A[RANK - 1 - i][RANK - 1 - i];
	}
	for (int i = 0; i < RANK; i++)
	{
		printf("x%d = %g\n", i + 1, x[i]);
	}
	printf("计算完成，按回车退出程序或按1重新运算\n");
}

int main()
{
	struct timeval startTime, stopTime;
	
_again:
	gettimeofday(&startTime, NULL);
	RANK = makematrix();
	getmatrix();
	Gauss_calculation();
	gettimeofday(&stopTime, NULL);
	double time = (stopTime.tv_sec - startTime.tv_sec) * 1000 +
		(double)(stopTime.tv_usec - startTime.tv_usec) * 0.001;
	cout << time << endl;
	getchar();

	if ('1' == getchar())
		goto _again;
	return 0;
}


