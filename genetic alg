#include "cuda_runtime.h"	 
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <ctime>
#include <cstdlib> 

using namespace std;

const int NumberOfPoint = 500; //количество точек
const int NumberOfIndividov = 1000; //кол-во индивидов в выборке
const int MathMutation = 5; //мутации 
const double dispersionMutation = 5.0f; //максимальная мутация
const int powCount = 3;
const double randMaxCount = 20.0f; //максимальный разброс рандома
const int KolOfPokoleni = 30; //максимальное кол-во поколений


__global__ void Errors(double* points, double* individs, double* errors, int powCount, int NumberOfPoint) //проверяем поколение
{
	int id = threadIdx.x; //разделяем на поотоки
	double ans = 0; //ответ
	int x = 1;
	for (int i = 0; i < NumberOfPoint; i++) //перебираем точки
	{
		for (int j = 0; j < powCount; j++)
		{
			for (int k = 0; k < j; k++)
			    x *= i;

			x *= individs[id * powCount + j]; //считаем поколение
			ans += x;
			x = 1;
		}
		ans = points[i] - ans; //считаем ошибку
		errors[id] += sqrt(ans * ans);
		ans = 0;
	}
}




void testErrors(double* points, double* individs, double* errors, int powCount, int NumberOfPoint, int Random) //тоже самое что и верхняя функция, проверяем через нее индивидов
{
	for (int id = 0; id < NumberOfIndividov; id++)
	{
		double ans = 0.0f;
		errors[id] = 0.0f;
		int x = 0;
		for (int i = 0; i < NumberOfPoint; i++)
		{
			for (int j = 0; j < powCount; j++)
			{
				x = pow(i, j);
				x *= individs[id * powCount + j];
				ans += x;
				x = 0;
			}
			ans = points[i] - ans;
			errors[id] += sqrt(ans * ans);
			ans = 0;
		}
	}
}


double Random(double a, double b) { //рандомно заполняем первое поколение
	double random = ((double)rand()) / (double)RAND_MAX;
	double d = b - a;
	double r = random * d;
	return(a + r);
}

void cpu() 
{ //обработка на cpu 
	double* pointsH = new double[NumberOfPoint]; //заводим массив точек
	for (int i = 0; i < NumberOfPoint; i++) pointsH[i] = Random(0, 20); //заполняем массив рандомно 

	double* individumsH = new double[NumberOfIndividov * powCount]; //заводим массив на индивидов
	for (int i = 0; i < NumberOfIndividov * powCount; i++) individumsH[i] = Random(0, randMaxCount); //заполняем массив рандомно

	double* errorsH = new double[NumberOfIndividov]; //заводим массив ошибок для каждого индивида
	for (int i = 0; i < NumberOfIndividov; i++) errorsH[i] = 1000; //ставим максимальную ошибку

	unsigned int start_time = clock(); // начальное время

	for (int pokolenie = 0; pokolenie < KolOfPokoleni; pokolenie++) //цикл перебираем поколения
	{
		testErrors(pointsH, individumsH, errorsH, powCount, NumberOfPoint, NumberOfIndividov); //тестируем поколение на ошибку
		double* errorsCrossOver = new double[NumberOfIndividov]; //заводим массив ошибок

		for (size_t i = 0; i != NumberOfIndividov; ++i) errorsCrossOver[i] = errorsH[i]; //записываем ошибки в массив
		sort(errorsCrossOver, errorsCrossOver + NumberOfIndividov); //соритируем данное поколение 

		int merodianCrossOvering = NumberOfIndividov / 2; 
		double merodianErrorCrossOvering = errorsCrossOver[merodianCrossOvering]; //переписываем ошибки данного поколения 
		double* theBestInd = new double[powCount]; //лучшие индивиды в поколение

		for (size_t i = 0; i < NumberOfIndividov; i++)
		{
			if (merodianErrorCrossOvering < errorsH[i]) { //записываем новое поколение если только оно лучше старого
				for (size_t j = 0; j < powCount; j++) individumsH[i * powCount + j] = 0;
			}
			if (errorsH[i] == errorsCrossOver[0]) {
				for (int j = 0; j < powCount; j++) theBestInd[j] = individumsH[i * powCount + j]; //сохраняем  лучшие индивид
			}
		}
		printf("error = %f\n", errorsCrossOver[0]); //выводим каждый раз ошибку
		for (int i = 0; i < NumberOfIndividov * powCount; i++) //перебираем всех индивидов 
		{
			if (individumsH[i] == 0) {
				individumsH[i] = theBestInd[rand() % powCount]; //записываем в новое поколение лучшего и рандомного 
			}

			if (MathMutation > (rand() % 100 + 1)) {
				individumsH[i] += Random(-dispersionMutation, dispersionMutation);  //применяем мутацию
			}
		}
	}
	unsigned int end_time = clock(); // конечное время
	unsigned int search_time = end_time - start_time; // искомое время
	printf("search_time_cpu = %i\n", search_time); //выводим время обработки 
}


void gpu() {//обработка на gpu, в данной подпрограмме мы делаем все тоже самое что и cpu только с использованием библиотеки для работы с gpu
	double* pointsH = new double[NumberOfPoint];
	for (int i = 0; i < NumberOfPoint; i++) pointsH[i] = Random(0, 20);

	double* individumsH = new double[NumberOfIndividov * powCount];
	for (int i = 0; i < NumberOfIndividov * powCount; i++) individumsH[i] = Random(0, randMaxCount);

	double* errorsH = new double[NumberOfIndividov];
	for (int i = 0; i < NumberOfIndividov; i++) errorsH[i] = 1000;

	unsigned int start_time_gpu = clock(); // начальное время
	double* pointsD = NULL;
	double* individumsD = NULL;
	double* errorsD = NULL;


	for (int pokolenie = 0; pokolenie < KolOfPokoleni; pokolenie++)
	{
		int NumberOfIndividovBytes = NumberOfIndividov * powCount * sizeof(double);
		int NumberOfPointBytes = NumberOfPoint * sizeof(double);

		cudaMalloc((void**)&pointsD, NumberOfPointBytes);
		cudaMalloc((void**)&individumsD, NumberOfIndividovBytes * powCount);
		cudaMalloc((void**)&errorsD, NumberOfIndividov * sizeof(double));

		cudaMemcpy(pointsD, pointsH, NumberOfPointBytes, cudaMemcpyHostToDevice);
		cudaMemcpy(individumsD, individumsH, NumberOfIndividovBytes, cudaMemcpyHostToDevice);
		cudaMemcpy(errorsD, errorsH, NumberOfIndividovBytes, cudaMemcpyHostToDevice);

		Errors << <1, NumberOfIndividov >> > (pointsD, individumsD, errorsD, powCount, NumberOfPoint);

		cudaMemcpy(errorsH, errorsD, NumberOfIndividov * sizeof(double), cudaMemcpyDeviceToHost);
        double* errorsCrossOver = new double[NumberOfIndividov];

		for (size_t i = 0; i != NumberOfIndividov; ++i) errorsCrossOver[i] = errorsH[i];
		
		sort(errorsCrossOver, errorsCrossOver + NumberOfIndividov);
		printf("error = %f\n", errorsCrossOver[0]);
		
		int merodianCrossOvering = NumberOfIndividov / 2;
		double merodianErrorCrossOvering = errorsCrossOver[merodianCrossOvering];
		double* theBestInd = new double[powCount];

		for (size_t i = 0; i < NumberOfIndividov; i++)
		{
			if (merodianErrorCrossOvering < errorsH[i]) {
				for (size_t j = 0; j < powCount; j++)
				{
					individumsH[i * powCount + j] = 0;
				}
			}
			if (errorsH[i] == errorsCrossOver[0]) {
				for (int j = 0; j < powCount; j++)
				{
					theBestInd[j] = individumsH[i * powCount + j];
				}
			}
		}
		for (int i = 0; i < NumberOfIndividov * powCount; i++)
		{
			if (individumsH[i] == 0) {
				individumsH[i] = theBestInd[rand() % powCount];
			}
			if (MathMutation > (rand() % 100 + 1)) {
				individumsH[i] += Random(-dispersionMutation, dispersionMutation);
			}
		}
	}
	unsigned int all_time_gpu = clock(); // все время
	unsigned int find_time_gpu = all_time_gpu - start_time_gpu; // искомое время
    printf("find_time_gpu = %i\n", find_time_gpu); //выводим время

	cudaFree(pointsD); cudaFree(individumsD); cudaFree(errorsD); //завершаем куду

	delete pointsH;	delete individumsH; delete errorsH; //чистим память
}


int main()
{
	cpu(); //обработка на cpu
	gpu(); //обработка на gpu
	system("pause");
	return 0;
}
