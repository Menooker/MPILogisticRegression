/*
Part of this file is copied from one of my (Menooker) project - Dogee - https://github.com/Menooker/Dogee
*/	
#include <mpi.h>
#include "libssp.h"
#include <cmath>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <chrono>
using namespace libssp;

using std::cin;
using std::cout;
using std::endl;
 
#include <vector>

float* local_param;
int ITER_NUM; //300
int THREAD_NUM; //2
float step_size; //0.005f
float TEST_PART; //0.2f
std::string PATH;

int g_param_len;
int g_num_points;
int num_nodes;
int rank;

class LocalDataset
{
public:
	float* dataset;
	float* labelset;
	float* testset;
	float* testlabel;
	int local_dataset_size;
	int local_train_size;
	int local_testset_size;
	int m_param_len;

	LocalDataset() :dataset(nullptr), labelset(nullptr),testset(nullptr), testlabel(nullptr), local_dataset_size(0)
	{}

	void GetThreadData(int tid, float* &odataset, float* &olabelset, float* &otestset, float* &otestlabel)
	{
		int toff = tid  * local_train_size / THREAD_NUM;
		int testoff = tid  * local_testset_size / THREAD_NUM;
		odataset = dataset + m_param_len * toff;
		olabelset = labelset + toff;
		otestset = dataset + (local_train_size + testoff)*m_param_len;
		otestlabel = labelset + local_train_size + testoff;
		return ;
	}

	void Load(int param_len, int num_points,int node_id)
	{
		const int local_line = num_points * (node_id-1) / (num_nodes-1);
		local_dataset_size =  num_points / (num_nodes-1);
		local_testset_size = local_dataset_size * TEST_PART;
		local_train_size = local_dataset_size - local_testset_size;
		labelset = new float[local_dataset_size];
		dataset = new float[param_len*local_dataset_size];
		testlabel = labelset + local_train_size;
		testset = dataset + param_len* local_train_size;
		m_param_len = param_len;
		printf("LL %d LS %d\n", local_line, local_dataset_size);
		int real_cnt = 0;
		int postive = 0;
		int datapoints = 0;
		ParseCSV(PATH.c_str(), [&](const char* cell, int line, int index){
			if (index > g_param_len)
			{
				std::cout << "CSV out of index, line="<< line<<" index="<<index<<"\n";
				return false;
			}
			if (line >= local_line + local_dataset_size)
				return false;
			if (line >= local_line )
			{
				if (index == g_param_len)
				{
					labelset[line - local_line] = atof(cell);
					datapoints++;
					if (cell[0] == '1')
						postive++;
				}
				else
					dataset[(line - local_line)*g_param_len + index] = atof(cell);
				real_cnt++;
			}

			return true;
		}, local_line);
		std::cout << "Loaded cells " << real_cnt << " num_data_points= " << real_cnt / param_len << " " << datapoints << " Positive= " << postive << std::endl;
	}
	void Free()
	{
		delete[]dataset;
		delete[]labelset;
	}
}local_dataset;




//get the sum of gradient vector over the subset of dataset
void slave_worker(float* thread_local_data, float* thread_local_label, int thread_point_num, 
	float* thread_test_data, float* thread_test_label, int thread_test_num,
	float* local_grad, float* local_loss)
{

		float thread_loss = 0;
		float* curdata = thread_local_data;
		memset(local_grad, 0, sizeof(float)*g_param_len);
		for (int i = 0; i < thread_point_num; i++)
		{
			double dot = 0;
			for (int j = 0; j < g_param_len; j++)
				dot += local_param[j] * curdata[j];
			double h = 1 / (1 + exp(-dot));
			double delta = thread_local_label[i] - h;
			thread_loss += delta*delta;
			for (int j = 0; j < g_param_len; j++)
				local_grad[j] += step_size * delta * curdata[j];
			curdata += g_param_len;
			//if (i % 50 == 0)
			//	printf("i=%d curdata=%p h=%f dot=%f delta=%f\n", i, curdata, h, dot, delta);
		}
		//printf("TGrad %p %f %f\n", local_grad, local_grad[20000], curdata[20000]);
		*local_loss = thread_loss;
		//slave_main will accumulate the local_grad and fetch the new parameters
	
}

void slave_main(uint32_t tid)
{
	//local gradients for all threads
	float** local_grad_arr = new float*[THREAD_NUM];
	//the parameter vector shared by all threads
	local_param = new float[g_param_len];
	memset(local_param, 0, sizeof(float)*g_param_len);
	//loss of all threads
	float* loss = new float[THREAD_NUM];
	local_dataset.Load(g_param_len, g_num_points, rank);
	
	int thread_point_num = local_dataset.local_train_size / THREAD_NUM;
	int thread_test_num = local_dataset.local_testset_size / THREAD_NUM;
	
	//get the initial parameters
	MPI_Bcast(local_param,g_param_len,MPI_FLOAT,0,MPI_COMM_WORLD);

	for (int i = 0; i < THREAD_NUM; i++)
	{
		local_grad_arr[i]=new float[g_param_len];
	}
	for (int itr = 0; itr < ITER_NUM; itr++)
	{
		//get gradients in parallel
		#pragma omp parallel for num_threads(THREAD_NUM)
		for (int i = 0; i < THREAD_NUM; i++)
		{
			//for each thread, calculate on part of the data, and output the gradient over each part of the data
			float* thread_local_data;
			float* thread_local_label;
			float* thread_test_data;
			float* thread_test_label;
			local_dataset.GetThreadData(i, thread_local_data, thread_local_label, thread_test_data, thread_test_label);
			slave_worker(thread_local_data, thread_local_label, thread_point_num,
				thread_test_data, thread_test_label, thread_test_num,
				local_grad_arr[i], loss + i);
		}

		//sum up the threads' loss and the gradient
		for (int i = 1; i < THREAD_NUM; i++)
		{
			loss[0] += loss[i];
			loss[i] = 0;
			for (int j = 0; j < g_param_len; j++)
				local_grad_arr[0][j] += local_grad_arr[i][j];
		}
		float dummy;

		//send the loss and gradient to Parameter Server
		Clock(&local_grad_arr[0][0],g_param_len,loss[0],local_param);
		//MPI_Reduce(&loss[0] , &dummy ,1, MPI_FLOAT , MPI_SUM ,0, MPI_COMM_WORLD );
		//std::cout << "grad=" << local_grad_arr[0][0] <<" loss=" <<loss[0] << std::endl;
		loss[0] = 0;
		//MPI_Allreduce(&local_grad_arr[0][0] , local_param , g_param_len, MPI_FLOAT , MPI_SUM , MPI_COMM_WORLD );
	}
	

	local_dataset.Free();
	for (int i = 0; i < THREAD_NUM; i++)
		delete []local_grad_arr[i];
	delete[]local_grad_arr;
	delete[]local_param;
}

void master()
{
	//fill in the initial parameter vector with random values
	float* g_param= new float[g_param_len];
	for(int i=0;i<g_param_len;i++)
		g_param[i]=(float) (rand()%100) / 100000; 
	float dummy=0,outloss;
	
	//broadcast the initial parameter vector
	MPI_Bcast(g_param,g_param_len,MPI_FLOAT,0,MPI_COMM_WORLD);

	auto t = std::chrono::system_clock::now();

	auto t2 = std::chrono::system_clock::now();
	//call the parameter server entry function
	ParameterServer(num_nodes-1,ITER_NUM,g_param,g_param_len,4,[&](int itr,float loss)
		{
			std::cout << "Iter"<<itr<<" took "<<
			std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - t2).count()
			<<" milliseconds, loss = " <<loss<<std::endl;
			t2 = std::chrono::system_clock::now();
		});

	std::cout << "Total time" << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - t).count()
		<<std::endl;
	delete[] g_param;
}

int main(int argc, char *argv[]){
	MPI_Init(&argc, &argv);
 	HelperInitCluster(argc, argv);
 	MPI_Comm_size( MPI_COMM_WORLD, &num_nodes );

 	//initialize the parameter with program arguments
	g_param_len = HelperGetParamInt("num_param");
	g_num_points = HelperGetParamInt("num_points");
	PATH = HelperGetParam("path");
	
	ITER_NUM = HelperGetParamInt("iter_num"); 
	THREAD_NUM = HelperGetParamInt("thread_num");
	step_size = HelperGetParamDouble( "step_size"); 
	TEST_PART = HelperGetParamDouble("test_partition"); 

	std::cout << "Parameters:\n" << "num_param :" << g_param_len << "\nnum_points : " << g_num_points
		<< "\niter_num : " << ITER_NUM << "\nthread_num : " << THREAD_NUM
		<< "\nstep_size : " << step_size << "\ntest_partition : " << TEST_PART << std::endl;

	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if(rank!=0)
	{
		slave_main(rank-1);
	}
	else
	{
		master();
	}
	MPI_Finalize(); 
	return 0;
}
