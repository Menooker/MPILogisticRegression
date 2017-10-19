/*
libssp.cpp
Part of this file is copied from one of my (Menooker) project - Dogee - https://github.com/Menooker/Dogee
*/	


#include <string>
#include <unordered_map>
#include <cmath>
#include <functional>
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>
#include <mutex>
#include <set>
#include <vector>
#include <iterator>
#include <limits>

#include <string.h>
#include <mpi.h>

#include <list>

#define MPI_TAG_GRAD 1
#define MPI_TAG_LOSS 2
	
namespace libssp
{

	std::unordered_map<std::string, std::string> param;
	std::string& HelperGetParam(const std::string&  str)
	{
		return param[str];
	}

	int HelperGetParamInt(const std::string& str)
	{
		return atoi(HelperGetParam(str).c_str());
	}

	double HelperGetParamDouble(const std::string& str)
	{
		return atof(HelperGetParam(str).c_str());
	}


	void HelperInitCluster(int argc, char* argv[])
	{
		for (int i = 1; i + 1 < argc; i+=2)
		{
			param[argv[i]] = argv[i + 1];
		}
	}

	void ParseCSV(const char* path, std::function<bool(const char* cell, int line, int index)> func, int skip_to_line, bool use_cache)
	{
		char buf[2048];
		int cnt = 0;
		int line = 0;
		buf[0] = 0;
		char* st;
		uint32_t lagacy = 0;
		char* last;

		FILE* f = (FILE*)fopen(path,"r");
		if (use_cache)
		{
			std::stringstream pathbuf;
			pathbuf << path <<"."<< skip_to_line << ".cache";
			std::fstream outfile(pathbuf.str(), std::ios::in);
			if (!outfile)
			{
				std::cout << "No line cache file\n";
			}
			else
			{
				long fp;
				outfile >> fp;
				std::cout << "Read cache file " << pathbuf.str() << "\nFile pointer is " << fp << std::endl;
				fseek(f, fp, SEEK_SET);
				line = skip_to_line;
			}
		}
		for (;;)
		{
			st = buf + lagacy;
			fgets(st, 2048 - lagacy, f);//str.getline(st, 512 - lagacy,0);//fgets(st, 2048 - lagacy, f);
			char *p = st;
			last = buf;
			while (*p)
			{
				if (*p == '\n')
				{
					if (use_cache && line + 1 == skip_to_line)
					{
						std::stringstream pathbuf;
						pathbuf << path <<"."<< skip_to_line << ".cache";
						std::fstream outfile(pathbuf.str(), std::ios::out);
						long fp = ftell(f) - (strlen(p) - 1);
						outfile << fp;
						std::cout << "Output cache file " << pathbuf.str() <<"\nFile pointer is "<< fp<< std::endl;
					}
					*p = 0;
					if (line >= skip_to_line && !func(last, line, cnt))
						return;
					line++;
					cnt = 0;
					last = p + 1;
					//if (line % 100 == 0)
					//	printf("line %d\n", line);
				}
				else if (*p == ',')
				{
					*p = 0;
					if (line >= skip_to_line && !func(last, line, cnt))
						return;
					cnt++;
					last = p + 1;
				}
				p++;
			}
			if (p <= buf  )
			{
				break;
			}
			if (feof(f))
			{
				break;
			}
			lagacy = p  - last;
			memmove(buf, last, lagacy);
		}

		if (line >= skip_to_line)
			func(last, line, cnt);
		fclose(f);
	}


	static std::vector<int> clocks;
	static std::list<int> waitlist;
	static int findmin(std::vector<int>& v,int& idx)
	{
		int m= std::numeric_limits<int>::max();
		for(int i=0;i<v.size();i++)
		{
			if(v[i]<m)
			{
				m=v[i];
				idx=i;
			}
		}
		return m;
	}

	void InitParameterServer(int num_slaves)
	{
		clocks.resize(num_slaves,0);
	}

	
	void Clock(float* grad,int size, float loss,float* outparam)
	{
		MPI_Send(grad,size,MPI_FLOAT,0,MPI_TAG_GRAD,MPI_COMM_WORLD);
		MPI_Send(&loss,1,MPI_FLOAT,0,MPI_TAG_LOSS,MPI_COMM_WORLD);
		MPI_Status status;
		MPI_Recv(outparam,size,MPI_FLOAT,0,MPI_TAG_GRAD,MPI_COMM_WORLD,&status);
	}

	void ParameterServer(int num_slaves,int iterations,float* param,int vecsize,int ssp,std::function<void(int,float)> func)
	{
		InitParameterServer(num_slaves);
		int min_done=0;
		std::vector<float> losses(num_slaves,0);
		float* grad=new float[vecsize];
		while(min_done<iterations)
		{
			//get new gradient
			MPI_Status status;
			MPI_Recv(grad,vecsize,MPI_FLOAT,MPI_ANY_SOURCE,MPI_TAG_GRAD,MPI_COMM_WORLD,&status);
			int doneslave = status.MPI_SOURCE;
			int slaveidx= doneslave-1;


			//get and update loss table
			float loss;
			MPI_Recv(&loss,1,MPI_FLOAT,doneslave,MPI_TAG_LOSS,MPI_COMM_WORLD,&status);
			//printf("Recv loss %f\n",loss);
			losses[slaveidx]=loss;
			clocks[slaveidx]++;

			//update the parameter
			for(int i=0;i<vecsize;i++)
			{
				param[i]+=grad[i];
			}

			//find the slowest slave's clock
			int minidx;
			int newmin=findmin(clocks,minidx);
			//check the wait list, if one of the node
			//is ready, send the new parameter
			for(auto it=waitlist.begin();it!=waitlist.end();)
			{
				if(clocks[*it]-newmin<=ssp)
				{
					MPI_Send(param,vecsize,MPI_FLOAT,*it+1,MPI_TAG_GRAD,MPI_COMM_WORLD);
					it=waitlist.erase(it);
				}
				else
				{
					it++;
				}
			}
			//if the slave node is ready,resume it; Else, put it in the waitlist
			if(clocks[slaveidx]-newmin<=ssp)
			{
				MPI_Send(param,vecsize,MPI_FLOAT,slaveidx+1,MPI_TAG_GRAD,MPI_COMM_WORLD);
			}
			else
			{
				waitlist.push_back(slaveidx);
			}

			//if the minium is updated, notify the functor "func"
			if(newmin!=min_done)
			{
				float losssum=0;
				for(auto l: losses)
				{
					losssum+=l;
				}
				func(min_done,losssum);
				min_done=newmin;
			}

		}
		delete []grad;
	}

}