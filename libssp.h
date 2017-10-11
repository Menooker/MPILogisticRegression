#include <string>
#include <functional>

namespace libssp
{
	extern std::string& HelperGetParam(const std::string&  str);

	extern int HelperGetParamInt(const std::string& str);

	extern double HelperGetParamDouble(const std::string& str);

	extern void HelperInitCluster(int argc, char* argv[]);
	extern void ParseCSV(const char* path, std::function<bool(const char* cell, int line, int index)> func, int skip_to_line=0, bool use_cache=true);
	extern void Clock(float* grad,int size, float loss,float* outparam);
	extern void ParameterServer(int num_slaves,int iterations,float* param,int vecsize,int ssp,std::function<void(int,float)> func);
}