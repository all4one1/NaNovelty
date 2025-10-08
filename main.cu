#include "cusolver.h"
#include "auto_test.h"


int main()
{
    cusolver::GPU_ gpu(0);
    
    auto_test();

    gpu.show_memory_usage_MB();
    return 0;
}
