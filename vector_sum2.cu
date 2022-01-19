#include <stdio.h>
#include <iomanip>
#include <numeric>
#include <string>
#include <chrono>
#include <cmath>
#include <random>
#define CLC 50                // Cycles per one number for averaging results
#define TPB 256     	      // THREADS_PER_BLOCK
#define ETmin 10     	      //vector lenght min exponent of two
#define ETmax 20 	          //vector lenght max exponent of two



void random_floats(float *commonArray, unsigned long CntArr){
	for (unsigned int i = 0; i <= CntArr; i++){
	commonArray[i] = static_cast <float>(static_cast <float> (rand()) / (static_cast <float> (RAND_MAX)));
 }

}



__global__ void vcAdd(float *g_idata, float *g_odata) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    sdata[tid] = g_idata[i]+g_idata[i+blockDim.x];
    __syncthreads();

        for(unsigned int s=blockDim.x/2;s>0; s>>=1) {
            if (tid<s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
        }
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


int main(void){

    int ETlen=ETmax-ETmin;
	int plotSize = (ETlen) * sizeof(double);
	double*	plotGPU = (double*)malloc(plotSize);
	double*	plotCPU = (double*)malloc(plotSize);

	double* pltGPU[ETmax];
	for (int i = 0; i<ETmax; i++)
		pltGPU[i]=(double*)malloc(2*sizeof(double));

    double* pltCPU[ETmax];
    for (int i = 0; i<ETmax; i++)
        pltCPU[i]=(double*)malloc(2*sizeof(double));





int F =  ETmin;

while (F<=ETmax){

    unsigned int pows[23];
	for(unsigned int i=0;i<23;i++)
	    pows[i]=pow(2,i);

	float *aGPU;
	float *oGPU;

	double srednee = 0.0;
	unsigned int cycle = CLC;
    float avgCPU  = 0.0f;
    float avgGPU  = 0.0f; 
    float timeCPU = 0.0f;
    float timeGPU = 0.0f;

	while (cycle>0){

		unsigned int arrSize = (pow(2,F)) * sizeof(float);
		unsigned int arrLenght = (pow(2,F));
//		unsigned int arrLenghtM = (pow(2,F-1));
//		unsigned int arrLenghtTwo = (pow(2,F-2));
        float* commonArray = (float*)malloc(arrSize);
		float* outPut = (float*)malloc(arrSize);
		random_floats(commonArray,arrLenght);
        //---------CPU add
		auto start = std::chrono::high_resolution_clock::now();
		float k = 0.0f;
		float out = 0.0f;
		for (unsigned int i=0;i<arrLenght;i++)
    	k+=commonArray[i];
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed_seconds = end-start;
        //---------------

		cudaMalloc(&aGPU,arrSize);
		cudaMalloc(&oGPU,arrSize);
		cudaMemcpy(aGPU, commonArray, arrSize, cudaMemcpyHostToDevice);

		auto start1 = std::chrono::high_resolution_clock::now();
		unsigned int kernelIter = F;

		while (kernelIter>=10){
    		vcAdd<<<(pows[kernelIter] + TPB - 1)/TPB/2,TPB,TPB*sizeof(float)>>>(aGPU,oGPU);
	    	kernelIter=kernelIter-1;
	    	}
		cudaMemcpy(commonArray, oGPU, arrSize, cudaMemcpyDeviceToHost);

		for(unsigned int i = 0; i < pows[F-8]; i++)
        	out +=commonArray[i];

		auto end1 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed_seconds1 = end1-start1;

//		printf("CPU SUM = %f\n",k);
//		printf("GPU SUM = %f\n", out);
//		printf("*****Speedup = %f\n",elapsed_seconds.count()/elapsed_seconds1.count());

		srednee+=elapsed_seconds.count()/elapsed_seconds1.count();
		cycle--;
        avgCPU+=k;
        avgGPU+=out;
        timeCPU+=elapsed_seconds.count();
        timeGPU+=elapsed_seconds1.count();
		free(commonArray);free(outPut);
		cudaFree(aGPU);cudaFree(oGPU);
		}
    cycle=CLC;
	printf("n = %d %d sredniy speedup = %f, sum difference percents = %f \n",F,pows[F],srednee/cycle,abs((((avgCPU-avgGPU)/avgCPU)/CLC)*100));


	pltGPU[F-10][0]=pow(2,F);
	pltGPU[F-10][1]=timeGPU/CLC;
	pltCPU[F-10][0]=pow(2,F);
	pltCPU[F-10][1]=timeCPU/CLC;


    F=F+1;
    cudaDeviceReset();
    }

FILE *gp = popen("gnuplot -persist -geometry 800x800" ,"w");

fprintf(gp,"set terminal x11\n");
if (gp == NULL){
    printf("Error opening pipe to GNU plot.\n");
    exit(0);
     }
fprintf(gp,"set  multiplot layout 3,1\n");
fprintf(gp,"set title \"Speed CPU vs GPU\"\n");
fprintf(gp,"set xlabel \"time, ms\"\n");
fprintf(gp,"set ylabel \"vector lenght\"\n");
fprintf(gp, "plot '-' using 1:2 with lines title \"CPU\",");
fprintf(gp,       "'' using 1:2 with lines title \"GPU\"\n");

for(int i = 0; i <= ETlen; i=i+1)
    fprintf(gp, "%f %f\n",pltCPU[i][1],pltCPU[i][0]);
fprintf(gp, "%s\n", "e");

for(int i = 0; i <= ETlen; i=i+1)
    fprintf(gp, "%f %f\n",pltGPU[i][1],pltGPU[i][0]);
fprintf(gp, "%s\n", "e");

fprintf(gp,"set title \"Speed CPU vs GPU (Logarithmic)\"\n");
fprintf(gp,"set log x\n");
fprintf(gp,"set log y\n");
fprintf(gp, "plot '-' using 1:2 with lines title \"CPU\",");
fprintf(gp,       "'' using 1:2 with lines title \"GPU\"\n");

for(int i = 0; i < ETlen; i=i+1)
    fprintf(gp, "%f %f\n",pltCPU[i][1],pltCPU[i][0]);
fprintf(gp, "%s\n", "e");

for(int i = 0; i < ETlen; i=i+1)
    fprintf(gp, "%f %f\n",pltGPU[i][1],pltGPU[i][0]);
fprintf(gp, "%s\n", "e");

fprintf(gp,"unset log y\n");
fprintf(gp,"set grid\n");
fprintf(gp,"set title \"GPU speed as a percentage of CPU speed \"\n");
fprintf(gp,"set ylabel \"percents, % \"\n");
fprintf(gp,"set xlabel \"vector lenght\"\n");


fprintf(gp, "plot '-' using 1:2 with lines title \"GPU speed\"\n");
for(int i = 0; i <= ETlen; i=i+1)
    fprintf(gp, "%f %f\n",pltGPU[i][0],((1/(pltGPU[i][1]/pltCPU[i][1])))*100);
fprintf(gp, "%s\n", "e");

fprintf(gp,"unset multiplot\n");

fflush(gp);
pclose(gp);



 return 0;

}
