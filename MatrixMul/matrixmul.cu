#include <chrono>
#include <stdio.h>
#include <iostream>
#include <locale>
#include <string>
#define ET_MIN 2            //one min dimension matrix exponent of two
#define ET_MAX 11   //11        //one min dimension matrix exponent of two
#define BLOCK_SIZE 16
#define AVG_ITER 1 			//Number of iterations to average the result with the same input values

void plots(double **pltCPU,double **pltGPU,int ETlen){

    FILE *gp = popen("gnuplot -persist -geometry 800x800" ,"w");

    fprintf(gp,"set terminal x11\n");
    if (gp == NULL){
        printf("Error opening pipe to GNU plot.\n");
        exit(0);
         }

    fprintf(gp,"set  multiplot layout 2,1\n");
    fprintf(gp,"set  style line 10 linetype 5 \n");
    fprintf(gp,"set grid\n");
    fprintf(gp,"set yrange [-20:150]\n");
    fprintf(gp,"set xrange[-100:1800]\n");

    fprintf(gp,"set title \"Speed CPU vs GPU\"\n");
    fprintf(gp,"set ylabel \"time, ms\"\n");
    fprintf(gp,"set xlabel \"matrix dimension\"\n");
    fprintf(gp, "plot '-' using 1:2 with lines title \"i3 4170\",");
    fprintf(gp,       "'' using 1:2 with lines title \"GTX 960\"\n");


    for(int i = 0; i < ETlen; i=i+1)
        fprintf(gp, "%f %f\n",pltCPU[i][0],pltCPU[i][1]);


    fprintf(gp, "%s\n", "e");

    for(int i = 0; i < ETlen; i=i+1)
        fprintf(gp, "%f %f\n",pltGPU[i][0],pltGPU[i][1]);
    fprintf(gp, "%s\n", "e");




    fprintf(gp,"set yrange[*:1000]\n");
    fprintf(gp,"set xrange[1:*]\n");

    fprintf(gp,"set title \"Speed CPU vs GPU (Logarithmic)\"\n");
    fprintf(gp,"set log x\n");
    fprintf(gp,"set log y\n");
    fprintf(gp, "plot '-' using 1:2 with lines title \"i3 4170\",");
    fprintf(gp,       "'' using 1:2 with lines title \"GTX 960\"\n");

    for(int i = 0; i < ETlen; i=i+1)
        fprintf(gp, "%f %f\n",pltCPU[i][0],pltCPU[i][1]);
    fprintf(gp, "%s\n", "e");

    for(int i = 0; i < ETlen; i=i+1)
        fprintf(gp, "%f %f\n",pltGPU[i][0],pltGPU[i][1]);
    fprintf(gp, "%s\n", "e");
/*
    fprintf(gp,"unset log y\n");
    fprintf(gp,"set grid\n");
    fprintf(gp,"set title \"GPU speed as a percentage of CPU speed \"\n");
    fprintf(gp,"set ylabel \"percents, % \"\n");
    fprintf(gp,"set xlabel \"vector lenght\"\n");
*/
    /*
    fprintf(gp, "plot '-' using 1:2 with lines title \"GPU speed\"\n");
    for(int i = 0; i <= ETlen; i=i+1)
        fprintf(gp, "%f %f\n",pltGPU[i][0],((1/(pltGPU[i][1]/pltCPU[i][1])))*100);
    fprintf(gp, "%s\n", "e");
    */
    fprintf(gp,"unset multiplot\n");

    fflush(gp);
    pclose(gp);
}


void random_floats(float *commonArray, unsigned int CntArr){
    for (unsigned int i = 0; i <= CntArr*CntArr; i++)
    commonArray[i] = static_cast <float>(static_cast <float> (rand()) / (static_cast <float> (RAND_MAX)));
}



__global__ void matrixMul(float* A, float* B, float* C, unsigned int n) {

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    C[n * row + col] = 0;

    if (row < n && col < n) {
        for (unsigned k = 0; k < n; k++) {
            C[n * row + col] += A[row * n + k] * B[k * n + col];
        }
    }
}

void matrixMulGPU(float* A, float* B, float* C, unsigned int n) {

    dim3 threadsPerBlock(n, n);
    dim3 blocksPerGrid(1, 1);
    if (n >= BLOCK_SIZE) {
        threadsPerBlock.x = BLOCK_SIZE;
        threadsPerBlock.y = BLOCK_SIZE;
        blocksPerGrid.x = ceil((n + threadsPerBlock.x - 1) / threadsPerBlock.x);
        blocksPerGrid.y = ceil((n + threadsPerBlock.y - 1) / threadsPerBlock.y);
    }
    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, n);
}

void matrixMulCPU(float* A, float* B, float* C, unsigned  int n) {


    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            double sum = 0;
            for (unsigned int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }

}





int main()
{
    int ETlen=ET_MAX-ET_MIN+1;
    double* pltGPU[ETlen];
    double* pltCPU[ETlen];
    for (int i = 0; i<ETlen; i++){
	    pltGPU[i]=(double*)malloc(2*sizeof(double));
	    pltCPU[i]=(double*)malloc(2*sizeof(double));
	}


	float avgCPU = 0.0f;
	float avgGPU = 0.0f;
	for(int k=ET_MIN;k<=ET_MAX;k++){
		int n=pow(2,k);
		for(int i=0;i<AVG_ITER;i++){

		    unsigned long arrSize = n * n * sizeof(float);

		    float *h_A = (float*)malloc(arrSize);
		    float *h_B = (float*)malloc(arrSize);
		    float *h_C_CPU = (float*)malloc(arrSize);
		    float *h_C_GPU = (float*)malloc(arrSize);

		    random_floats(h_A, n);
		    random_floats(h_B, n);

		    float *d_A, *d_B, *d_C;

		    cudaMalloc(&d_A, arrSize);
		    cudaMalloc(&d_B, arrSize);
		    cudaMalloc(&d_C, arrSize);
		    auto startGPU = std::chrono::high_resolution_clock::now();

		    cudaMemcpy(d_A, h_A, arrSize, cudaMemcpyHostToDevice);
		    cudaMemcpy(d_B, h_B, arrSize, cudaMemcpyHostToDevice);

		    matrixMulGPU(d_A, d_B, d_C, n);

		    cudaMemcpy(h_C_GPU, d_C, arrSize, cudaMemcpyDeviceToHost);
		    auto endGPU = std::chrono::high_resolution_clock::now();

		    auto startCPU = std::chrono::high_resolution_clock::now();
		    matrixMulCPU(h_A, h_B, h_C_CPU, n);
		    auto endCPU = std::chrono::high_resolution_clock::now();

		    std::chrono::duration<double> elapsed_secondsCPU = endCPU-startCPU;
		    std::chrono::duration<double> elapsed_secondsGPU = endGPU-startGPU;
		    printf("Count on GPU (s): %f\n", elapsed_secondsGPU.count());
			printf("Count on CPU (s): %f\n", elapsed_secondsCPU.count());

		    printf("CPU[0,0] = %f\nGPU[0,0] = %f\n",h_C_CPU[0],h_C_GPU[0]);


            avgCPU += elapsed_secondsCPU.count();
            avgGPU += elapsed_secondsGPU.count();
		    cudaFreeHost(h_A);
		    cudaFreeHost(h_B);
		    cudaFreeHost(h_C_GPU);
		    cudaFreeHost(h_C_CPU);

		    cudaFree(d_A);
		    cudaFree(d_B);
		    cudaFree(d_C);
	}
    pltGPU[k-ET_MIN][0]=n;
    pltGPU[k-ET_MIN][1]=avgGPU*1000/AVG_ITER;
    pltCPU[k-ET_MIN][0]=n;
    pltCPU[k-ET_MIN][1]=avgCPU*1000/AVG_ITER;


	avgCPU = 0.0f;
	avgGPU = 0.0f;
	}
    for(int i=0;i<ETlen;i++){
    printf("pltGPU[%d]= %f %f\n",i,pltGPU[i][0],pltGPU[i][1]-pltCPU[i][1]);
  //  printf("pltCPU[%d]= %f %f\n",i,pltCPU[i][0],pltCPU[i][1]);

    }
plots(pltCPU,pltGPU,ETlen);
    return 0;
}
