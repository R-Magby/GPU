#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include <time.h> 
#include <string.h>
#define N 1024


__global__ void suma_matriz_global(float *A,float *B,float *C,int Nx, int Ny){

	int idx=threadIdx.x + blockDim.x*blockIdx.x;
	int idy=threadIdx.y + blockDim.y*blockIdx.y;
    C[idy*Nx+idx] = A[idy*Nx+idx] + B[idy*Nx+idx];
    
}


void rellenar(float *V,float num,int n){
    for (int i = 0; i < n; i++) {
        V[i] = num;
    }
}

void verificar(float *A,float num){
    bool check = true;
    int i=0;
    while(check==1 && i<N*N){
        if (A[i]==num){
            i++;
        }
        else {
            check=false;
        }
    }
    if (check==true){
        printf("La matriz C cumple la condicion.");
    }
    else {
        printf("La matriz C NO cumple la condicion");
    }
}


int main(){

    float *A,*B,*C;
    float *cuda_A,*cuda_B,*cuda_C;
    float *host_A,*host_B,*host_C;

    int nstream=4;

    int size = N*N*sizeof(float);
    double time_spent = 0.0;
    clock_t begin = clock();

    dim3 bloque(32,32);
    dim3 grid(ceil(N/(32*2)),ceil(N/(32*2)));

    A=(float *)malloc(size);B=(float *)malloc(size);C=(float *)malloc(size);


    cudaMallocHost((void **)&host_A, size); 
    cudaMallocHost((void **)&host_B, size); 
    cudaMallocHost((void **)&host_C, size); 

    rellenar(host_A,1.0,N*N);rellenar(host_B,2.0,N*N);

    cudaMalloc((void **)&cuda_A, size);
    cudaMalloc((void **)&cuda_B, size);
    cudaMalloc((void **)&cuda_C, size);


        cudaStream_t *stream;
        stream = (cudaStream_t*) new cudaStream_t[nstream];
        for (int i = 0; i < nstream; i++)
            cudaStreamCreate(&stream[i]);

        cudaDeviceSynchronize ();


        for (int i=0;i<nstream;i++){
            int i_byte_stream=i*N*N/nstream;

            cudaMemcpyAsync(&cuda_A[i_byte_stream], &host_A[i_byte_stream], size/nstream,cudaMemcpyHostToDevice, stream[i]);
            cudaMemcpyAsync(&cuda_B[i_byte_stream], &host_B[i_byte_stream], size/nstream,cudaMemcpyHostToDevice, stream[i]);

            suma_matriz_global<<<grid,bloque,0,stream[i]>>>(&cuda_A[i_byte_stream],&cuda_B[i_byte_stream], &cuda_C[i_byte_stream],N,N);

            cudaMemcpyAsync(&host_C[i_byte_stream], &cuda_C[i_byte_stream], size/nstream,cudaMemcpyDeviceToHost, stream[i]);

        }
        for (int i = 0; i < nstream; i++) {
            cudaStreamSynchronize(stream[i]);
           }        
        for(int idx=0;idx<N;idx++){
            for(int idy=0;idy<1;idy++){
                printf("%d | %f\n",idy*N+idx,host_C[idy*N+idx]);
            }
        }
        verificar(host_C,3.0);
        clock_t end = clock();
        
        time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
        printf("\nTiempo de ejecucion : %f seconds\n", time_spent);
        for (int i = 0; i < nstream; i++)
            cudaStreamDestroy(stream[i]);
    


    cudaError_t err = cudaGetLastError();
    printf("Error: %s\n",cudaGetErrorString(err));
    
    free(A); free(B); free(C);
    cudaFree(cuda_A); cudaFree(cuda_B); cudaFree(cuda_C);
}