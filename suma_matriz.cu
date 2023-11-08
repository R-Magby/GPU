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
__global__ void suma_matriz_shared(float *A,float *B,float *C,int Nx, int Ny){
    extern __shared__ float array[];
    float *A_shared=(&array[0]);
    float *B_shared=(&array[Nx]);

	int idx=threadIdx.x + blockDim.x*blockIdx.x;
	int idy=threadIdx.y + blockDim.y*blockIdx.y;
    int x=threadIdx.x;
    int y=threadIdx.y;

    A_shared[y*blockDim.x+x]=A[idy*Nx+idx];
    B_shared[y*blockDim.x+x]=B[idy*Nx+idx];

    __syncthreads();
    C[idy*Nx+idx] = A_shared[y*blockDim.x+x] + B_shared[y*blockDim.x+x];
    
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


int main(int argc, char *argv[]){

    float *A,*B,*C;
    float *cuda_A,*cuda_B,*cuda_C;
    char *mode=argv[1];
    int size = N*N*sizeof(float);

    double time_spent = 0.0;
    clock_t begin = clock();

    dim3 bloque(32,32);
    dim3 grid(ceil(N/32),ceil(N/32));

    A=(float *)malloc(size);B=(float *)malloc(size);C=(float *)malloc(size);

    rellenar(A,1.0,N*N);rellenar(B,2.0,N*N);

    cudaMalloc((void **)&cuda_A, size);
    cudaMalloc((void **)&cuda_B, size);
    cudaMalloc((void **)&cuda_C, size);




    if (strcmp(mode,"sequential")==0){
        for(int idx=0;idx<N;idx++){
            for(int idy=0;idy<N;idy++){
                C[idy*N+idx] = A[idy*N+idx] + B[idy*N+idx];
            }
        }
        printf("En secuencial.\n");
        verificar(C,3.0);
        clock_t end = clock();

        time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
        printf("\nTiempo de ejecucion : %f seconds\n", time_spent);
    }
    else if (strcmp(mode,"GPU")==0){
        cudaMemcpy(cuda_A, A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_B, B, size, cudaMemcpyHostToDevice);
        suma_matriz_global<<<grid,bloque>>>(cuda_A,cuda_B,cuda_C,N,N);
        cudaMemcpy(C,cuda_C, size, cudaMemcpyDeviceToHost);
        printf("En GPU.\n");

        verificar(C,3.0);
        clock_t end = clock();
        
        time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
        printf("\nTiempo de ejecucion : %f seconds\n", time_spent);

    }
    else if (strcmp(mode,"GPU_shared")==0){

        suma_matriz_shared<<<grid,bloque,2*N*sizeof(float)>>>(cuda_A,cuda_B,cuda_C,N,N);
        cudaMemcpy(C,cuda_C, size, cudaMemcpyDeviceToHost);
        printf("En GPU_shared.\n");

        verificar(C,3.0);
        clock_t end = clock();
        
        time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
        printf("\nTiempo de ejecucion : %f seconds\n", time_spent);

    }
    else if (strcmp(mode,"GPU_stream")==0){
        int nstream=4;
        cudaStream_t *stream;
        stream = (cudaStream_t*) new cudaStream_t[nstream];
        for (int i = 0; i < nstream; i++)
            cudaStreamCreate(&stream[i]);
        for (int i=0;i<4;i++){
            int istream=i* size/nstream;
            cudaMemcpyAsync(&cuda_B[istream], &A[istream], size/nstream,cudaMemcpyHostToDevice, stream[i]);
            cudaMemcpyAsync(&cuda_A[istream], &B[istream], size/nstream,cudaMemcpyHostToDevice, stream[i]);
            suma_matriz_global<<<grid,bloque,0,stream[i]>>>(cuda_A,cuda_B,cuda_C,N,N);
            cudaMemcpyAsync(&C[istream], &cuda_C[istream], size/nstream,cudaMemcpyDeviceToHost, stream[i]);

        }
        cudaDeviceSynchronize();

        verificar(C,3.0);
        clock_t end = clock();
        
        time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
        printf("\nTiempo de ejecucion : %f seconds\n", time_spent);
        for (int i = 0; i < nstream; i++)
            cudaStreamDestroy(stream[i]);
    }
    else{
        printf("Por favor escriba sequential, GPU, GPU_shared o GPU_stream ");
    }

       cudaError_t err = cudaGetLastError();
    printf("Error: %s\n",cudaGetErrorString(err));
    free(A); free(B); free(C);
    cudaFree(cuda_A); cudaFree(cuda_B); cudaFree(cuda_C);
}