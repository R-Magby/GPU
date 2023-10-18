#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include <time.h> 

__global__ void suma_matriz(int *A,int *B,int *C,int Nx){
	int idx=threadIdx.x + blockDim.x*blockIdx.x;
	int idy=threadIdx.y + blockDim.y*blockIdx.y;
	

	C[idy*Nx+idx] = A[idy*Nx+idx]+B[idy*Nx+idx];

}

void rellenar(int *A,int Nx,int Ny){
	for(int i=0;i<Nx;i++){
		for(int j=0;j<Ny;j++){
			A[j*Nx+i]=j*Nx+i;
		}
	}
}

main(int argc,char *argv[]){

	int *A,*B,*C,*cuA,*cuB,*cuC;
        int *D,*E,*F,*cuD,*cuE,*cuF;


	int Nx=atoi(argv[1]);
        int Ny=atoi(argv[2]);

	int size=Nx*Ny*sizeof(int);

	A=(int*)malloc(size);
	B=(int*)malloc(size);
	C=(int*)malloc(size);
	rellenar(A,Nx,Ny);
	rellenar(B,Nx,Ny);
	

        cudaMalloc((void **)&cuA, size);
        cudaMalloc((void **)&cuB, size);
        cudaMalloc((void **)&cuC, size);


        D=(int*)malloc(size);
        E=(int*)malloc(size);
        F=(int*)malloc(size);
        rellenar(D,Nx,Ny);
        rellenar(E,Nx,Ny);


        cudaMalloc((void **)&cuD, size);
        cudaMalloc((void **)&cuE, size);
        cudaMalloc((void **)&cuF, size);


	cudaMemcpy(cuA,A,size,cudaMemcpyHostToDevice);
        cudaMemcpy(cuB,B,size,cudaMemcpyHostToDevice);

        cudaMemcpy(cuD,D,size,cudaMemcpyHostToDevice);
        cudaMemcpy(cuE,E,size,cudaMemcpyHostToDevice);


	dim3 grid(1,1);
	dim3 bloque(32,32);
	cudaStream_t stream1, stream2 ;
	
	cudaStreamCreate ( &stream1) ;
	cudaStreamCreate ( &stream2) ;

//	cudaMemcpyAsync ( cuA, A, size, cudaMemcpyHostToDevice, stream1 ) ;


	suma_matriz<<<grid,bloque,0,stream1>>>(cuA,cuB,cuC,Nx);
	suma_matriz<<<grid,bloque,0,stream2>>>(cuD,cuE,cuF,Nx);


    cudaError_t err = cudaGetLastError();
    printf("Error: %s\n",cudaGetErrorString(err));

    // Copiar resultado al host
    cudaMemcpy(C, cuC, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(F, cuF, size, cudaMemcpyDeviceToHost);

	for(int i=0;i<Nx;i++){
                for(int j=0;j<Ny;j++){
                        printf("%d |",C[j*Nx+i]);
                        printf("%d \n",F[j*Nx+i]);

                }
        }

	free(A); free(B); free(C);
    cudaFree(cuA); cudaFree(cuB); cudaFree(cuC);

	return 0;
}


