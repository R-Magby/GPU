#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include <time.h> 


__global__ void onda2d(float *u, float *u_m1, float *u_p1,const float dt,const float dx,const float c,const int T) {
    int x = threadIdx.x ;
    int y = threadIdx.y ;
    int Nx = blockDim.x ;
    int Ny = blockDim.y ;
    __shared__ float temp[32*32];

    if (x>0 && x<Nx-1 && y>0 && y<Ny-1){
        temp[y*Ny+x]=u[y*Ny+x];

        for (int t=0;t<T;t++){
            
            u_p1[y*Ny+x]=dt*dt/(dx*dx)*c*c*(temp[y*Ny+x+1]+temp[y*Ny+x-1]+
                temp[(y+1)*Ny+x]+temp[(y-1)*Ny+x]-4.0*temp[y*Ny+x]) + 2.0*temp[y*Ny+x]-u_m1[y*Ny+x];
            __syncthreads();

            u_m1[y*Ny+x]=temp[y*Ny+x];
        
            temp[y*Ny+x]=u_p1[y*Ny+x];
            u[t*Nx*Ny+y*Ny+x]=temp[y*Ny+x];
            __syncthreads();

        }
    }
}


// llenar el array con los indices
void gauss(float *data,float t,float c,float dx,float dy,float y_0,float x_0,int Nx, int Ny) {
    float r;
    for (int y=1;y<Ny-1;y++){
        for (int x=1;x<Nx-1;x++){
            r=sqrt((x*dx-x_0)*(x*dx-x_0)+(y*dy-y_0)*(y*dy-y_0));
		    data[y*Ny+x] = 2*exp(-(r-c*t)*(r-c*t)/4);
        }
    }
}   
void guardar_salida(float *data,int Nx, int Ny,int T) {
    FILE *fp = fopen("onda_2d_cuda.dat", "w");

    for (int t=0;t<T;t++){
        for (int y=0;y<Ny;y++){
            for (int x=0;x<Nx;x++){
                fprintf(fp, "%g\n", data[t*Nx*Ny+y*Ny+x]);
            }
        }
    }
}
int main(int argc, char *argv[]){
	float  *u, *u_m1, *u_p1,*d_u, *d_u_m1, *d_u_p1;

    int Nx=atoi(argv[1]);
    int Ny=atoi(argv[2]);
    int T=atoi(argv[3]);
    float c=atoi(argv[4]);

	int size = Nx*Ny*sizeof(float);

    float dx=1.0;
    float dy=1.0;
    float dt=1.0/T;

    double time_spent = 0.0;
    clock_t begin = clock();

	u = (float *)malloc(size*T); gauss(u,dt,c,dx,dy, Nx/2,Ny/2,Nx,Ny);
	u_m1 = (float *)malloc(size); gauss(u_m1,0,c,dx,dy, Nx/2, Ny/2,Nx,Ny);
	u_p1 = (float *)malloc(size);

    


    // Asignar memoria al lado del device (GPU)
    cudaMalloc((void **)&d_u, size*T);
    cudaMalloc((void **)&d_u_m1, size);
    cudaMalloc((void **)&d_u_p1, size);

    // Copiar al device
    cudaMemcpy(d_u, u, size*T, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u_m1, u_m1, size, cudaMemcpyHostToDevice);

    // Invocar kernel con N bloques de 1 thread cada uno
    dim3 bloque (Nx,Ny);
    onda2d<<<1,bloque>>>(d_u,d_u_m1,d_u_p1,dt,dx,c,T);
    
    cudaError_t err = cudaGetLastError();
    printf("Error: %s\n",cudaGetErrorString(err));

    // Copiar resultado al host
    cudaMemcpy(u, d_u, size*T, cudaMemcpyDeviceToHost);
    guardar_salida(u,Nx,Ny,T);
    clock_t end = clock();
 

    time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
 
    printf("todo %f seconds\n", time_spent);

	free(u); free(u_m1); free(u_p1);
    cudaFree(d_u); cudaFree(d_u_m1); cudaFree(d_u_p1);

	return 0;
}

