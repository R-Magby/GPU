#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include <time.h> 


__global__ void onda2d(float *u, float *u_m1, float *u_p1,float dt,float dx,float c,int T,int Nx, int Ny,int t) {
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;

    if (x>0 && x<Nx-1 && y>0 && y<Ny-1){

            u_p1[y*Ny+x]=dt*dt/(dx*dx)*c*c*(u[t*Nx*Ny+y*Ny+x+1]+u[t*Nx*Ny+y*Ny+x-1]+
                u[t*Nx*Ny+(y+1)*Ny+x]+u[t*Nx*Ny+(y-1)*Ny+x]-4.0*u[t*Nx*Ny+y*Ny+x]) + 2.0*u[t*Nx*Ny+y*Ny+x] - u_m1[y*Ny+x];
            
		u_m1[y*Ny+x]=u[t*Nx*Ny+y*Ny+x];

            u[(t+1)*Nx*Ny+y*Ny+x]=u_p1[y*Ny+x];
    

    }
     __syncthreads();


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
	
	fwrite(&(data[0]),sizeof(float),Nx*Ny*T,fp);
	fclose(fp);
}

int main(int argc, char *argv[]){
	float  *u_s1, *u_m1_s1, *u_p1_s1,*d_u_s1, *d_u_m1_s1, *d_u_p1_s1;
        float  *u_s2, *u_m1_s2, *u_p1_s2,*d_u_s2, *d_u_m1_s2, *d_u_p1_s2;


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

	u_s1 = (float *)malloc(size*T); gauss(u_s1,dt,c,dx,dy, 3*Nx/4,3*Ny/4,Nx,Ny);
	u_m1_s1 = (float *)malloc(size); gauss(u_m1_s1,0,c,dx,dy, 3*Nx/4, 3*Ny/4,Nx,Ny);
	u_p1_s1 = (float *)malloc(size);

        u_s2 = (float *)malloc(size*T); gauss(u_s2,dt,c,dx,dy, Nx/4,Ny/4,Nx,Ny);
        u_m1_s2 = (float *)malloc(size); gauss(u_m1_s2,0,c,dx,dy, Nx/4, Ny/4,Nx,Ny);
        u_p1_s2 = (float *)malloc(size);




    // Asignar memoria al lado del device (GPU)
    cudaMalloc((void **)&d_u_s1, size*T);
    cudaMalloc((void **)&d_u_m1_s1, size);
    cudaMalloc((void **)&d_u_p1_s1, size);


    cudaMalloc((void **)&d_u_s2, size*T);
    cudaMalloc((void **)&d_u_m1_s2, size);
    cudaMalloc((void **)&d_u_p1_s2, size);

cudaStream_t stream1, stream2;
cudaStreamCreate ( &stream1) ;
cudaStreamCreate ( &stream2) ;

   
    // Copiar al device
    cudaMemcpyAsync(d_u_s1, u_s1, size*T, cudaMemcpyHostToDevice,stream1);
    cudaMemcpyAsync(d_u_m1_s1, u_m1_s1, size, cudaMemcpyHostToDevice,stream1);

    cudaMemcpyAsync(d_u_s2, u_s2, size*T, cudaMemcpyHostToDevice,stream2);
    cudaMemcpyAsync(d_u_m1_s2, u_m1_s2, size, cudaMemcpyHostToDevice,stream2);


    // Invocar kernel con N bloques de 1 thread cada uno
    dim3 bloque (32,32);
    dim3 grid(2,2);

    printf("%d %d\n",(int)ceil(Nx/32),(int)ceil(Ny/32));
    

    for (int t=0;t<T;t++){ 
    	onda2d<<<grid,bloque,0,stream1>>>(d_u_s1,d_u_m1_s1,d_u_p1_s1,dt,dx,c,T,Nx,Ny,t);
        onda2d<<<grid,bloque,0,stream2>>>(d_u_s2,d_u_m1_s2,d_u_p1_s2,dt,dx,c,T,Nx,Ny,t);

   }
    cudaError_t err = cudaGetLastError();
    printf("Error: %s\n",cudaGetErrorString(err));

    // Copiar resultado al host
       cudaMemcpyAsync(u_s2, d_u_s2, size*T, cudaMemcpyDeviceToHost,stream1);
       cudaMemcpyAsync(u_s1, d_u_s1, size*T, cudaMemcpyDeviceToHost,stream2);
	
	float *phi_final=u_s1+u_s2;

	 guardar_salida(phi_final,Nx,Ny,T);
    clock_t end = clock();
 

    time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
 
    printf("todo %f seconds\n", time_spent);

	free(u_s1); free(u_m1_s1); free(u_p1_s1);
    cudaFree(d_u_s1); cudaFree(d_u_m1_s1); cudaFree(d_u_p1_s1);

        free(u_s2); free(u_m1_s2); free(u_p1_s2);
    cudaFree(d_u_s2); cudaFree(d_u_m1_s2); cudaFree(d_u_p1_s2);

	return 0;
}
