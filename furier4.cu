#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include <time.h> 
#include <string.h>
#include <complex>
#pragma comment(lib, "cufft.lib")

#include <cufft.h>
#include <cufftw.h>

    
__global__ void masa_grilla(float *C, float *rho, float *r,float step_grid,int N_body,int N_g,float m_p,int t){

        int i =threadIdx.x + blockDim.x*blockIdx.x;

        if(i<N_body){
        if (r[t*N_body*2  +i*2]/step_grid>=0 && r[t*N_body*2  +i*2]/step_grid<N_g && r[t*N_body*2  +i*2+1]/step_grid>=0 && r[t*N_body*2  +i*2+1]/step_grid<N_g){

        int idx_grid= (int)round(r[t*N_body*2  +i*2]/step_grid);

        int idy_grid= (int)round(r[t*N_body*2  +i*2+1]/step_grid);
        
        int id_grid=idy_grid*N_g + idx_grid;

        float d_x = fabs(C[id_grid*2]-r[t*N_body*2  +i*2]) , d_y = fabs(C[id_grid*2+1]-r[t*N_body*2  +i*2+1]);
        float t_x = fabs(step_grid  -  d_x),  t_y = fabs(step_grid  -  d_y);
        int signox=1;
        int signoy=1;

        rho[id_grid] = (rho[id_grid] + m_p * t_x * t_y) /(step_grid*step_grid);

        if (C[id_grid*2]-r[t*N_body*2  +i*2]>0 ){signox=-1;}if( C[id_grid*2+1]-r[t*N_body*2  +i*2+1]>0){signoy=-1;}

        int celda_horizontal =  id_grid  + signox;
        int celda_vertical = id_grid +  signoy*N_g;
        int celda_inversa =  id_grid +  signox +  signoy*N_g;


        if (idx_grid >0  &&  idx_grid  < N_g-1){
            rho[celda_horizontal] = (rho[celda_horizontal] + m_p * d_x * t_y)/(step_grid*step_grid) ;} 

        if (idy_grid >0  &&  idy_grid < N_g-1){
            rho[celda_vertical] = (rho[celda_vertical] + m_p * t_x * d_y )/(step_grid*step_grid);}

        if (idx_grid >0  &&  idx_grid  < N_g-1  &&  idy_grid >0  &&  idy_grid < N_g-1){

            rho[celda_inversa] = (rho[celda_inversa] + m_p * d_x * d_y)/(step_grid*step_grid) ;}

        }
    }
    __syncthreads();


}
__global__ void real_complejo(float *f, cufftComplex *fc, int N){

    int idx=threadIdx.x + blockDim.x*blockIdx.x ;
	int idy=threadIdx.y + blockDim.y*blockIdx.y ;
    int id= idy*N+idx;
    if (idy<N && idx<N){
    fc[id].x=f[id]*4*3.14159;
    fc[id].y=0.0f;}
    
    __syncthreads();

}
__global__ void complejo_real(float *f, cufftComplex *fc, int N){

    int idx=threadIdx.x + blockDim.x*blockIdx.x ;
	int idy=threadIdx.y + blockDim.y*blockIdx.y  ;
    int id= idy*N+idx;
    if (idy<N && idx<N){
        f[id]=fc[id].x/((float)N*(float)N);}
    
    __syncthreads();

}
__global__ void poisson(cufftComplex *ft, cufftComplex *ft_k,float *k, int N, int inter,float step){

    int idx=threadIdx.x + blockDim.x*blockIdx.x  ;
	int idy=threadIdx.y + blockDim.x*blockIdx.y  ;
    int id= idy*N+idx;

    int i=0;
        if (idx<N && idy<N){
            float k2=k[idx]*k[idx]+k[idy]*k[idy];
            if (idx==0 && idy==0){k2=1.0f;}
            ft_k[id].x=-ft[id].x/(k2);
            ft_k[id].y=-ft[id].y/(k2);

            }

        
    __syncthreads();

}

__global__ void gravedad(float *phi_n, float *gravity,int N, float delta){

	int idx=threadIdx.x + blockDim.x*blockIdx.x  ;
	int idy=threadIdx.y + blockDim.y*blockIdx.y ;
    int id =idy*N+idx;
    phi_n[id]-=phi_n[0];
    if (idx<N && idy<N){

        gravity[id]= - (phi_n[idy*N+idx+1]-phi_n[idy*N+idx-1])/(2*delta);
        gravity[N*N+id]= - (phi_n[(idy+1)*N+idx]-phi_n[(idy-1)*N+idx])/(2*delta);
    
    }

    __syncthreads();

}

__global__ void actualizacion(float *C, float *r, float *v, float *gravity ,int N_body,int N_g,float  step_grid, float m_p, float dt, int t){

    int i =threadIdx.x + blockDim.x*blockIdx.x;
    float g_x,v_leap,g_y;

    if(i<N_body){

    if (r[t*N_body*2  +i*2]/step_grid>=0 && r[t*N_body*2  +i*2]/step_grid<N_g && r[t*N_body*2 + i*2+1]/step_grid>=0 && r[t*N_body*2  +i*2+1]/step_grid<N_g){

    int idx_grid= (int)round(r[t*N_body*2+i*2]/step_grid);

    int idy_grid= (int)round(r[t*N_body*2+i*2+1]/step_grid);
    int id_grid=idy_grid*N_g + idx_grid;

    float d_x = fabs(C[id_grid*2]-r[t*N_body*2+i*2]) , d_y = fabs(C[id_grid*2+1]-r[t*N_body*2+i*2+1]);
    float t_x = fabs(step_grid  -  d_x),  t_y = fabs(step_grid  -  d_y);


    int signox=1;
    int signoy=1;
    if (C[id_grid*2]-r[t*N_body*2  +i*2]>0.0 ){signox=-1;}if( C[id_grid*2+1]-r[t*N_body*2  +i*2+1]>0.0){signoy=-1;}


    g_x= gravity[id_grid]*t_x * t_y;
    g_y= gravity[N_g*N_g + id_grid]*t_x * t_y;



    int celda_horizontal =  id_grid  + signox;
    int celda_vertical = id_grid +  signoy*N_g;
    int celda_inversa =  id_grid +  signox +  signoy*N_g;


    if (idx_grid >0  &&  idx_grid  < N_g-1){
         g_x += gravity[celda_horizontal]*d_x * t_y;
         g_y += (gravity[N_g*N_g + celda_horizontal]* d_x * t_y );} 

    if (idy_grid >0  &&  idy_grid < N_g-1){
        g_x += gravity[celda_vertical]*t_x * d_y;
        g_y += (gravity[N_g*N_g + celda_vertical]* t_x * d_y );}

    if (idx_grid >0  &&  idx_grid  < N_g-1  &&  idy_grid >0  &&  idy_grid < N_g-1){
        g_x += gravity[celda_inversa]*d_x * d_y;
        g_y += (gravity[N_g*N_g + celda_inversa]* d_x * d_y );}


    }

        float tdt=dt;
//    act coordenda x        

        v_leap   =   v[i*2] + g_x*tdt/(2.0*m_p);
        r[(t+1)*N_body*2+i*2]= r[t*N_body*2+i*2] + v_leap*tdt;
        v[i*2]  =  v_leap+g_x*tdt/(2.0*m_p);

//    act coordenda y        
        v_leap  =  v[i*2+1] + g_y*tdt/(2.0*m_p);
        r[(t+1)*N_body*2+i*2+1]= r[t*N_body*2+i*2+1] + v_leap*tdt;
        v[i*2+1]   =   v_leap + g_y*tdt/(2.0*m_p);
    }
        __syncthreads();

}

void guardar_particulas(float *data,int size) {

    FILE *fp = fopen("n_body.dat", "w");
	
	fwrite(&(data[0]),sizeof(float),size,fp);
	fclose(fp);
}
void guardar_salida(float *data,int Nx, int Ny,int T) {
    FILE *fp = fopen("n_body2.dat", "w");

    for (int t=0;t<T;t++){
        for (int y=0;y<Ny;y++){
            for (int x=0;x<Nx;x++){
                fprintf(fp, "%g\n", data[t*Nx*Ny+y*Nx+x]);
            }
        }
    }
}
void guardar_masas(float *data, int size) {

    FILE *fp = fopen("masas.dat", "w");
	
	fwrite(&(data[0]),sizeof(float),size,fp);
	fclose(fp);
}
void rellenar_r(float *r,int size){

    for(int i=0;i<size;i++){
        r[i*2]=(float)rand()/(float)(RAND_MAX)+rand()%(1000)+0.01;
        r[i*2+1]=(float)rand()/(float)(RAND_MAX)+rand()%(1000)+0.01;
        
    }
}
void rellenar_v(float *r,int size){

    for(int i=0;i<size;i++){
        r[i*2]=(float)rand()/(float)(RAND_MAX)*0.0 ;
        r[i*2+1]=(float)rand()/(float)(RAND_MAX)*0.0;
        
    }
}
void rellenar_rho(float *masa,int size){

    for(int i=0;i<size;i++){

            masa[i]=0.0;

        
    }
}
void rellenar_grilla(float *grilla,int size,float step_grid){

    for(int i=0;i<size;i++){
        for(int j=i*size;j<size+i*size;j++){

            grilla[j*2+1]=i*step_grid;
            grilla[j*2]=(j-i*size)*step_grid;

        }
    }
}
void imprimir_archivo(float *v2,int size){
    FILE *arch; 
    arch=fopen("n_body.dat","rb");
    int numElem = fgetc(arch);
    fread(&v2, sizeof(float), numElem, arch);
    fclose(arch);
}


int main(int argc, char *argv[]){


    float *dot_r;
    float *cuda_dot_r;

    float *r_tn;
    float *cuda_r_tn;

    float *grilla;
    float *cuda_grilla;
    float *cuda_grilla_G;


    float *rho,*cuda_rho;

    float *phi,*cuda_phi_n;

    float *cuda_err;

    int N_grilla=256;
    int N_b=1000;
    int L_size = 1024;

    int size_cuerpos = 2*N_b*sizeof(float);
    int size_grilla = 2*N_grilla*N_grilla*sizeof(float);
    const int size_time=5000;

    float dt=0.03;

    double time_spent = 0.0;
    clock_t begin = clock();



    r_tn=(float *)malloc(size_time*size_cuerpos);
    dot_r=(float *)malloc(size_cuerpos);
    grilla=(float *)malloc(size_grilla);
    rho=(float  *)malloc(N_grilla*N_grilla*sizeof(float));

    phi=(float  *)malloc(N_grilla*N_grilla*sizeof(float));

    rellenar_rho(r_tn,N_b*size_time*2);

    rellenar_r(r_tn,N_b);

    rellenar_v(dot_r,N_b);

    rellenar_grilla(grilla,N_grilla,(float)L_size/N_grilla);
    rellenar_rho(rho,N_grilla*N_grilla);
    rellenar_rho(phi,N_grilla*N_grilla);








    cudaMalloc((void **)&cuda_r_tn, size_time*size_cuerpos);
    cudaMalloc((void **)&cuda_dot_r, size_cuerpos);
    cudaMalloc((void **)&cuda_grilla, size_grilla);
    cudaMalloc((void **)&cuda_grilla_G, size_grilla);

    cudaMalloc((void **)&cuda_phi_n, N_grilla*N_grilla*sizeof(float));
    cudaMalloc((void **)&cuda_rho, N_grilla*N_grilla*sizeof(float));

    cudaMemcpy(cuda_dot_r, dot_r, size_cuerpos, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_grilla, grilla, size_grilla, cudaMemcpyHostToDevice);

    cudaMemcpy(cuda_rho, rho, N_grilla*N_grilla*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_phi_n, phi, N_grilla*N_grilla*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(cuda_r_tn, r_tn, size_time*2*N_b*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&cuda_err, sizeof(float)*2);

//variables para la transfomada de fourier.

    float *k=(float  *)malloc(N_grilla*sizeof(float));
    for (int i =0; i<N_grilla/2;i++){
        k[i]=(i)*2*3.1415/((float)N_grilla);

        k[i+N_grilla/2]=(i-N_grilla/2)*2*3.1415/((float)N_grilla);
    }
    /*for (int i =0; i<N_grilla;i++){
        k[i]=(i)*2*3.1415/((float)N_grilla);
    }*/

    float *k_d;
    cudaMalloc((void**)&k_d, N_grilla*sizeof(float));
    cudaMemcpy(k_d, k, N_grilla*sizeof(float), cudaMemcpyHostToDevice);

    cufftComplex *ft_d,*f_dc,*ft_d_k,*u_dc;
    cudaMalloc((void**)&ft_d, N_grilla*N_grilla*sizeof(cufftComplex));
    cudaMalloc((void**)&ft_d_k ,N_grilla*N_grilla*sizeof(cufftComplex));
    cudaMalloc((void**)&f_dc, N_grilla*N_grilla*sizeof(cufftComplex));
    cudaMalloc((void**)&u_dc, N_grilla*N_grilla*sizeof(cufftComplex));


   int thread=32;
   dim3 bloque(thread,thread);
   dim3 grid((int)ceil((float)(N_grilla)/thread),(int)ceil((float)(N_grilla)/thread));

   cufftHandle plan1[size_time],plan2[size_time];
   cufftPlan2d(plan1,N_grilla,N_grilla,CUFFT_R2C);
   cufftPlan2d(plan2,N_grilla,N_grilla,CUFFT_C2R);

   printf("%f \n", dt);

    for (int t=0;t<size_time;t++){
        

        masa_grilla<<<(int)ceil((float)N_b/1024),1024>>>(cuda_grilla,cuda_rho,cuda_r_tn,(float) L_size/N_grilla,N_b ,N_grilla,1,t);
        cudaDeviceSynchronize();

        //eal_complejo<<<grid,bloque>>>(cuda_rho,f_dc,N_grilla);
        cudaDeviceSynchronize();
        cufftExecR2C(plan1[t], cuda_rho,ft_d);
        cudaDeviceSynchronize(); 

        poisson<<<grid,bloque>>>(ft_d,ft_d_k,k_d,N_grilla,100,(float) L_size/N_grilla);

        cudaDeviceSynchronize();
        cufftExecC2R(plan2[2],ft_d_k,cuda_phi_n);
        cudaDeviceSynchronize();

        //complejo_real<<<grid,bloque>>>(cuda_phi_n,u_dc,N_grilla);
        cudaDeviceSynchronize();


        gravedad<<<grid,bloque>>>(cuda_phi_n,cuda_grilla_G,N_grilla,(float) L_size/N_grilla);
        cudaDeviceSynchronize();


        actualizacion<<<(int)ceil((float)N_b/1024.0),1024>>>(cuda_grilla,cuda_r_tn,cuda_dot_r,cuda_grilla_G, N_b, N_grilla,  (float)L_size/N_grilla,  1, dt,  t );
        cudaDeviceSynchronize();
        if (t==0){
            cudaMemcpy(rho,cuda_rho,N_grilla*N_grilla*sizeof(float) , cudaMemcpyDeviceToHost);

        }  
    }

        cudaMemcpy(r_tn,cuda_r_tn, size_time*size_cuerpos, cudaMemcpyDeviceToHost);




        clock_t end = clock();
        
        time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
        printf("\nTiempo de ejecucion : %f seconds\n", time_spent);
		

        //guardar_particulas(r_tn,size_time*N_b*2);
        guardar_salida(r_tn,2,N_b,size_time);

        guardar_masas(rho,N_grilla*N_grilla);
    cudaError_t err = cudaGetLastError();
    printf("Error: %s\n",cudaGetErrorString(err));
free(dot_r); free(grilla); free(r_tn); free(phi); free(rho);

 cudaFree(cuda_dot_r); cudaFree(cuda_grilla);cudaFree(cuda_rho); cudaFree(cuda_err); cudaFree(cuda_grilla_G);
    cudaFree(cuda_phi_n); cudaFree(cuda_r_tn);


}