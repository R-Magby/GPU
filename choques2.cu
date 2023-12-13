#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include <time.h> 
#include <string.h>
#define atoa(x) #x

    
__global__ void masa_grilla(float *C, float *rho, float *temp,float *r,float step_grid,int N_body,int N_g,float m_p,int t){

        int i =threadIdx.x + blockDim.x*blockIdx.x;

        if(i<N_body){
            float gridx=temp[i*3  ]/step_grid;
            float gridy=temp[i*3+1]/step_grid;
            float gridz=temp[i*3+2]/step_grid;
    
         if (gridx>=0 &&gridx<N_g && gridy>=0 && gridy<N_g && gridz>=0 && gridz<N_g){
    
         int idx_grid= (int)round(gridx);
         int idy_grid= (int)round(gridy);
         int idz_grid= (int)round(gridz);
    
         
         int id_grid= idz_grid * N_g*N_g  + idy_grid * N_g + idx_grid;
    
         float d_x = fabs(C[id_grid*3]-temp[i*3  ]) , d_y = fabs(C[id_grid*3+1]-temp[i*3+1]),d_z = fabs(C[id_grid*3+2]-temp[i*3+2]);
         float t_x = fabs(step_grid  -  d_x),  t_y = fabs(step_grid  -  d_y),t_z = fabs(step_grid  -  d_z);
         int signox=1;
         int signoy=1;
         int signoz=1;
    

        float delta3d=step_grid*step_grid*step_grid;
        rho[id_grid] = (rho[id_grid] + m_p * t_x * t_y * t_z) /(delta3d);

        if (C[id_grid*3]-temp[i*3  ]>0 ){signox=-1;}if( C[id_grid*3+1]-temp[i*3+1]>0){signoy=-1;}if( C[id_grid*3+2]-temp[i*3+2]>0){signoz=-1;}

        int celda_xx =  id_grid  + signox;
        int celda_yy = id_grid +  signoy*N_g;
        int celda_zz = id_grid +  signoz*N_g*N_g;

        int celda_xy =  id_grid +  signox +  signoy*N_g;
        int celda_xz =  id_grid +  signox +  signoz*N_g*N_g;
        int celda_yz =  id_grid +  signoy*N_g +  signoz*N_g*N_g;

        int celda_xyz =  id_grid + signox + signoy*N_g +  signoz*N_g*N_g;



        if (idx_grid >0  &&  idx_grid  < N_g-1){
            rho[celda_xx] = (rho[celda_xx] + m_p * d_x * t_y * t_z)/(delta3d) ;} //xx
        if (idy_grid >0  &&  idy_grid < N_g-1){
            rho[celda_yy] = (rho[celda_yy] + m_p * t_x * d_y *t_z)/(delta3d);} //yy
        if (idz_grid >0  &&  idz_grid  < N_g-1){
            rho[celda_zz] = (rho[celda_zz] + m_p * t_x * t_y *d_z)/(delta3d) ;} //zz

        if (idx_grid >0  &&  idx_grid  < N_g-1  &&  idy_grid >0  &&  idy_grid < N_g-1){//xy
            rho[celda_xy] = (rho[celda_xy] + m_p * d_x * d_y * t_z)/(delta3d) ;}
        if (idx_grid >0  &&  idx_grid  < N_g-1  &&  idz_grid >0  &&  idz_grid < N_g-1){//xz
            rho[celda_xz] = (rho[celda_xz] + m_p * d_x * t_y * d_z)/(delta3d) ;}
        if (idy_grid >0  &&  idy_grid  < N_g-1  &&  idz_grid >0  &&  idz_grid < N_g-1){//yz
            rho[celda_yz] = (rho[celda_yz] + m_p * t_x * d_y * d_z)/(delta3d) ;}

        if (idx_grid >0  &&  idx_grid  < N_g-1  &&  idy_grid >0  &&  idy_grid < N_g-1 &&  idz_grid >0  &&  idz_grid < N_g-1){//xyz
            rho[celda_xyz] = (rho[celda_xyz] + m_p * d_x * d_y *d_z)/(delta3d) ;}

        }
    }
            __syncthreads();

}

__global__ void potencial(float *phi_n, float *masa, float *err,int N, float delta, int inter){

	int idx=threadIdx.x + blockDim.x*blockIdx.x  ;
	int idy=threadIdx.y + blockDim.y*blockIdx.y;
	int idz=threadIdx.z + blockDim.z*blockIdx.z;

    int i=1;
    float err_min=0.0001;
    float thread_error;
    float temp;
    float phi_n1;

    err[0]=1.0;
    err[1]=0.0;

    while (i<inter && err[0] >= err_min){
        thread_error=0.0;        
        err[1]=0.0;

        if (idx<N-1 && idy<N-1 && idx>0 && idy>0 && idz<N-1 && idz>0 ){
            phi_n1 = (
            phi_n[idz*N*N+ (idy+1)*N+idx]+  phi_n[idz*N*N+(idy-1)*N+idx]+
            phi_n[idz*N*N+(idy)*N+idx+1] +   phi_n[idz*N*N+(idy)*N+idx-1]+
            phi_n[(idz+1)*N*N+ (idy)*N+idx]  +  phi_n[(idz-1)*N*N+ (idy)*N+idx]
                -4.0*3.1415*delta*delta*masa[idz*N*N+idy*N+idx])/6.0;

            thread_error += (phi_n1-phi_n[idz*N*N+idy*N+idx])*(phi_n1-phi_n[idz*N*N+idy*N+idx]);

        
        err[1]+=thread_error;

    __syncthreads();



    temp=phi_n[idz*N*N+idy*N+idx];
    phi_n[idz*N*N+idy*N+idx]=phi_n1;
    phi_n1=temp;
    //phi_n[idy*N+idx]=phi_n1[idy*N+idx];
    err[0]=err[1];

    i++;
  //  if(i%100==0){
   //     printf("i:%d | error= %f\n",i,err[0]);
    //}
    
    }
    __syncthreads();


}

__global__ void gravedad(float *phi_n, float *gravity,int N, float delta){

	int idx=threadIdx.x + blockDim.x*blockIdx.x  ;
	int idy=threadIdx.y + blockDim.y*blockIdx.y   ;
	int idz=threadIdx.z + blockDim.z*blockIdx.z;
    int id =idz*N*N+idy*N+idx;

    if (idx<N-1 && idy<N-1 && idx>0 && idy>0 && idz<N-1 && idz>0 ){

        gravity[id*3]=   -(phi_n[idz*N*N+idy*N+idx+1]   - phi_n[idz*N*N+idy*N+idx-1])/(2*delta);
        gravity[id*3+1]= -(phi_n[idz*N*N+(idy+1)*N+idx] - phi_n[idz*N*N+(idy-1)*N+idx])/(2*delta);
        gravity[id*3+2]= -(phi_n[(idz+1)*N*N+idy*N+idx] - phi_n[(idz-1)*N*N+ idy*N+idx])/(2*delta);
    }
    else if(idx==N-1 || idy==N-1 || idx==0|| idy==0 || idz==N-1 || idz==0 ){//ultimo
        gravity[id*3]=   0.0;
        gravity[id*3+1]= 0.0;
        gravity[id*3+2]= 0.0;
    }

    __syncthreads();

}

__global__ void actualizacion(float *C, float *temp,float *r, float *v, float *gravity ,int N_body,int N_g,float  step_grid, float m_p, float dt, int t){

    int i =threadIdx.x + blockDim.x*blockIdx.x;
    float g_x,v_leap,g_y,g_z;

    if(i<N_body){
        float gridx=temp[i*3  ]/step_grid;
        float gridy=temp[i*3+1]/step_grid;
        float gridz=temp[i*3+2]/step_grid;

     if (gridx>=0 &&gridx<N_g && gridy>=0 && gridy<N_g && gridz>=0 && gridz<N_g){

     int idx_grid= (int)round(gridx);
     int idy_grid= (int)round(gridy);
     int idz_grid= (int)round(gridz);

     
     int id_grid= idz_grid * N_g*N_g  + idy_grid * N_g + idx_grid;

     float d_x = fabs(C[id_grid*3]-temp[i*3  ]) , d_y = fabs(C[id_grid*3+1]-temp[i*3+1]),d_z = fabs(C[id_grid*3+2]-temp[i*3+2]);
     float t_x = fabs(step_grid  -  d_x),  t_y = fabs(step_grid  -  d_y),t_z = fabs(step_grid  -  d_z);
     int signox=1;
     int signoy=1;
     int signoz=1;


     if (C[id_grid*3]-temp[i*3  ]>0 ){signox=-1;}if( C[id_grid*3+1]-temp[i*3+1]>0){signoy=-1;}if( C[id_grid*3+2]-temp[i*3+2]>0){signoz=-1;}

     int celda_xx =  id_grid  + signox;
     int celda_yy = id_grid +  signoy*N_g;
     int celda_zz = id_grid +  signoz*N_g*N_g;

     int celda_xy =  id_grid +  signox +  signoy*N_g;
     int celda_xz =  id_grid +  signox +  signoz*N_g*N_g;
     int celda_yz =  id_grid +  signoy*N_g +  signoz*N_g*N_g;

     int celda_xyz =  id_grid + signox + signoy*N_g +  signoz*N_g*N_g;


    g_x = gravity[id_grid*3]*t_x * t_y *t_z;
    g_y = gravity[id_grid*3+1]*t_x * t_y *t_z;
    g_z = gravity[id_grid*3+2]*t_x * t_y *t_z;


    if (idx_grid >0  &&  idx_grid  < N_g-1){
        g_x += gravity[celda_xx*3]* d_x * t_y * t_z ;
        g_y += (gravity[celda_xx*3+1]* d_x * t_y * t_z) ;
        g_z += (gravity[celda_xx*3+2]* d_x * t_y * t_z) ;} //xx
    if (idy_grid >0  &&  idy_grid < N_g-1){
        g_x += (gravity[celda_yy*3]* t_x * d_y * t_z) ;
        g_y += (gravity[celda_yy*3+1]* t_x * d_y * t_z) ;
        g_z += (gravity[celda_yy*3+2]* t_x * d_y * t_z) ;} //yy
    if (idz_grid >0  &&  idz_grid  < N_g-1){
        g_x += (gravity[celda_zz*3]*t_x * t_y * d_z) ;
        g_y += (gravity[celda_zz*3+1]* t_x * t_y * d_z) ;
        g_z += (gravity[celda_zz*3+2]* t_x * t_y * d_z) ; } //zz

    if (idx_grid >0  &&  idx_grid  < N_g-1  &&  idy_grid >0  &&  idy_grid < N_g-1){//xy
        g_x += (gravity[celda_xy*3]* d_x * d_y * t_z) ;
        g_y += (gravity[celda_xy*3+1]* d_x * d_y * t_z) ;
        g_z += (gravity[celda_xy*3+2]* d_x * d_y * t_z) ; }
    if (idx_grid >0  &&  idx_grid  < N_g-1  &&  idz_grid >0  &&  idz_grid < N_g-1){//xz
        g_x += (gravity[celda_xz*3]* d_x * t_y * d_z) ;
        g_y += (gravity[celda_xz*3+1]* d_x * t_y * d_z) ;
        g_z += (gravity[celda_xz*3+2]* d_x * t_y * d_z) ; }
    if (idy_grid >0  &&  idy_grid  < N_g-1  &&  idz_grid >0  &&  idz_grid < N_g-1){//yz
        g_x += (gravity[celda_yz*3]* t_x * d_y * d_z) ;
        g_y += (gravity[celda_yz*3+1]* t_x * d_y * d_z) ;
        g_z += (gravity[celda_yz*3+2]* t_x * d_y * d_z) ; }

    if (idx_grid >0  &&  idx_grid  < N_g-1  &&  idy_grid >0  &&  idy_grid < N_g-1 &&  idz_grid >0  &&  idz_grid < N_g-1){//xyz
        g_x += (gravity[celda_xyz*3]* d_x * d_y * d_z) ;
        g_y += (gravity[celda_xyz*3+1]* d_x * d_y * d_z) ;
        g_z += (gravity[celda_xyz*3+2]* d_x * d_y * d_z) ; }

    }
    if (t%10==0){
        //    act coordenda x        

        v_leap   =   v[i*3] + g_x*dt/(2.0*m_p);
        r[(t/10+1)*N_body*3+i*3]= temp[i*3+0] + v_leap*dt;
        temp[i*3]=r[(t/10+1)*N_body*3+i*3];

        v[i*3]  =  v_leap+g_x*dt/(2.0*m_p);

//    act coordenda y        
        v_leap  =  v[i*3+1] + g_y*dt/(2.0*m_p);
        r[(t/10+1)*N_body*3+i*3+1]= temp[i*3+1] + v_leap*dt;
        temp[i*3+1]=r[(t/10+1)*N_body*3+i*3+1];
        v[i*3+1]   =   v_leap + g_y*dt/(2.0*m_p);

//    act coordenda z
        v_leap  =  v[i*3+2] + g_z*dt/(2.0*m_p);
        r[(t/10+1)*N_body*3+i*3+2]= temp[i*3+2] + v_leap*dt;
        temp[i*3+2]=r[(t/10+1)*N_body*3+i*3+2];

        v[i*3+2]   =   v_leap + g_z*dt/(2.0*m_p);
    }
    else{
//    act coordenda x        

        v_leap   =   v[i*3] + g_x*dt/(2.0*m_p);
        temp[i*3+0]= temp[i*3+0] + v_leap*dt;
        v[i*3]  =  v_leap+g_x*dt/(2.0*m_p);

//    act coordenda y        
        v_leap  =  v[i*3+1] + g_y*dt/(2.0*m_p);
        temp[i*3+1]= temp[i*3+1] + v_leap*dt;
        v[i*3+1]   =   v_leap + g_y*dt/(2.0*m_p);

//    act coordenda z
        v_leap  =  v[i*3+2] + g_z*dt/(2.0*m_p);
        temp[i*3+2]= temp[i*3+2] + v_leap*dt;
        v[i*3+2]   =   v_leap + g_z*dt/(2.0*m_p);
    }
    }
        __syncthreads();

}
void guardar_grilla(float *data, int size) {

    FILE *fp = fopen("grilla.dat", "w");
	
	fwrite(&(data[0]),sizeof(float),size,fp);
	fclose(fp);
}

void guardar_particulas(char *i,float *data,int size) {
    char arch[20];
    //printf("%s",i);

    strcat(strcpy(arch,"n_body"),i);
    strcat(arch,".dat");

    //printf("%s",arch);

    FILE *fp = fopen(arch, "wb");
    //printf("%s",arch);
	
	fwrite(&(data[0]),sizeof(float),size,fp);
	fclose(fp);
}

void guardar_salida(float *data,int Nx, int Ny,int T) {
    FILE *fp = fopen("n_body2.dat", "w");

    for (int t=0;t<T;t++){
            for (int y=0;y<Ny;y++){
                for (int x=0;x<Nx;x++){
                    fprintf(fp, "%g\n", data[t*Nx*Ny+y*Nx+ x]);
                
            }
        }
    }
}
void guardar_masas(float *data, int size) {

    FILE *fp = fopen("masas.dat", "w");
	
	fwrite(&(data[0]),sizeof(float),size,fp);
	fclose(fp);
}
void rellenar_r(float *r,int size,int  centro_x,int  centro_y,int  centro_z, int ini){
    int radio=100;
    int pos_x,pos_y,pos_z;
    pos_x=centro_x-radio;
    pos_y=centro_y-radio;
    pos_z=centro_z-radio;
    for(int i=ini;i<size;i++){
            
        r[i*3]=rand()%(2*radio+1)+pos_x;
        r[i*3+1]=rand()%(2*radio+1)+pos_z;
        r[i*3+2]=rand()%(2*radio+1)+pos_y;

        
    }
}
void rellenar_v(float *v,float *r, int size,int ini,float rap){
    //float radio,rx,rz;
    for(int i=ini;i<size;i++){
        //radio= sqrt((r[i*3]-32)*(r[i*3]-32)+(r[i*3+1]-32)*(r[i*3+1]-32)+(r[i*3+2]-32)*(r[i*3+2]-32));
        v[i*3]=(float)rand()/(float)(RAND_MAX)*rap ;
        v[i*3+1]=0.7*(float)rand()/(float)(RAND_MAX)*rap;
        v[i*3+2]=(float)rand()/(float)(RAND_MAX)*rap;

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
            for(int k=j*size;k<size+j*size;k++){

                grilla[k*3+2]=i*step_grid;
                grilla[k*3+1]=(j-i*size)*step_grid;
                grilla[k*3]=(k-j*size)*step_grid;
            
            }
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
    float *cuda_r_tn,*temp_r;

    float *grilla;
    float *cuda_grilla;
    float *cuda_grilla_G;


    float *rho,*cuda_rho;

    float *phi,*cuda_phi_n;

    float *cuda_err;

    int N_grilla=256;
    int N_b=1000;
    int L_size = 512;// dejando el valor en 1 se evita el error de acceso de memoria

    int size_cuerpos = 3*N_b*sizeof(float);
    int size_grilla = 3*N_grilla*N_grilla*N_grilla*sizeof(float);

    int step_save=10;
    int size_time=500;

    float dt=0.002;// el valor de dt tiene  q ser 1/T si no error de acceso de memoria, act: si L_size funciona para cualquier valor de variable, L_size era el problema

    double time_spent = 0.0;
    clock_t begin = clock();



    r_tn=(float *)malloc(size_time*size_cuerpos);
    dot_r=(float *)malloc(size_cuerpos);
    grilla=(float *)malloc(size_grilla);
    rho=(float  *)malloc(N_grilla*N_grilla*N_grilla*sizeof(float));

    phi=(float  *)malloc(N_grilla*N_grilla*N_grilla*sizeof(float));

    rellenar_rho(r_tn,N_b*size_time*3);

    rellenar_r(r_tn,N_b/2,150,200,150,0);
    rellenar_r(r_tn,N_b,350,300,350,N_b/2);


    rellenar_v(dot_r,r_tn,N_b/2,0,20.0);
    rellenar_v(dot_r,r_tn,N_b,N_b/2,-20.0);


    rellenar_grilla(grilla,N_grilla,(float)L_size/N_grilla);
    rellenar_rho(rho,N_grilla*N_grilla*N_grilla);
    rellenar_rho(phi,N_grilla*N_grilla*N_grilla);








    cudaMalloc((void **)&cuda_r_tn, size_time*size_cuerpos);
    cudaMalloc((void **)&temp_r, size_cuerpos);

    cudaMalloc((void **)&cuda_dot_r, size_cuerpos);
    cudaMalloc((void **)&cuda_grilla, size_grilla);
    cudaMalloc((void **)&cuda_grilla_G, size_grilla);

    cudaMalloc((void **)&cuda_phi_n, N_grilla*N_grilla*N_grilla*sizeof(float));
    cudaMalloc((void **)&cuda_rho, N_grilla*N_grilla*N_grilla*sizeof(float));

    cudaMemcpy(cuda_dot_r, dot_r, size_cuerpos, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_grilla, grilla, size_grilla, cudaMemcpyHostToDevice);

    cudaMemcpy(cuda_rho, rho, N_grilla*N_grilla*N_grilla*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_phi_n, phi, N_grilla*N_grilla*N_grilla*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(cuda_r_tn, r_tn, size_time*3*N_b*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(temp_r, r_tn, 3*N_b*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&cuda_err, sizeof(float)*2);

   int thread=16;
   dim3 bloque(thread,thread,4);
   dim3 grid((int)ceil((float)(N_grilla)/thread),(int)ceil((float)(N_grilla)/thread),(int)ceil((float)(N_grilla)/4));

   
   printf("%f \n", dt);
    for (int i=0;i<10;i++){
        for (int t=0;t<size_time*step_save;t++){
            if (t%200==0){
                printf("t=%d\n",t);
            }
    
            masa_grilla<<<(int)ceil((float)N_b/1024),1024>>>(cuda_grilla,cuda_rho,temp_r,cuda_r_tn,(float) L_size/N_grilla ,N_b,N_grilla,1,t%size_time);
            cudaDeviceSynchronize();


            potencial<<<grid,bloque>>>(cuda_phi_n,cuda_rho,cuda_err,N_grilla,(float) L_size/N_grilla,10000);
            cudaDeviceSynchronize();


            gravedad<<<grid,bloque>>>(cuda_phi_n,cuda_grilla_G,N_grilla,(float) L_size/N_grilla);
            cudaDeviceSynchronize();


            actualizacion<<<(int)ceil((float)N_b/1024),1024>>>(cuda_grilla,temp_r,cuda_r_tn,cuda_dot_r,cuda_grilla_G, N_b, N_grilla,  (float)L_size/N_grilla,  1, dt,  t );
            cudaDeviceSynchronize();

            }
            
            char str[10];
            sprintf(str, "%d", i);
            cudaMemcpy(r_tn,cuda_r_tn, size_time*size_cuerpos, cudaMemcpyDeviceToHost);
            
            printf("%s\n",str);
            guardar_particulas(str,r_tn,3*N_b*size_time);
            printf("ri=[%f | %f |%f]",r_tn[0],r_tn[1],r_tn[2]);

            printf("rf=[%f | %f |%f]",r_tn[(size_time-1)*N_b*3],r_tn[(size_time-1)*N_b*3+1],r_tn[(size_time-1)*N_b*3+2]);
            cudaMemcpy(cuda_r_tn, &r_tn[(size_time-1)*N_b*3], 3*N_b*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(temp_r, &r_tn[(size_time-1)*N_b*3], 3*N_b*sizeof(float), cudaMemcpyHostToDevice);


            cudaDeviceSynchronize();

        }


        clock_t end = clock();
        
        time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
        printf("\nTiempo de ejecucion : %f seconds\n", time_spent);
		
       // guardar_particulas(arch,r_tn,3*N_b*size_time);
       // guardar_salida(r_tn,3,N_b,size_time);
    cudaError_t err = cudaGetLastError();
    printf("Error: %s\n",cudaGetErrorString(err));
    free(r_tn); free(dot_r); free(grilla); free(r_tn); free(phi); free(rho);

    cudaFree(cuda_r_tn); cudaFree(cuda_dot_r); cudaFree(cuda_grilla);cudaFree(cuda_rho); cudaFree(cuda_err); cudaFree(cuda_grilla_G);
    cudaFree(cuda_phi_n); cudaFree(cuda_r_tn);


}