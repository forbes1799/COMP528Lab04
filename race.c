#include<stdio.h>
#include<stdlib.h>
#include<omp.h>

int main(void){

	const int MAX = 20;

	float array[MAX];
	float x[MAX];
	float y[MAX];	

	int i;
	
	for(i = 0; i < MAX; i++){
		array[i] = 0;
		x[i] = i;
		y[i] = x[i] + 1;
		
	}
	
	printf("X = ");
	for(i = 0; i < MAX; i++){
		printf("%f ", x[i]);
	}
	printf("\n");	

	printf("Y = ");
	for(i = 0; i < MAX; i++){
		printf("%f ", y[i]);
	}
	printf("\n");
	
	#pragma omp parallel for
	for(i = 1; i < MAX - 1; i++){
		array[i] += x[i] + y[i - 1];
		array[i + 1] += 2*x[i];
	}

	for(i = 0; i < MAX; i++){
		printf("%f ", array[i]); 
	}
	printf("\n");

	return 0; 
}
