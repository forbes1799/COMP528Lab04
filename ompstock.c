#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<omp.h>

#define N 10

/*
Program for returning multiple values using OMP
*/

int main(void){

    float stockCost[N] = {10.99, 2.99, 1.40, 1.50, 1.10, 3.50, 6.00, 7.43, 4.12, 100.01};
    float stockSellPrice[N] = {12.00, 3.50, 2.40, 2.01, 1.15, 3.60, 6.80, 8.00, 4.22, 100.02};
    char stockItem[N][500]={{"Batteries"},
                {"Eggs"},
                {"Bread"},
                {"Milk"},
                {"Potatoes"},
                {"Mayonaise"},
                {"Pizza"},
                {"Lasagna"},
                {"Cheese"},
                {"Butter"}
                };

    /*Declaration of variables*/
    float bestProfit = 0;
    char bestItem[500];

    /*
	Way to get the maximum number of threads possible, outside of a parallel region
	Different to omp_get_num_threads which only gets the current number of active threads
	*/
    int numThreads = omp_get_max_threads();
    
    /*
	Set up arrays to reserve memory locations for the threads using malloc
    Arrays are equal to the size that has been allocated for them
	*/
    float *threadBestProfit = (float *)malloc(numThreads * sizeof(float));
    char **threadBestItem = (char **)malloc(numThreads * sizeof(char *));

    for(int j = 0; j < N; j++){
        threadBestItem[j] = (char *)malloc(500 * sizeof(char));
    }

    #pragma omp parallel
    {
        /*Get the threadID to access the thread arrays*/
        int threadID = omp_get_thread_num();
        threadBestProfit[threadID] = 0;

        #pragma omp parallel for
        for(int i = 0; i < N; i++){
            float profit = stockSellPrice[i] - stockCost[i];
            if(profit > /*bestProfit*/){
                /*bestProfit*/ = profit;
                strcpy(/*bestItem*/, stockItem[i]);
            }
        }

        #pragma omp single
        {
            /*
            TASK
            Check through each thread's memory location
            update bestProfit if it makes the most profit
            Update the bestItem if that item's profit is the most.
            */
           
        }
        
    }

    printf("Stock item '%s' would make the most profit at £%.2f, for each item sold\n", bestItem, bestProfit);

    
}