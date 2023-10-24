#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<omp.h>

#define N 10

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
    
    float bestProfit = 0;
    char bestStockItem[500];

    for(int i = 0; i < N; i++){
        float profit = stockSellPrice[i] - stockCost[i];
        if(profit > bestProfit){
            bestProfit = profit;
            strcpy(bestStockItem, stockItem[i]);
        }
    }

    printf("Stock item '%s' would make the most profit at £%.2f, for each item sold.\n", bestStockItem, bestProfit);
}