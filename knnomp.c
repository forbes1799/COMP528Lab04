#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<omp.h>

//=======file-reader.c functions=======//
int readNumOfPoints(char*);
int readNumOfFeatures(char*);
int readNumOfClasses(char*);
double *readDataPoints(char*, int, int);
void *writeResultsToFile(double*, int, int, char*);

//=======this file's functions=======//
int classify(double*, double*, int, int, int, int, int);
int modifyNearestNeighbours(int**, double**, double, int, int, int);
double getDistance(int, int, double*, double*, int);

int main(int argc, char* argv[]){

    printf("\n\n===============STARTING KNN===============\n\n");   

    double frS = omp_get_wtime();

    //Load arguments 
    char *inFile = argv[1];
    char *testFile = argv[2];
    char *outFile = argv[3];
    int k = atoi(argv[4]);

    //Read training data
    int numTrainPoints = readNumOfPoints(inFile);
    int numFeatures = readNumOfFeatures(inFile);
    int numClasses = readNumOfClasses(inFile);
    double *train_data = readDataPoints(inFile, numTrainPoints, numFeatures);

    //Read test data
    int numTestPoints = readNumOfPoints(testFile);
    double *test_data = readDataPoints(testFile, numTestPoints, numFeatures);

    double frE = omp_get_wtime();

    //Classify the dataset
    double s_time = omp_get_wtime();
    classify(train_data, test_data, numTrainPoints, numTestPoints, numFeatures, numClasses, k);
    double e_time = omp_get_wtime();

    printf("\nProgram took %lf ms to run\n", (e_time - s_time) * 1000 );

    double fwS = omp_get_wtime();

    writeResultsToFile(test_data, numTestPoints, numFeatures, outFile);

    printf("\n\n===============ENDING KNN===============\n\n");

    free(test_data);
    free(train_data);

    double fwE = omp_get_wtime();

    printf("Reading files took: %f seconds\n\n", frE - frS);

    printf("Parallel time took: %f seconds\n\n", e_time - s_time);

    printf("Writing file took: %f seconds \n\n", fwE - fwS);

    printf("Overall took: %f seconds \n\n", fwE - frS);

    return 0;
}

/**
 * Predicts labels using knn
 * @param train: The train dataset as a 1-D array
 * @param test: The test dataset to be predicted as a 1-D array
 * @param numTrainPoints: The number of points in the train dataset
 * @param numTestPoints: The number of points in the test dataset
 * @param numFeatures: The number of features in both datasets
 * @param numClasses: The maximum numbe of classes
 * @param k: The number of nearest neighbours to evaluate
*/
int classify (double *train, double *test, int numTrainPoints, int numTestPoints, int numFeatures, int numClasses, int k){

    //Final column
    int labelIndex = numFeatures - 1;

    printf("Beginning classification process:\n");

    //Arrays for the values and nearest neighbours
    int **nearest = (int**)malloc(numTestPoints * sizeof(int*));
    double **nearestValues = (double**)malloc(numTestPoints * sizeof(double*));

    if(nearest == NULL || nearestValues == NULL){
        return EXIT_FAILURE;
    }

    for(int i = 0; i < numTestPoints; i++){
        nearest[i] = (int*)malloc(k * sizeof(int));
        nearestValues[i] = (double*)malloc(k * sizeof(double));
        if(nearest[i] == NULL || nearestValues[i] == NULL){
            return EXIT_FAILURE;
        }
    }

    //Initialise parallel region
    #pragma omp parallel default(none) shared(train, test, numTrainPoints,\
                                              numTestPoints, numFeatures, \
                                              labelIndex, numClasses, k, \
                                              nearest, nearestValues)
    {
        double threadS = omp_get_wtime();
        //Reshape 1-D array
        double (*train_data)[numFeatures] = (double(*)[numFeatures]) train;
        double (*test_data)[numFeatures] = (double(*)[numFeatures]) test;

        //Initialise nearest values - Nowait to immediately move onto next for loop without waiting for synchronisation
        #pragma omp for
        for(int i = 0; i < numTestPoints; i++){
            #pragma omp simd
            for(int j = 0; j < k; j++){
                nearest[i][j] = -1;
                nearestValues[i][j] = __DBL_MAX__;
            }
        }

        #pragma omp master 
        printf("Finding the nearest neighbours\n");

        //Find the nearest neighbours. 
        #pragma omp for 
        for(int i = 0; i < numTestPoints; i++){
            for(int j = 0; j < numTrainPoints; j++){
                double dist = getDistance(i, j, train, test, numFeatures);
                modifyNearestNeighbours(nearest, nearestValues, dist, i, j, k);
            }
        }

        #pragma omp master 
        printf("Counting most common neighbours\n");

        //Count the nearest neighbour labels
        int *classCounts;

        #pragma omp for
        for(int i = 0; i < numTestPoints; i++){
            classCounts = (int *)calloc(numClasses, sizeof(int));

            for(int j = 0; j < k; j++){
                classCounts[(int)train_data[nearest[i][j]][labelIndex]]++;
            }

            int mostCommon = -1;
            int highestClassCount = -1;
            int tie = 0;

            for(int x = 0; x < numClasses; x++){
                if(classCounts[x] > highestClassCount){
                    mostCommon = x;
                    highestClassCount = classCounts[x];
                    tie = 1; //Reset count if a new highest is found
                }
                else if(classCounts[x] == highestClassCount){
                    tie++;   //Increase count if a new 
                } 
            }

            //If there is more than 1 class with the same highest count, then there is a tie, and we choose the nearest neighbour
            if(tie > 1){
                test_data[i][labelIndex] = train_data[nearest[i][0]][labelIndex];
            }
            else{
                test_data[i][labelIndex] = (double)mostCommon;
            }

            free(classCounts);
        }
        double threadE = omp_get_wtime();

        printf("Thread %d took %f to execute\n", omp_get_thread_num(), threadE - threadS);
    }
    printf("Classified!\n");
    free(nearest);
    free(nearestValues);
    return 0;
}

/**
 * Function shifts nearest neighbours to the right. Sorts the array as it inserts. Easy handling when counting
 * @param nearest: An array of the nearest neighbours for each test point
 * @param nearestValues: An array of the distances of the nearest neighbours from each test point
 * @param dist: The current distance between the current test point and the current train point
 * @param testPoint: The current test point
 * @param trainPoint: The current train point
 * @param k: The number of nearest neighbours 
 *
**/
inline int modifyNearestNeighbours(int **nearest, double **nearestValues, double dist, int testPoint, int trainPoint, int k){
    int inserted = 0;
    int pos = 1;

    //Don't enter this while loop unless it is than the largest value
    while(dist < nearestValues[testPoint][k - 1] && inserted == 0){ 
        if(k - pos == 0){
            nearestValues[testPoint][k - pos] = dist;
            nearest[testPoint][k - pos] = trainPoint;
            inserted = 1;
        }
        else if(dist < nearestValues[testPoint][k - pos - 1]){
            nearestValues[testPoint][k - pos] = nearestValues[testPoint][k - pos - 1];
            nearest[testPoint][k - pos] = nearest[testPoint][k - pos - 1];
            pos++;
        }
        else{
            nearestValues[testPoint][k - pos] = dist;
            nearest[testPoint][k - pos] = trainPoint;
            inserted = 1;
        }
    }
    return 0;
}

/**
 * Finds the squared distance between two points based on the features
 * @param testPoint: The current test point
 * @param trainPoint: The current train point for which we are trying to find the distance from the test point
 * @param train: The train dataset as a 1-D array
 * @param test: The test dataset as a 1-D array
 * @param numFeatures: The number of features shared between both datasets
**/
inline double getDistance(int testPoint, int trainPoint, double *train, double *test, int numFeatures){
    double (*train_data)[numFeatures] = (double(*)[numFeatures]) train;
    double (*test_data)[numFeatures] = (double(*)[numFeatures]) test;
    double sum = 0.0;
    double diff;

    //SIMD won't compile unless sum race condition can be resolved
    #pragma omp simd
    for(int i = 0; i < numFeatures - 1; i++){
        diff = train_data[trainPoint][i] - test_data[testPoint][i];
        sum += diff * diff;
    }
    return sum;
}