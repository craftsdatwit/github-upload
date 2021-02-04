#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "mpi.h"
#include <math.h>
#include <omp.h>

int eDistance(int point1[6], int point2[6], int dims);

int main(int argc, char **argv)
{

    //Usual MPI init
    MPI_Init(&argc, &argv);
    int rank, numranks;
    MPI_Comm_size(MPI_COMM_WORLD, &numranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status stat;
    double starttime;

    //Setting up variables
    int w = 6;
    int nrowTest = 10000;
    int nrowTrain = 200001;
    int testWidth = 6;
    int testHeight = 10000;
    int trainWidth = 6;
    int trainHeight = 200001;

    //Create & Allocate space for matricies using 1d pointers
    int *testData = (int*)malloc(testWidth*testHeight*sizeof(int));
    int *trainData = (int*)malloc(trainWidth*trainHeight*sizeof(int));
    int *tempTrain = (int*)malloc(trainWidth*sizeof(int));
    int *tempTest = (int*)malloc(testWidth*sizeof(int));
    int *sendcounts = (int*)malloc(numranks*sizeof(int));
    int *myTestRows = (int*)malloc(numranks*sizeof(int));
    int *myStartRow = (int*)malloc(numranks*sizeof(int));
    int *displs = (int*)malloc(numranks*sizeof(int));
    int *myPredicted = (int*)malloc(testHeight*sizeof(int));
    int *predicted = (int*)malloc(nrowTest*sizeof(int));
    double *class0 = (double*)malloc(2*sizeof(double));
    double *class1 = (double*)malloc(2*sizeof(double));
    
    //START READ IN
    char buf[1024];
    int row_count = 0;
    int field_count = 0;
    int x = -1;
    int y = -1;

    //Read in training Data
    FILE *fp1 = fopen("./traindata.csv", "r");
    if (!fp1)
    {
        printf("Cannot open file\n");
        return 1;
    }
    while (fgets(buf, 1024, fp1))
    {
        field_count = 0;
        row_count++;
        x++;

        char *field = strtok(buf, ","); //Delimiter

        while (field) //While next
        {
            y++;
            if (y == 6){ y = 0; } //Resets j index
                
            *(trainData + x +y) = atoi(field);
            //trainData[i][j] = atoi(field); //Gets int from char at index field
            //printf("%d,", trainData[i][j]);
            field = strtok(NULL, ",");
            field_count++;
        }
        //printf("\n");
    }
    fclose(fp1);

    if(rank == 0)
    {
        
        for(int i = 0; i < numranks; i++)
        {
            //Calculates the total amount of data for testData & how many rows per rank with their start indexes
            int myTest = (testHeight*testWidth)/numranks;
            int myStart = i*myTest;
            int myEnd = myStart+myTest;
            int rows = testHeight/numranks;
            int startRow = i*rows;
            int endRow = startRow+rows;
            if(i == numranks-1)
            {
                myEnd = testHeight*testWidth;
                endRow = testHeight;
            }
            int totalRows = endRow - startRow;
            int myTestSize = myEnd-myStart;
            displs[i] = myStart;
            sendcounts[i] = myTestSize;
            myTestRows[i] = totalRows;
            myStartRow[i] = startRow;
            //printf("Rank: %d, myStart: %d myEnd: %d\n",i,myStart,myEnd);

            
        }

        //Reset Variables
        row_count = 0;
        field_count = 0;
        int i = -1;
        int j = -1;

        //Read in testing Data
        FILE *fp2 = fopen("./testdata1.csv", "r");
        if (!fp2)
        {
            printf("Cannot open file\n");
            return 1;
        }

        while (fgets(buf, 1024, fp2))
        {
            field_count = 0;
            row_count++;
            i++;
            char *field = strtok(buf, ","); //Delimiter

            while (field) //While next
            {
                j++;
                if (j == 6){ j = 0; } //Resets j

                
                *(testData +i+j) = atoi(field);

                //testData[i][j] = atoi(field); //Gets int from char at index field
                //printf("%d,", testData[i][j]);
                field = strtok(NULL, ",");
                field_count++;
            }
            //printf("\n");
        }
        fclose(fp2);
        //END READING 


    } //end if rank 0

    //Give everyone their start and end.
    MPI_Bcast(sendcounts,numranks,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(displs,numranks,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(myTestRows,numranks,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(myStartRow,numranks,MPI_INT,0,MPI_COMM_WORLD);
    
    //MPI_Bcast(trainData,trainHeight*trainWidth,MPI_INT,0,MPI_COMM_WORLD);
    int *myTestData = (int*)malloc(sendcounts[rank]*sizeof(int));


    //Scatter the test matrix to each myTest matrix
    MPI_Scatterv(testData,sendcounts,displs,MPI_INT,myTestData,sendcounts[rank],MPI_INT,0,MPI_COMM_WORLD);



#pragma omp parallel default(shared)
    {

        //Allocate space for matricies and arrays
        
            //START KNN

            int dims = 6; //Dimensions

            int distance; //temp int for distance
            int class; //temp in for class

            int mins[3][3]; //Stores the 3 lowest values KNN: 3
            int max = 10000; //Max int value

            int class_0; //Class 0 total
            int class_1; //Class 1 total
            
            
#pragma omp for nowait schedule(dynamic,1)
            for (int i = 0; i < myTestRows[rank]; i++){ //For loop over number of rows of test data
                //printf("TestRows[rank] %d, %d\n",rank, myTestRows[rank]);
                //Resetting Variables
                mins[0][0] = max;
                mins[1][0] = max;
                mins[2][0] = max;
                class_0 = 0;
                class_1 = 0;
                
                //Puts row from mytest data into temp test data
                for(int z = 0; z < testWidth; z++)
                {
                    tempTest[z] = *(myTestData+i+z);
                }
               
                
                for (int j = 0; j < 200000; j++){ //For loop over num rows of training data
                    
                    //Puts row from trainData into temp train data
                    for(int h = 0; h < trainWidth; h++)
                    {
                        tempTrain[h] = *(trainData+j+h);
                    }

                    distance = eDistance(tempTest, tempTrain, dims); //Gets euclidian distance
                    
                    class = *(trainData+j+0);
                    //class = trainData[j][0]; //Get's class at given index

                    //printf("Result: %d, Class: %d\n", result[j][0], result[j][1]);

                    //FINDING 3 LOWEST VALUES KNN: 3
                    if (distance < mins[0][0]){

                        mins[2][0] = mins[1][0]; //Distance
                        mins[2][1] = mins[1][1]; //Class

                        mins[1][0] =  mins[0][0]; //Distance
                        mins[1][1] =  mins[0][1]; //Class

                        mins[0][0] = distance; //Distance
                        mins[0][1] = class; //Class

                    
                    }else if (distance < mins[1][0]){

                        mins[2][0] = mins[1][0]; //Distance
                        mins[2][1] = mins[1][1]; //Class

                        mins[1][0] = distance; //Distance
                        mins[1][1] = class; //Class

                    }else if(distance < mins[2][0]){

                        mins[2][0] = distance; //Distance
                        mins[2][1] = class; //Class
                    }

                }

                /**
                printf("Minumum 0 -> Distance: %d | Class: %d \n", mins[0][0], mins[0][1]);
                printf("Minumum 1 -> Distance: %d | Class: %d \n", mins[1][0], mins[1][1]);
                printf("Minumum 2 -> Distance: %d | Class: %d \n", mins[2][0], mins[2][1]);
                printf("\n \n");
                **/

                //IF BLOCK that will do sumation for total of each class for majority vote
                if (mins[0][1] == 0){  //Min 0 Class 0
                    class_0++;
                }if (mins[1][1] == 0){ //Min 1 Class 0
                    class_0++;
                }if (mins[2][1] == 0){ //Min 2 Class 0
                    class_0++;
                }if (mins[0][1] == 1){ //Min 0 Class 1
                    class_1++;
                }if (mins[1][1] == 1){ //Min 1 Class 1
                    class_1++;
                }if (mins[2][1] == 1){ //Min 2 Class 1
                    class_1++;
                }

                
                if (class_0 > class_1){ //Prints if majority vote ruled class 0
                    myPredicted[i] = 0;
                    printf("Rank %d: Predicted class = 0 | Actual Class = %d\n", rank, *(myTestData + i +0));
                }
                if (class_1 > class_0){ //Prints if majority vote ruled class 1
                    myPredicted[i] = 1;
                    printf("Rank %d: Predicted class = 1 | Actual Class = %d\n", rank, *(myTestData + i +0));
                }


            }//END FOR
            
    }//end omp parrallel

    MPI_Gatherv(myPredicted, myTestRows[rank], MPI_INT, predicted,myTestRows,myStartRow,MPI_INT, 0, MPI_COMM_WORLD);

    if(rank == 0)
    {
        //Make sure that each class starts off with 0's
        for(int i = 0; i < 2; i++)
        {
            class0[i] = 0.0;
            class1[i] = 0.0;
        }

        //Calculate Confusion Matrix
        for(int i = 0; i < nrowTest; i++)
        {
            //Class 1
            if(*(testData + i +0) == 0)
            {
                if(predicted[i] == 0)
                {
                    class0[0] = class0[0] + 1.0;
                }
                if(predicted[i] == 1)
                {
                    class1[0] = class1[0] + 1.0;
                }
            }
         
            //Class 2
            if(*(testData + i +0) == 1)
            {   
                if(predicted[i] == 0)
                {
                    class0[1] = class0[1] + 1;
                }
                if(predicted[i] == 1)
                {
                    class1[1] = class1[1] + 1;
                }
            }
            

        }

        double class0acc = class0[0]/(class0[0] + class0[1]);
        double class1acc = class1[0]/(class1[0] + class1[1]);
        double diagonal = class0[0] + class1[1];
        double sumTotal = nrowTest;

        double totalAccuracy = diagonal/sumTotal;
        double error = (sumTotal - diagonal)/sumTotal;

        printf("Total Accuracy: %.5f%c\n",(totalAccuracy*100.0),'%');
        printf("Class 0 Accuracy: %.5f%c\n",(class0acc*100.0),'%');
        printf("Class 1 Accuracy: %.5f%c\n",(class1acc*100.0),'%');
        printf("Total Error Rate: %.5f%c\n",(error*100.0),'%');

        printf("Confusion Matrix: \n");
        printf("%.0f\t %.0f\n",class0[0],class0[1]);
        printf("%.0f\t\t %.0f\n",class1[0],class1[1]);

        
    }//end if rank == 0

    free(testData);
    free(trainData);
    free(tempTrain);
    free(tempTest);
    free(sendcounts);
    free(displs);
    free(myTestData);
    free(myPredicted);
    free(predicted);
    free(class0);
    free(class1);
    free(myTestRows);
    free(myStartRow);

    MPI_Finalize();

    return 0;

}//end main


int eDistance(int point1[6], int point2[6], int dims){ //Euclidian Distance Function

    int d = 0;

    for (int i=1; i< dims; i++){      

        d = d + ( (point1[i] - point2[i]) * (point1[i] - point2[i]) );

    }

    d=sqrt(d);

    return d;
    }
