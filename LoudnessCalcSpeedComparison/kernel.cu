
#include "CudaLoudnessCalc.cuh"
#include "SequentialLoudnessCalc.h"


int main(void)
{

    const char* folder = "AudioFiles/";

    FILE* fp;
    char str[400];

    //open txt with path to files to read
    fp = fopen("AudioFiles/AudioList.txt", "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);

    //calc how many files there are
    int nOfFiles = 0;
    while (EOF != (fscanf(fp, "%*[^\n]"), fscanf(fp, "%*c")))
    {
        nOfFiles++;
    }
    nOfFiles++;
    rewind(fp);

    SequentialLoudnessCalc(nOfFiles, fp, str, folder);
    CudaLoudnessCalc(nOfFiles, fp, str);


    return EXIT_SUCCESS;
}