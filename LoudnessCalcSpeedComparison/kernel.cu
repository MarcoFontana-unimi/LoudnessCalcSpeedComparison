
#include "CudaLoudnessCalc.cuh"
#include "SequentialLoudnessCalc.h"

#define N_OF_ITERATIONS 1

void elapsedTimes(char* str, const char* folder)
{
    FILE* fp;

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


    float cudaAverageTimes[2] = { 0.0f, 0.0f };
    float sequentialAverageTimes[2] = { 0.0f, 0.0f };

    for (int i = 0; i < N_OF_ITERATIONS; i++)
    {
        float* tempSeqTimes = SequentialLoudnessCalc(nOfFiles, fp, str, folder);
        float* tempCudaTimes = CudaLoudnessCalc(nOfFiles, fp, str);

        sequentialAverageTimes[0] += (tempSeqTimes[0] / N_OF_ITERATIONS);
        sequentialAverageTimes[1] += (tempSeqTimes[1] / N_OF_ITERATIONS);

        cudaAverageTimes[0] += (tempCudaTimes[0] / N_OF_ITERATIONS);
        cudaAverageTimes[1] += (tempCudaTimes[1] / N_OF_ITERATIONS);
    }

    float seqElapsedClean = sequentialAverageTimes[0] - sequentialAverageTimes[1];
    float cudaElapsedClean = cudaAverageTimes[0] - cudaAverageTimes[1];

    printf("\n\nAverage times:\n");

    printf("\nTime measured: %.3f seconds.\n", sequentialAverageTimes[0]);
    printf("Sequential overhead: %.3f seconds.\n", sequentialAverageTimes[1]);
    printf("Sequential Time without overhead: %.3f seconds.\n\n", seqElapsedClean);

    printf("\nTime measured: %.3f seconds.\n", cudaAverageTimes[0]);
    printf("CUDA overhead: %.3f seconds.\n", cudaAverageTimes[1]);
    printf("CUDA Time without overhead: %.3f seconds.\n\n", cudaElapsedClean);

    fclose(fp);
}

void testCompliance(char* str, const char* folder)
{
    FILE* audioTestList;

    //open txt with path to files to read
    audioTestList = fopen("AudioFiles/TestAudio.txt", "r");
    if (audioTestList == NULL)
        exit(EXIT_FAILURE);

    //calc how many files there are
    int nOfTestFiles = 0;
    while (EOF != (fscanf(audioTestList, "%*[^\n]"), fscanf(audioTestList, "%*c")))
    {
        nOfTestFiles++;
    }
    nOfTestFiles++;
    rewind(audioTestList);

    float expectedResults[9] = { -23.0f, -33.0f, -23.0f, -23.0f, -23.0f, -23.0f, -23.0f, -23.0f, -23.0f };
    sequentialTestcompliance(nOfTestFiles, audioTestList, str, folder, expectedResults);
    cudaTestcompliance(nOfTestFiles, audioTestList, str, expectedResults);

    fclose(audioTestList);
}


int main(void)
{

    const char* folder = "AudioFiles/";
    char str[400];

    testCompliance(str, folder);

    elapsedTimes(str, folder);

    return EXIT_SUCCESS;
}