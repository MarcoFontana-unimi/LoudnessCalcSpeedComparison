
#include "CudaLoudnessCalc.cuh"
#include "SequentialLoudnessCalc.h"

#define N_OF_ITERATIONS 10

float** elapsedTimes(FILE* fp, char* str, const char* folder)
{

    //calc how many files there are
    int nOfFiles = 0;
    while (EOF != (fscanf(fp, "%*[^\n]"), fscanf(fp, "%*c")))
    {
        nOfFiles++;
    }
    nOfFiles++;
    rewind(fp);


    static float cudaAverageTimes[2] = { 0.0f, 0.0f };
    static float sequentialAverageTimes[2] = { 0.0f, 0.0f };

    printf("\n\nTesting for %i iterations:\n", N_OF_ITERATIONS);

    for (int i = 0; i < N_OF_ITERATIONS; i++)
    {
        printf("\nIteration %i ...", i + 1);
        float* tempSeqTimes = SequentialLoudnessCalc(nOfFiles, fp, str, folder);
        printf("\n  Sequential done");
        float* tempCudaTimes = CudaLoudnessCalc(nOfFiles, fp, str);
        printf("\n  Cuda done\n");

        sequentialAverageTimes[0] += (tempSeqTimes[0] / N_OF_ITERATIONS);
        sequentialAverageTimes[1] += (tempSeqTimes[1] / N_OF_ITERATIONS);

        cudaAverageTimes[0] += (tempCudaTimes[0] / N_OF_ITERATIONS);
        cudaAverageTimes[1] += (tempCudaTimes[1] / N_OF_ITERATIONS);
    }

    fclose(fp);

    static float* times[2] = { sequentialAverageTimes , cudaAverageTimes };

    return times;
}

void elapsedTimesMultipleAudio(char* str, const char* folder)
{
    FILE* fp;

    //open txt with path to files to read
    fp = fopen("AudioFiles/AudioList.txt", "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);

    float** elapsedTimesArray = elapsedTimes(fp, str, folder);

    float seqElapsedClean = elapsedTimesArray[0][0] - elapsedTimesArray[0][1];
    float cudaElapsedClean = elapsedTimesArray[1][0] - elapsedTimesArray[1][1];

    printf("\n\nAverage times for multiple audio files:\n");

    printf("\nTime measured: %.3f seconds.\n", elapsedTimesArray[0][0]);
    printf("Sequential overhead: %.3f seconds.\n", elapsedTimesArray[0][1]);
    printf("Sequential Time without overhead: %.3f seconds.\n\n", seqElapsedClean);

    printf("\nTime measured: %.3f seconds.\n", elapsedTimesArray[1][0]);
    printf("CUDA overhead: %.3f seconds.\n", elapsedTimesArray[1][1]);
    printf("CUDA Time without overhead: %.3f seconds.\n\n", cudaElapsedClean);
}

void elapsedTimesSingleAudio(char* str, const char* folder)
{
    FILE* fp;

    //open txt with path to files to read
    fp = fopen("AudioFiles/LongAudio.txt", "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);

    float** elapsedTimesArray = elapsedTimes(fp, str, folder);

    float seqElapsedClean = elapsedTimesArray[0][0] - elapsedTimesArray[0][1];
    float cudaElapsedClean = elapsedTimesArray[1][0] - elapsedTimesArray[1][1];

    printf("\n\nAverage times for single audio file:\n");

    printf("\nTime measured: %.3f seconds.\n", elapsedTimesArray[0][0]);
    printf("Sequential overhead: %.3f seconds.\n", elapsedTimesArray[0][1]);
    printf("Sequential Time without overhead: %.3f seconds.\n\n", seqElapsedClean);

    printf("\nTime measured: %.3f seconds.\n", elapsedTimesArray[1][0]);
    printf("CUDA overhead: %.3f seconds.\n", elapsedTimesArray[1][1]);
    printf("CUDA Time without overhead: %.3f seconds.\n\n", cudaElapsedClean);
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

    elapsedTimesMultipleAudio(str, folder);
    elapsedTimesSingleAudio(str, folder);

    printf("Press ENTER key to Continue\n");
    getchar();

    return EXIT_SUCCESS;
}