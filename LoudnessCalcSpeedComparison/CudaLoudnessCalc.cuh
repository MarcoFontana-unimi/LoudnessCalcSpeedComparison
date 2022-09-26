
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifdef _WIN32
//  For Windows (32- and 64-bit)
#   include <windows.h>
#   define SLEEP(msecs) Sleep(msecs)
#elif __unix
//  For linux, OSX, and other unixes
#   define _POSIX_C_SOURCE 199309L // or greater
#   include <time.h>
#   define SLEEP(msecs) do {            \
        struct timespec ts;             \
        ts.tv_sec = msecs/1000;         \
        ts.tv_nsec = msecs%1000*1000;   \
        nanosleep(&ts, NULL);           \
        } while (0)
#else
#   error "Unknown system"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <string.h>
#include <pthread.h>
#include "sndfile.h"
#include <time.h>

#define ABSOLUTE_SILENCE -70
#define MAX_BLOCK_DIM 512.0

struct filterCoeffs {
    float headA0 = -1.69065929318241;
    float headA1 = 0.73248077421585;

    float headB0 = 1.53512485958697;
    float headB1 = -2.69169618940638;
    float headB2 = 1.19839281085285;

    float highPassA0 = -1.99004745483398;
    float highPassA1 = 0.99007225036621;

    float highPassB0 = 1.0;
    float highPassB1 = -2.0;
    float highPassB2 = 1.0;
};

struct arg_struct {
    char infilename[400];
    float* audioPointer;
    float* out;
    cudaStream_t* stream;
    int* counter;
    int index;
    int nOfFiles;
};

//for a window apply pre filters and find it's sum of all squared samples
__global__ void segmentSquaredByChannel(float* data, float* out, int segmentLength, struct filterCoeffs coeffs,
    int maxNOfFrames, int channels)
{

    unsigned int channel = ((blockIdx.x * blockDim.x) + threadIdx.x) % channels;
    unsigned int frameIndex = ((blockIdx.x * blockDim.x) + threadIdx.x) / channels;

    if (frameIndex < maxNOfFrames)
    {

        //apply the 2 second order IIR filters
        float inputAcc, outputAcc;
        float buffer1 = 0;
        float buffer2 = 0;

        float inputAccLow, outputAccLow;
        float buffer1Low = 0;
        float buffer2Low = 0;

        float partialSample = 0;

        for (long s = ((long)frameIndex * segmentLength) + channel;
            s < ((long)frameIndex * segmentLength) + segmentLength; s += channels)
        {


            inputAcc = data[s];
            inputAcc = inputAcc - (coeffs.headA0 * buffer1);
            inputAcc = inputAcc - (coeffs.headA1 * buffer2);
            outputAcc = inputAcc * coeffs.headB0;
            outputAcc = outputAcc + (coeffs.headB1 * buffer1);
            outputAcc = outputAcc + (coeffs.headB2 * buffer2);

            buffer2 = buffer1;
            buffer1 = inputAcc;

            inputAccLow = outputAcc;
            inputAccLow = inputAccLow - (coeffs.highPassA0 * buffer1Low);
            inputAccLow = inputAccLow - (coeffs.highPassA1 * buffer2Low);
            outputAccLow = inputAccLow * coeffs.highPassB0;
            outputAccLow = outputAccLow + (coeffs.highPassB1 * buffer1Low);
            outputAccLow = outputAccLow + (coeffs.highPassB2 * buffer2Low);

            buffer2Low = buffer1Low;
            buffer1Low = inputAccLow;

            partialSample += outputAccLow * outputAccLow;

        }

        //memorize the sum of squared samples for the given data

        out[(frameIndex * channels) + channel] = partialSample;
    }

}

//squared mean of a 400ms window
__global__ void windowedSquaredMean(float* squaredSegments, float* out, float totalWindowLength, int channels, int maxNOfFrames)
{
    int channel = ((blockIdx.x * blockDim.x) + threadIdx.x) % channels;
    int frameIndex = ((blockIdx.x * blockDim.x) + threadIdx.x) / channels;

    if (frameIndex < maxNOfFrames)
    {
        out[((frameIndex)*channels) + channel] = (squaredSegments[((frameIndex + 3) * channels) + channel]
            + squaredSegments[((frameIndex + 2) * channels) + channel]
            + squaredSegments[((frameIndex + 1) * channels) + channel]
            + squaredSegments[(frameIndex * channels) + channel]) / totalWindowLength;
    }
}

/*
*
* Sum of window channels weighted as per ITU_R_BS._1770._4 recommendation
*
*/
__global__ void weightedSumStereo(int nChannels, float* squaredMeanByChannel, float* out, int maxNOfBlocks)
{
    int frameIndex = ((blockDim.x * blockIdx.x) + threadIdx.x) * nChannels;
    if (frameIndex / nChannels < maxNOfBlocks)
    {
        float tempSum = 0;

        for (int j = 0; j < nChannels; j++)
            tempSum += squaredMeanByChannel[j + frameIndex];

        out[(blockDim.x * blockIdx.x) + threadIdx.x] = -0.691 + 10 * log10(tempSum);
    }
}

__global__ void weightedSum3To5Channels(int nChannels, float* squaredMeanByChannel, float* out, int maxNOfBlocks)
{
    int frameIndex = ((blockDim.x * blockIdx.x) + threadIdx.x) * nChannels;

    if (frameIndex / nChannels < maxNOfBlocks)
    {
        float tempSum = 0;
        int channel = 0;

        for (channel = 0; channel < nChannels - 2; channel++)
            tempSum += squaredMeanByChannel[channel + frameIndex];
        for (channel = nChannels - 2; channel < nChannels; channel++)
            tempSum += squaredMeanByChannel[channel + frameIndex] * 1.41;

        out[(blockDim.x * blockIdx.x) + threadIdx.x] = -0.691 + 10 * log10(tempSum);
    }

}

__global__ void weightedSumMoreThan5(int nChannels, float* squaredMeanByChannel, float* out, int maxNOfBlocks)
{
    int frameIndex = ((blockDim.x * blockIdx.x) + threadIdx.x) * nChannels;

    if (frameIndex / nChannels < maxNOfBlocks)
    {
        float tempSum = 0;
        int channel = 0;

        for (channel = 0; channel < 3; channel++)
        {
            tempSum += squaredMeanByChannel[channel + frameIndex];
        }
        //skip the LTE channel
        for (channel = 4; channel < 6; channel++)
        {
            tempSum += squaredMeanByChannel[channel + frameIndex] * 1.41;
        }
        for (channel = 6; channel < nChannels; channel++)
        {
            tempSum += squaredMeanByChannel[channel + frameIndex];
        }

        out[(blockDim.x * blockIdx.x) + threadIdx.x] = -0.691 + 10 * log10(tempSum);
    }

}

/*
*
* Parallel reduce implementation with last warp unrolling
*
*/

__device__ void warpUnroll(volatile float* sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void parallelReduce(float* inData, int nArrayElems)
{
    extern __shared__ float sharedData[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    if (i + blockDim.x < nArrayElems)
        sharedData[tid] = inData[i] + inData[i + blockDim.x];
    else if (i < nArrayElems)
        sharedData[tid] = inData[i];
    else
        sharedData[tid] = 0;

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (tid < s)
        {
            sharedData[tid] += sharedData[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) warpUnroll(sharedData, tid);

    if (tid == 0) inData[blockIdx.x] = sharedData[0];

}


//Sum of array after parallel reduce and conversion to LUFS
__device__ float gatingLoudnessArraySum(float* newBlockLoudness, int length, int counter)
{
    float res = 0;

    for (int i = 0; i < length; i++)
    {
        res += newBlockLoudness[i];
    }

    return -0.691 + 10 * log10(res / counter) - 10;
}

//calc of the relative gate for the loudness as per ITU_R_BS._1770._4 recommendation
__global__ void relativeGateCalcInit(int nWindows, int nChannels, float* blockLoudness,
    float* squaredMeanByChannel, float* newBlockLoudness, int* counter, int index)
{
    int blockIndex = ((blockDim.x * blockIdx.x) + threadIdx.x);
    float tempTotLoudness = 0;

    if (blockIndex < nWindows)
    {
        if (blockLoudness[blockIndex] > ABSOLUTE_SILENCE)
        {
            for (int j = 0; j < nChannels; j++)
            {
                tempTotLoudness += squaredMeanByChannel[j + (blockIndex * nChannels)];
            }
            atomicAdd(&counter[index], 1);
        }

        newBlockLoudness[blockIndex] = tempTotLoudness;
    }

}

//calc of the float gated loudness as per ITU_R_BS._1770._4 recommendation
__global__ void gatedLoudnessReductionInit(float* relativeGate, int nWindows, float* blockLoudness, float* newBlockLoudness,
    int* counter, int index)
{
    int blockIndex = ((blockDim.x * blockIdx.x) + threadIdx.x);

    if (blockIndex < nWindows)
    {
        if (blockLoudness[blockIndex] > relativeGate[index])
        {
            newBlockLoudness[blockIndex] = __exp10f((0.691f + blockLoudness[blockIndex]) / 10);
            atomicAdd(&counter[index], 1);
        }
        else
            newBlockLoudness[blockIndex] = 0;
    }
}

//save relative gate value to global mem
__global__ void relativeGate(float* newBlockLoudness, int length,
    float* out, int index, int* counter)
{
    out[index] = gatingLoudnessArraySum(newBlockLoudness, length, counter[index]);
    counter[index] = 0;
}

//save final loudness value in unified mem
__global__ void loudnessToHost(float* blockLoudness, int length,
    float* out, int index, int* counter)
{

    float currLoudness = gatingLoudnessArraySum(blockLoudness, length, counter[index]) + 10;

    out[index] = currLoudness;
}

//launch final set of kernels to calc the file loudness
void retrieveLoudness(int nOfWindows, int channels, int index, float* blockLoudness,
    float* squaredMeanByChannel, float* out, cudaStream_t stream, int* counter)
{
    int nOfBlocks = ceil(nOfWindows / MAX_BLOCK_DIM);
    double blockDim = MAX_BLOCK_DIM;
    float* newBlockLoudness = &blockLoudness[nOfWindows];
    relativeGateCalcInit << <nOfBlocks, blockDim, 0, stream >> > (nOfWindows, channels, blockLoudness, squaredMeanByChannel,
        newBlockLoudness, counter, index);

    //to avoid too many threads that don't do work
    if (nOfWindows % (int)MAX_BLOCK_DIM > MAX_BLOCK_DIM / 2)
    {
        nOfBlocks = ceil(nOfWindows / MAX_BLOCK_DIM * 2);
    }
    else
    {
        blockDim = MAX_BLOCK_DIM / 2;
        nOfBlocks = ceil(nOfWindows / (MAX_BLOCK_DIM));
    }

    parallelReduce << <nOfBlocks, blockDim, sizeof(float)* blockDim, stream >> > (newBlockLoudness, nOfWindows);
    relativeGate << < 1, 1, 0, stream >> > (newBlockLoudness, nOfBlocks, out, index, counter);

    gatedLoudnessReductionInit << < nOfBlocks * 2, blockDim, 0, stream >> > (out, nOfWindows, blockLoudness, newBlockLoudness, counter, index);

    parallelReduce << <nOfBlocks, blockDim, sizeof(float)* blockDim, stream >> > (newBlockLoudness, nOfWindows);

    loudnessToHost << < 1, 1, 0, stream >> > (newBlockLoudness, nOfBlocks, out, index, counter);
}

void calcLoudness(float* audioFile, float* out, int index, SF_INFO	sfinfo,
    struct filterCoeffs coeffs, cudaStream_t stream, int* counter)
{

    //100 ms window
    int samplesPerWindow = (int)(sfinfo.samplerate * 0.1f * sfinfo.channels);

    long nOfSegments = floor(sfinfo.frames * sfinfo.channels / (float)samplesPerWindow);
    float* squaredSegments = &audioFile[sfinfo.frames * sfinfo.channels];
    int blockDim = MAX_BLOCK_DIM;
    int nOfBlocks = ceil((nOfSegments * sfinfo.channels) / MAX_BLOCK_DIM);

    segmentSquaredByChannel << <nOfBlocks, MAX_BLOCK_DIM, 0, stream >> > (audioFile, squaredSegments, samplesPerWindow, coeffs,
        nOfSegments, sfinfo.channels);

    //number of samples in a 400ms window for 1 channel
    float totalWindowLength = floor((samplesPerWindow * 4 / (float)sfinfo.channels));

    //windows of 400ms overlapping by 300 ms
    int nOfWindows = nOfSegments - 3;
    float* squaredMeanByChannel = &squaredSegments[nOfSegments * sfinfo.channels];
    nOfBlocks = ceil((nOfWindows * sfinfo.channels) / MAX_BLOCK_DIM);

    windowedSquaredMean << <nOfBlocks, MAX_BLOCK_DIM, 0, stream >> > (squaredSegments, squaredMeanByChannel, totalWindowLength,
        sfinfo.channels, nOfWindows);

    //loudness of each 400ms window when all channels are summed

    float* blockLoudness = &squaredMeanByChannel[nOfWindows * sfinfo.channels];

    //to avoid too many threads that don't do work
    if (nOfWindows % (int)MAX_BLOCK_DIM > MAX_BLOCK_DIM / 2)
    {
        nOfBlocks = ceil(nOfWindows / MAX_BLOCK_DIM);
    }
    else
    {
        blockDim = MAX_BLOCK_DIM / 2;
        nOfBlocks = ceil(nOfWindows / (MAX_BLOCK_DIM / 2));
    }

    /*
    *
    * Pre gating loudness of each 400ms window in LUFS as per ITU_R_BS._1770._4 recommendation.
    *
    */
    if (sfinfo.channels > 3 && sfinfo.channels < 6)
    {
        weightedSum3To5Channels << <nOfBlocks, blockDim, 0, stream >> > (sfinfo.channels, squaredMeanByChannel,
            blockLoudness, nOfWindows);
    }
    else if (sfinfo.channels >= 6)
    {
        weightedSumMoreThan5 << <nOfBlocks, blockDim, 0, stream >> > (sfinfo.channels, squaredMeanByChannel,
            blockLoudness, nOfWindows);
    }
    else
    {
        weightedSumStereo << <nOfBlocks, blockDim, 0, stream >> > (sfinfo.channels, squaredMeanByChannel,
            blockLoudness, nOfWindows);
    }

    retrieveLoudness(nOfWindows, sfinfo.channels, index, blockLoudness,
        squaredMeanByChannel, out, stream, counter);

}

//cpu threads to read audio file and launch kernels
void* readFileAndLaunchThreads(void* arguments)
{
    const char* folder = "AudioFiles/";
    struct arg_struct* args = (struct arg_struct*)arguments;

    SNDFILE* infile;

    SF_INFO	sfinfo;
    memset(&sfinfo, 0, sizeof(sfinfo));

    struct filterCoeffs coeffs;

    cudaStream_t stream;

    //each file gets it's own non-blocking stream
    stream = args->stream[args->index];

    char* path = (char*)malloc(strlen(folder) + strlen(args->infilename) + 1);
    strcpy(path, folder);
    strcat(path, args->infilename);

    if (!(infile = sf_open(path, SFM_READ, &sfinfo)))
    {
        printf("Not able to open input file %s.\n", path);
        puts(sf_strerror(NULL));
    };

    if (sfinfo.samplerate == 48000)
    {
        //filter coeffs for 48000hz sampling rate (from ITU_R_BS._1770._4 guidelines)
        coeffs.headA0 = -1.69065929318241;
        coeffs.headA1 = 0.73248077421585;

        coeffs.headB0 = 1.53512485958697;
        coeffs.headB1 = -2.69169618940638;
        coeffs.headB2 = 1.19839281085285;

        coeffs.highPassA0 = -1.99004745483398;
        coeffs.highPassA1 = 0.99007225036621;

        coeffs.highPassB0 = 1.0;
        coeffs.highPassB1 = -2.0;
        coeffs.highPassB2 = 1.0;
    }
    else if (sfinfo.samplerate == 44100)
    {
        coeffs.headA0 = -1.6636551132560202;
        coeffs.headA1 = 0.7125954280732254;

        coeffs.headB0 = 1.5308412300503478;
        coeffs.headB1 = -2.650979995154729;
        coeffs.headB2 = 1.1690790799215869;

        coeffs.highPassA0 = -1.9891696736297957;
        coeffs.highPassA1 = 0.9891990357870394;

        coeffs.highPassB0 = 0.9995600645425144;
        coeffs.highPassB1 = -1.999120129085029;
        coeffs.highPassB2 = 0.9995600645425144;
    }

    __int64 audioLength = sfinfo.frames * sfinfo.channels;

    //100 ms window
    int samplesPerWindow = (int)(sfinfo.samplerate * 0.1f * sfinfo.channels);

    int nOfSegments = floor(sfinfo.frames * sfinfo.channels / (float)samplesPerWindow);
    int nOfWindows = nOfSegments - 3;

    size_t bytesToAlloc = (sizeof(float) * audioLength)
        + (sizeof(float) * nOfSegments * sfinfo.channels)
        + (sizeof(float) * nOfWindows * sfinfo.channels)
        + (sizeof(float) * nOfWindows * 2);

    //check if device mem is big enough for this file
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    if (totalMem < bytesToAlloc)
    {
        printf("\n%s is too big... \n", args->infilename);
        return NULL;
    }

    float* audioFileCPU = (float*)malloc(sizeof(float) * audioLength);
    sf_read_float(infile, audioFileCPU, audioLength);

    //wait for device to have enough free memory
    while (freeMem < bytesToAlloc)
    {
        SLEEP(1000);
        cudaMemGetInfo(&freeMem, &totalMem);
    }

    cudaEvent_t copyDone;
    cudaEventCreateWithFlags(&copyDone, cudaEventDisableTiming);

    //call async functions to create a correct sequence inside each stream
    cudaMallocAsync(&args->audioPointer, bytesToAlloc, stream);
    cudaMemcpyAsync(args->audioPointer, audioFileCPU, sizeof(float) * audioLength,
        cudaMemcpyHostToDevice, stream);
    cudaEventRecord(copyDone, stream);

    calcLoudness(args->audioPointer, args->out,
        args->index, sfinfo, coeffs, stream, args->counter);

    cudaFreeAsync(args->audioPointer, stream);

    free(path);
    sf_close(infile);

    //wait until copyAsync is done
    cudaEventSynchronize(copyDone);
    free(audioFileCPU);
    cudaEventDestroy(copyDone);

    return NULL;
}

void* CudaOverheadCalc(void* arguments)
{
    const char* folder = "AudioFiles/";
    struct arg_struct* args = (struct arg_struct*)arguments;

    SNDFILE* infile;

    SF_INFO	sfinfo;
    memset(&sfinfo, 0, sizeof(sfinfo));

    struct filterCoeffs coeffs;

    cudaStream_t stream;

    //each file gets it's own non-blocking stream
    stream = args->stream[args->index];

    char* path = (char*)malloc(strlen(folder) + strlen(args->infilename) + 1);
    strcpy(path, folder);
    strcat(path, args->infilename);

    if (!(infile = sf_open(path, SFM_READ, &sfinfo)))
    {
        printf("Not able to open input file %s.\n", path);
        puts(sf_strerror(NULL));
    };

    if (sfinfo.samplerate == 48000)
    {
        //filter coeffs for 48000hz sampling rate (from ITU_R_BS._1770._4 guidelines)
        coeffs.headA0 = -1.69065929318241;
        coeffs.headA1 = 0.73248077421585;

        coeffs.headB0 = 1.53512485958697;
        coeffs.headB1 = -2.69169618940638;
        coeffs.headB2 = 1.19839281085285;

        coeffs.highPassA0 = -1.99004745483398;
        coeffs.highPassA1 = 0.99007225036621;

        coeffs.highPassB0 = 1.0;
        coeffs.highPassB1 = -2.0;
        coeffs.highPassB2 = 1.0;
    }
    else if (sfinfo.samplerate == 44100)
    {
        coeffs.headA0 = -1.6636551132560202;
        coeffs.headA1 = 0.7125954280732254;

        coeffs.headB0 = 1.5308412300503478;
        coeffs.headB1 = -2.650979995154729;
        coeffs.headB2 = 1.1690790799215869;

        coeffs.highPassA0 = -1.9891696736297957;
        coeffs.highPassA1 = 0.9891990357870394;

        coeffs.highPassB0 = 0.9995600645425144;
        coeffs.highPassB1 = -1.999120129085029;
        coeffs.highPassB2 = 0.9995600645425144;
    }

    __int64 audioLength = sfinfo.frames * sfinfo.channels;

    //100 ms window
    int samplesPerWindow = (int)(sfinfo.samplerate * 0.1f * sfinfo.channels);

    int nOfSegments = floor(sfinfo.frames * sfinfo.channels / (float)samplesPerWindow);
    int nOfWindows = nOfSegments - 3;

    size_t bytesToAlloc = (sizeof(float) * audioLength)
        + (sizeof(float) * nOfSegments * sfinfo.channels)
        + (sizeof(float) * nOfWindows * sfinfo.channels)
        + (sizeof(float) * nOfWindows * 2);

    //check if device mem is big enough for this file
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    if (totalMem < bytesToAlloc)
    {
        printf("\n%s is too big... \n", args->infilename);
        return NULL;
    }

    float* audioFileCPU = (float*)malloc(sizeof(float) * audioLength);
    sf_read_float(infile, audioFileCPU, audioLength);

    //wait for device to have enough free memory
    /*while (freeMem < bytesToAlloc)
    {
        SLEEP(1000);
        cudaMemGetInfo(&freeMem, &totalMem);
    }

    cudaEvent_t copyDone;
    cudaEventCreateWithFlags(&copyDone, cudaEventDisableTiming);

    //call async functions to create a correct sequence inside each stream
    cudaMallocAsync(&args->audioPointer, bytesToAlloc, stream);
    cudaMemcpyAsync(args->audioPointer, audioFileCPU, sizeof(float) * audioLength,
        cudaMemcpyHostToDevice, stream);
    cudaEventRecord(copyDone, stream);

    calcLoudness(args->audioPointer, args->out,
        args->index, sfinfo, coeffs, stream, args->counter);

    cudaFreeAsync(args->audioPointer, stream);*/

    free(path);
    sf_close(infile);

    //wait until copyAsync is done
    //cudaEventSynchronize(copyDone);
    free(audioFileCPU);
    //cudaEventDestroy(copyDone);

    return NULL;
}

void CudaLoudnessCalc(int nOfFiles, FILE* fp, char* str)
{
    //one cpu thread for each file
    pthread_t* ThHandle = (pthread_t*)malloc(sizeof(pthread_t) * nOfFiles);
    int ThErr;

    //array with final results
    float* filesLoudness;
    cudaMallocManaged(&filesLoudness, sizeof(float) * nOfFiles);

    //helper array for some kernels
    int* counters;
    cudaMalloc(&counters, sizeof(int) * nOfFiles);

    //init non blocking streams for each file
    cudaStream_t* stream = (cudaStream_t*)malloc(sizeof(cudaStream_t) * nOfFiles);
    for (int i = 0; i < nOfFiles; i++)
    {
        cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);
    }


    float** audioFilePointers = (float**)malloc(sizeof(float*) * nOfFiles);

    printf("Analyzing with CUDA please wait...\n");

    //args for each cpu thread
    struct arg_struct* args = (arg_struct*)malloc(sizeof(arg_struct) * nOfFiles);

    int fileIndex = 0;

    clock_t start = clock();

    while (EOF != fscanf(fp, "%[^\n]\n", str))
    {
        args[fileIndex].audioPointer = audioFilePointers[fileIndex];
        args[fileIndex].out = filesLoudness;
        strcpy(args[fileIndex].infilename, str);
        args[fileIndex].counter = counters;
        args[fileIndex].index = fileIndex;
        args[fileIndex].stream = stream;
        args[fileIndex].nOfFiles = nOfFiles;

        ThErr = pthread_create(&ThHandle[fileIndex], NULL, readFileAndLaunchThreads, (void*)&args[fileIndex]);
        if (ThErr != 0) {
            printf("\nThread Creation Error %d. Exiting abruptly... \n", ThErr);
            exit(EXIT_FAILURE);
        }

        fileIndex++;
    }

    //wait for all cpu threads to complete
    for (int i = 0; i < nOfFiles; i++) {
        pthread_join(ThHandle[i], NULL);
    }
    rewind(fp);

    //wait for all kernels to complete
    cudaDeviceSynchronize();
    clock_t end = clock();

    //cleanup memory
    cudaFree(counters);
    for (int i = 0; i < nOfFiles; i++)
    {
        cudaStreamDestroy(stream[i]);
    }

    //print results
    /*int counter = 0;
    while (EOF != fscanf(fp, "%[^\n]\n", str))
    {
        printf("%s: %f \n", str, filesLoudness[counter]);
        counter++;
    }*/

    double elapsed = double(end - start) / CLOCKS_PER_SEC;


    printf("Measuring file read overhead for CUDA...\n");

    fileIndex = 0;

    clock_t startOverhead = clock();

    while (EOF != fscanf(fp, "%[^\n]\n", str))
    {
        args[fileIndex].audioPointer = audioFilePointers[fileIndex];
        args[fileIndex].out = filesLoudness;
        strcpy(args[fileIndex].infilename, str);
        args[fileIndex].counter = counters;
        args[fileIndex].index = fileIndex;
        args[fileIndex].stream = stream;
        args[fileIndex].nOfFiles = nOfFiles;

        ThErr = pthread_create(&ThHandle[fileIndex], NULL, CudaOverheadCalc, (void*)&args[fileIndex]);
        if (ThErr != 0) {
            printf("\nThread Creation Error %d. Exiting abruptly... \n", ThErr);
            exit(EXIT_FAILURE);
        }

        fileIndex++;
    }

    //wait for all cpu threads to complete
    for (int i = 0; i < nOfFiles; i++) {
        pthread_join(ThHandle[i], NULL);
    }
    rewind(fp);

    clock_t endOverhead = clock();

    float elapsedOverhead = (float)(endOverhead - startOverhead) / CLOCKS_PER_SEC;
    printf("\nTime measured: %.3f seconds.\n", elapsed);
    printf("CUDA overhead: %.3f seconds.\n", elapsedOverhead);

    float elapsedClean = elapsed - elapsedOverhead;

    printf("CUDA Time without overhead: %.3f seconds.\n\n", elapsedClean);

    free(audioFilePointers);
    free(ThHandle);
}