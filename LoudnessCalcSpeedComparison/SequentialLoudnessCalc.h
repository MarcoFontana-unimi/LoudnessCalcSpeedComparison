#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "sndfile.h"

#define ABSOLUTE_SILENCE -70.0

#define REFERENCE_LEVEL -16.0

#define REFERENCE_SAMPLING_FREQUENCY 48000.0

//pre-filter coeffs to model a spherical head
float headA[2];
float headB[3];

//high-pass coeffs
float highPassA[2];
float highPassB[3];

float totalWindowLength;

//for a window apply pre filters and find it's sum of all samples
void segmentSquaredByChannel(int channel, float* data, float* out, int segmentIndex, int segmentLength, SF_INFO info)
{
    //Variables to apply the 1st pre-filter
    float pastX0 = 0;
    float pastX1 = 0;

    float pastZ0 = 0;
    float pastZ1 = 0;


    //Variables for the high-pass filter
    float pastZlow0 = 0;
    float pastZlow1 = 0;

    float pastY0 = 0;
    float pastY1 = 0;


    float partialSample = 0;

    for (int s = channel; s < segmentLength; s += info.channels)
    {
        /*
         *
         * “K” frequency weighting for each sample
         *
         */

         //apply the 1st pre-filter to the sample
        float yuleSample = headB[0] * data[s] + headB[1] * pastX0 + headB[2] * pastX1
            - headA[0] * pastZ0 - headA[1] * pastZ1;

        pastX1 = pastX0;
        pastZ1 = pastZ0;
        pastX0 = data[s];
        pastZ0 = yuleSample;

        //apply the high-pass filter to the sample
        float tempsample = highPassB[0] * yuleSample + highPassB[1] * pastZlow0 + highPassB[2] * pastZlow1
            - highPassA[0] * pastY0 - highPassA[1] * pastY1;

        pastZlow1 = pastZlow0;
        pastY1 = pastY0;
        pastZlow0 = yuleSample;
        pastY0 = tempsample;

        partialSample += tempsample * tempsample;

    }

    //memorize the sum of squared samples for the given data
    out[(segmentIndex * info.channels) + channel] = partialSample;

}

//squared mean of a 400ms segment
void windowedSquaredMean(int channel, int segmentIndex, float* squaredSegments, float* out, int nChannels)
{
    out[((segmentIndex - 3) * nChannels) + channel] = (squaredSegments[((segmentIndex - 3) * nChannels) + channel]
        + squaredSegments[((segmentIndex - 2) * nChannels) + channel]
        + squaredSegments[((segmentIndex - 1) * nChannels) + channel]
        + squaredSegments[(segmentIndex * nChannels) + channel]) / totalWindowLength;
}

/*
*
* Sum of window channels weighted as per ITU_R_BS._1770._4 recommendation
*
*/
float weightedSumStereo(int frameIndex, int nChannels, float* squaredMeanByChannel)
{

    float tempSum = 0;

    for (int j = 0; j < nChannels; j++)
        tempSum += squaredMeanByChannel[j + frameIndex];

    return tempSum;
}

float weightedSum3To5Channels(int frameIndex, int nChannels, float* squaredMeanByChannel)
{

    float tempSum = 0;
    int channel = 0;

    for (channel = 0; channel < nChannels - 2; channel++)
        tempSum += squaredMeanByChannel[channel + frameIndex];
    for (channel = nChannels - 2; channel < nChannels; channel++)
        tempSum += squaredMeanByChannel[channel + frameIndex] * 1.41;

    return tempSum;

}

float weightedSumMoreThan5(int frameIndex, int nChannels, float* squaredMeanByChannel)
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

    return tempSum;

}


//calc of the relative gate for the loudness as per ITU_R_BS._1770._4 recommendation
float relativeGateCalc(int nWindows, int nChannels, float* blockLoudness, float* squaredMeanByChannel)
{

    float tempTotLoudness = 0;
    int nonSilenceSegments = 0;
    /*
     * removal of segments below the silence threshold
     */
    for (int i = 0; i < nWindows; i++)
    {
        if (blockLoudness[i] > ABSOLUTE_SILENCE)
        {
            for (int j = 0; j < nChannels; j++)
            {
                tempTotLoudness += squaredMeanByChannel[j + (i * nChannels)];
            }
            nonSilenceSegments++;
        }
    }

    //relative gate as per ITU_R_BS._1770._4 recommendation
    return -0.691 + 10 * log10(tempTotLoudness / nonSilenceSegments) - 10;
}

//calc of the double gated loudness as per ITU_R_BS._1770._4 recommendation
float gatedLoudnessCalc(float relativeGate, int nWindows, int nChannels, float* blockLoudness, float* squaredMeanByChannel)
{

    float tempTotLoudness = 0;
    int aboveGatesSegments = 0;

    /*
     * removal of segments below the relative gate threshold
     */

    if (nChannels > 3 && nChannels < 6)
    {

        for (int i = 0; i < nWindows; i++)
        {
            if (blockLoudness[i] > relativeGate)
            {
                int frameIndex = i * nChannels;
                tempTotLoudness += pow(10, (0.691 + blockLoudness[i]) / 10);

                aboveGatesSegments++;
            }
        }

    }
    else if (nChannels >= 6)
    {

        for (int i = 0; i < nWindows; i++)
        {
            if (blockLoudness[i] > relativeGate)
            {
                int frameIndex = i * nChannels;
                tempTotLoudness += pow(10, (0.691 + blockLoudness[i]) / 10);

                aboveGatesSegments++;
            }
        }

    }
    else
    {

        for (int i = 0; i < nWindows; i++)
        {
            if (blockLoudness[i] > relativeGate)
            {
                int frameIndex = i * nChannels;
                tempTotLoudness += pow(10, (0.691 + blockLoudness[i]) / 10);

                aboveGatesSegments++;
            }
        }

    }

    return -0.691 + 10 * log10(tempTotLoudness / aboveGatesSegments);
}

void calcLoudness(const char* infilename, float* out, int index)
{
    int	readcount;

    SNDFILE* infile;

    SF_INFO	sfinfo;
    memset(&sfinfo, 0, sizeof(sfinfo));

    if (!(infile = sf_open(infilename, SFM_READ, &sfinfo)))
    {
        printf("Not able to open input file %s.\n", infilename);
        puts(sf_strerror(NULL));
    };

    //100 ms window
    int samplesPerWindow = (int)(sfinfo.samplerate * 0.1f * sfinfo.channels);
    float* sampleBuffer = (float*)malloc(sizeof(float) * samplesPerWindow);

    //number of samples in a 400ms window for 1 channel
    totalWindowLength = floor(samplesPerWindow * 4 / (float)sfinfo.channels);

    if (sfinfo.samplerate == 48000)
    {
        //filter coeffs for 48000hz sampling rate (from ITU_R_BS._1770._4 guidelines)
        float headAtemp[2] = { -1.69065929318241, 0.73248077421585 };
        memcpy(headA, headAtemp, sizeof(headA));
        float headBtemp[3] = { 1.53512485958697, -2.69169618940638, 1.19839281085285 };
        memcpy(headB, headBtemp, sizeof(headB));

        float highPassAtemp[2] = { -1.99004745483398, 0.99007225036621 };
        memcpy(highPassA, highPassAtemp, sizeof(highPassA));
        float highPassBtemp[3] = { 1.0, -2.0, 1.0 };
        memcpy(highPassB, highPassBtemp, sizeof(highPassB));
    }
    else if (sfinfo.samplerate == 44100)
    {
        float headAtemp[2] = { -1.6636551132560202, 0.7125954280732254 };
        memcpy(headA, headAtemp, sizeof(headA));
        float headBtemp[3] = { 1.5308412300503478, -2.650979995154729, 1.1690790799215869 };
        memcpy(headB, headBtemp, sizeof(headB));

        float highPassAtemp[2] = { -1.9891696736297957, 0.9891990357870394 };
        memcpy(highPassA, highPassAtemp, sizeof(highPassA));
        float highPassBtemp[3] = { 0.9995600645425144, -1.999120129085029, 0.9995600645425144 };
        memcpy(highPassB, highPassBtemp, sizeof(highPassB));
    }

    int nOfSegments = sfinfo.frames * sfinfo.channels / samplesPerWindow;
    float* squaredSegments = (float*)malloc(sizeof(float) * nOfSegments * sfinfo.channels);

    int segNumber = 0;

    //call segmentSquaredByChannel for each 100ms window
    while ((readcount = (int)sf_read_float(infile, sampleBuffer, samplesPerWindow)) == samplesPerWindow)
    {
        for (int i = 0; i < sfinfo.channels; i++)
        {
            segmentSquaredByChannel(i, sampleBuffer, squaredSegments, segNumber, samplesPerWindow, sfinfo);
        }
        segNumber++;
    }
    free(sampleBuffer);

    int nOfWindows = nOfSegments - 3;
    float* squaredMeanByChannel = (float*)malloc(sizeof(float) * nOfWindows * sfinfo.channels);

    //squared mean of all 400ms window
    for (int i = 3; i < nOfSegments; i++)
    {
        for (int j = 0; j < sfinfo.channels; j++)
        {
            windowedSquaredMean(j, i, squaredSegments, squaredMeanByChannel, sfinfo.channels);
        }
    }
    free(squaredSegments);
    sf_close(infile);

    /*
    *
    * Pre gating loudness of each 400ms window in LUFS as per ITU_R_BS._1770._4 recommendation.
    * this specific implementation will not work well for channel counts above 5.
    *
    */

    //loudness of each 400ms window when all channels are summed

    float* blockLoudness = (float*)malloc(sizeof(float) * nOfWindows);

    if (sfinfo.channels > 3 && sfinfo.channels < 6)
    {
        for (int i = 0; i < nOfWindows; i++)
        {
            int frameIndex = i * sfinfo.channels;
            float tempSum = weightedSum3To5Channels(frameIndex, sfinfo.channels, squaredMeanByChannel);

            blockLoudness[i] = -0.691 + 10 * log10(tempSum);
        }
    }
    else if (sfinfo.channels >= 6)
    {
        for (int i = 0; i < nOfWindows; i++)
        {
            int frameIndex = i * sfinfo.channels;
            float tempSum = weightedSumMoreThan5(frameIndex, sfinfo.channels, squaredMeanByChannel);

            blockLoudness[i] = -0.691 + 10 * log10(tempSum);
        }
    }
    else
    {
        for (int i = 0; i < nOfWindows; i++)
        {
            int frameIndex = i * sfinfo.channels;
            float tempSum = weightedSumStereo(frameIndex, sfinfo.channels, squaredMeanByChannel);

            blockLoudness[i] = -0.691 + 10 * log10(tempSum);
        }
    }

    float relativeGate = relativeGateCalc(nOfWindows, sfinfo.channels, blockLoudness, squaredMeanByChannel);

    float currLoudness = gatedLoudnessCalc(relativeGate, nOfWindows, sfinfo.channels, blockLoudness, squaredMeanByChannel);
    free(blockLoudness);
    free(squaredMeanByChannel);

    out[index] = currLoudness;

}

void sequentialOverheadCalc(const char* infilename, float* out, int index)
{
    int	readcount;

    SNDFILE* infile;

    SF_INFO	sfinfo;
    memset(&sfinfo, 0, sizeof(sfinfo));

    if (!(infile = sf_open(infilename, SFM_READ, &sfinfo)))
    {
        printf("Not able to open input file %s.\n", infilename);
        puts(sf_strerror(NULL));
    };

    //100 ms window
    int samplesPerWindow = (int)(sfinfo.samplerate * 0.1f * sfinfo.channels);
    float* sampleBuffer = (float*)malloc(sizeof(float) * samplesPerWindow);

    //number of samples in a 400ms window for 1 channel
    totalWindowLength = floor(samplesPerWindow * 4 / (float)sfinfo.channels);

    if (sfinfo.samplerate == 48000)
    {
        //filter coeffs for 48000hz sampling rate (from ITU_R_BS._1770._4 guidelines)
        float headAtemp[2] = { -1.69065929318241, 0.73248077421585 };
        memcpy(headA, headAtemp, sizeof(headA));
        float headBtemp[3] = { 1.53512485958697, -2.69169618940638, 1.19839281085285 };
        memcpy(headB, headBtemp, sizeof(headB));

        float highPassAtemp[2] = { -1.99004745483398, 0.99007225036621 };
        memcpy(highPassA, highPassAtemp, sizeof(highPassA));
        float highPassBtemp[3] = { 1.0, -2.0, 1.0 };
        memcpy(highPassB, highPassBtemp, sizeof(highPassB));
    }
    else if (sfinfo.samplerate == 44100)
    {
        //filter coeffs for 44100hz sampling rate precalculated for speed
        float headAtemp[2] = { -1.6636551132560202, 0.7125954280732254 };
        memcpy(headA, headAtemp, sizeof(headA));
        float headBtemp[3] = { 1.5308412300503478, -2.650979995154729, 1.1690790799215869 };
        memcpy(headB, headBtemp, sizeof(headB));

        float highPassAtemp[2] = { -1.9891696736297957, 0.9891990357870394 };
        memcpy(highPassA, highPassAtemp, sizeof(highPassA));
        float highPassBtemp[3] = { 0.9995600645425144, -1.999120129085029, 0.9995600645425144 };
        memcpy(highPassB, highPassBtemp, sizeof(highPassB));
    }

    int nOfSegments = sfinfo.frames * sfinfo.channels / samplesPerWindow;
    //float* squaredSegments = (float*)malloc(sizeof(float) * nOfSegments * sfinfo.channels);

    int segNumber = 0;
    while ((readcount = (int)sf_read_float(infile, sampleBuffer, samplesPerWindow)) == samplesPerWindow)
    {
        for (int i = 0; i < sfinfo.channels; i++)
        {
            //segmentSquaredByChannel(i, sampleBuffer, squaredSegments, segNumber, samplesPerWindow, sfinfo);
        }
        segNumber++;
    }
    free(sampleBuffer);
    sf_close(infile);

}

float* SequentialLoudnessCalc(int nOfFiles, FILE* fp, char* str, const char* folder)
{

    float* filesLoudness = (float*)malloc(sizeof(float) * nOfFiles);
    int fileNumber = 0;

    clock_t start = clock();
    while (EOF != fscanf(fp, "%[^\n]\n", str))
    {
        char* path = (char*)malloc(strlen(folder) + strlen(str) + 1);
        strcpy(path, folder);
        strcat(path, str);
        calcLoudness(path, filesLoudness, fileNumber);
        free(path);
        fileNumber++;
    }
    rewind(fp);

    clock_t end = clock();
    float elapsed = (float)(end - start) / CLOCKS_PER_SEC;

    
    //test overhead
    
    clock_t startOverhead = clock();

    while (EOF != fscanf(fp, "%[^\n]\n", str))
    {
        char* path = (char*)malloc(strlen(folder) + strlen(str) + 1);
        strcpy(path, folder);
        strcat(path, str);

        sequentialOverheadCalc(path, filesLoudness, fileNumber);
        free(path);
        fileNumber++;
    }
    rewind(fp);

    clock_t endOverhead = clock();
    float elapsedOverhead = (float)(endOverhead - startOverhead) / CLOCKS_PER_SEC;

    free(filesLoudness);

    //return elapsed times
    static float times[2];
    times[0] = elapsed;
    times[1] = elapsedOverhead;

    return times;
}


bool sequentialTestcompliance(int nOfFiles, FILE* fp, char* str, const char* folder, float* expectedResults)
{

    printf("Testing sequential compliance to ITU_R_BS._1770._4 ...    ");

    float* filesLoudness = (float*)malloc(sizeof(float) * nOfFiles);
    int fileNumber = 0;

    while (EOF != fscanf(fp, "%[^\n]\n", str))
    {
        char* path = (char*)malloc(strlen(folder) + strlen(str) + 1);
        strcpy(path, folder);
        strcat(path, str);
        calcLoudness(path, filesLoudness, fileNumber);
        free(path);
        fileNumber++;
    }
    rewind(fp);

    bool isCorrect = true;
    for (int i = 0; i < nOfFiles; i++)
    {
        if (fabs(filesLoudness[i] - expectedResults[i]) > 0.1f)
        {
            isCorrect = false;
        }
    }

    printf("%s", isCorrect ? "OK\n\n" : "Not compliant\n\n");


    free(filesLoudness);

    return isCorrect;

}