#pragma once

#include <cuda_runtime.h>
#include <numeric>
#include <string>
#include <unordered_map>
#include <deque>

class CudaMeasurableEvent
{
public:
    CudaMeasurableEvent(const std::string& name);

    ~CudaMeasurableEvent();

private:
    cudaEvent_t mStart, mStop;
    std::string mName;
};

struct StatisticsSample
{
    std::deque<float> samples;
    float mean, variance;

    void update()
    {
        const std::size_t n = samples.size();
        if (n <= 1)
        {
            return;
        }

        mean = std::accumulate(samples.begin(), samples.end(), 0.0) / n;

        auto varianceC = [this, &n](float accumulator, const float& v)
        {
            return accumulator + ((v - mean) * (v - mean) / (n - 1));
        };

        variance = std::accumulate(samples.begin(), samples.end(), 0.0, varianceC);
    }
};

class Statistics
{
public:
    static Statistics& get()
    {
        static Statistics instance;
        return instance;
    }

    void setMaximumSamplesPerEntry(std::size_t maxSamples)
    {
        mMaxSamplesPerEntry = maxSamples;
    }

    void addSample(const std::string& name, float sample)
    {
        auto& entry = mData[name];
        entry.samples.push_back(sample);
        if (entry.samples.size() > mMaxSamplesPerEntry)
        {
            entry.samples.pop_front();
        }
        entry.update();
    }

    const StatisticsSample* getSample(const std::string& name) const
    {
        auto it = mData.find(name);
        if (it == mData.end())
        {
            return nullptr;
        }
        return &it->second;
    }

private:
    std::unordered_map<std::string, StatisticsSample> mData;


    size_t mMaxSamplesPerEntry = 60;
};

#define CUDA_MEASURABLE_EVENT(name) auto scopeEvent = CudaMeasurableEvent(name)
