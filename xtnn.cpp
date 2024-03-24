#include <string>
#include <cmath>
#include <chrono>
#include <iostream>
#include <ctime>
#include <thread>
#include <algorithm>

#include "xtnn.hpp"

typedef std::chrono::high_resolution_clock::time_point timePoint;

void Model::train(vec2& input, vec2& output, size_t max_epochs, nntype weightDecay, size_t threads, unsigned short optimizer)
{
    size_t fraction = (size_t)(max_epochs / threads * 1E-3);
    time_t currentTime = std::time(0);
    std::cout << "Started training at " << std::string(std::ctime(&currentTime)) << '\n';

    // Define the chunk size for threading
    size_t chunk_size = input.size() / threads;
    std::vector<std::thread> thread_pool;

    {
        timePoint train_start = std::chrono::high_resolution_clock::now();

        // Function to perform training within a chunk
        auto train_chunk = [&](size_t start, size_t end)
        {
            if (weightDecay == -1)
            {
                if (optimizer == 0)
                {
                    for (size_t currentEpoch = 0; currentEpoch < fraction; currentEpoch++)
                    {
                        for (size_t i = start; i < end; i++)
                        {
                            NN.forwardProp(input[i]);
                            NN.gd(output[i]);
                        }
                    }
                }
                else if (optimizer == 1)
                {
                    for (size_t currentEpoch = 0; currentEpoch < fraction; currentEpoch++)
                    {
                        for (size_t i = start; i < end; i++)
                        {
                            NN.forwardProp(input[i]);
                            NN.adam(output[i]);
                        }
                    }
                }
            }
            else
            {
                NN.setWeightDecay(weightDecay);
                if (optimizer == 0)
                {
                    for (size_t currentEpoch = 0; currentEpoch < fraction; currentEpoch++)
                    {
                        for (size_t i = start; i < end; i++)
                        {
                            NN.forwardProp(input[i]);
                            NN.gdWeightDecay(output[i]);
                        }
                    }
                }
                else if (optimizer == 1)
                {
                    for (size_t currentEpoch = 0; currentEpoch < fraction; currentEpoch++)
                    {
                        for (size_t i = start; i < end; i++)
                        {
                            NN.forwardProp(input[i]);
                            NN.adamWeightDecay(output[i]);
                        }
                    }
                }
            }
        };

        // Create threads and assign chunks of work
        for (size_t i = 0; i < threads - 1; i++) thread_pool.emplace_back(train_chunk, i * chunk_size, (i + 1) * chunk_size);

        // Handle the last chunk which may contain remaining elements
        thread_pool.emplace_back(train_chunk, (threads - 1) * chunk_size, input.size());

        // Join all threads
        for (std::vector<std::thread>::iterator::value_type& thread : thread_pool) thread.join();

        timePoint fraction_end = std::chrono::high_resolution_clock::now();
        size_t time = std::chrono::duration_cast<std::chrono::milliseconds>(fraction_end - train_start).count();
        size_t estimated = time * 1E3 * threads;

        std::cout << fraction << " epochs took " << time << "ms.\nFollowing predictions are not based on AI and are therefore more precise, but are still ((+-)" << time + 1 << "ms).\n"
            << "Estimated: " << estimated << "ms left.\n"
            << "(Estimated: " << estimated / 1000.f << "s left.)\n"
            << "(Estimated: " << (estimated / 1000.f) / 60.f << "m left.)\n"
            << "(Estimated: " << ((estimated / 1000.f) / 60.f) / 60.f << "h left.)\n"
            << "(Estimated: " << (((estimated / 1000.f) / 60.f) / 60.f) / 24.0f << "d left.)\n";
    }

    // Continue training for remaining epochs
    auto continue_training = [&](size_t start, size_t end)
    {
        if (weightDecay == -1)
        {
            if (optimizer == 0)
            {
                for (size_t currentEpoch = fraction; currentEpoch < max_epochs; currentEpoch++)
                {
                    for (size_t i = start; i < end; i++)
                    {
                        NN.forwardProp(input[i]);
                        NN.gd(output[i]);
                    }
                }
            }
            else if (optimizer == 1)
            {
                for (size_t currentEpoch = fraction; currentEpoch < max_epochs; currentEpoch++)
                {
                    for (size_t i = start; i < end; i++)
                    {
                        NN.forwardProp(input[i]);
                        NN.adam(output[i]);
                    }
                }
            }
        }
        else
        {
            if (optimizer == 0)
            {
                for (size_t currentEpoch = fraction; currentEpoch < max_epochs; currentEpoch++)
                {
                    for (size_t i = start; i < end; i++)
                    {
                        NN.forwardProp(input[i]);
                        NN.gdWeightDecay(output[i]);
                    }
                }
            }
            else if (optimizer == 1)
            {
                for (size_t currentEpoch = fraction; currentEpoch < max_epochs; currentEpoch++)
                {
                    for (size_t i = start; i < end; i++)
                    {
                        NN.forwardProp(input[i]);
                        NN.adamWeightDecay(output[i]);
                    }
                }
            }
        }
    };

    // Create threads for remaining epochs
    for (size_t i = 0; i < threads - 1; ++i) thread_pool.emplace_back(continue_training, i * chunk_size, (i + 1) * chunk_size);

    // Handle the last chunk which may contain remaining elements
    thread_pool.emplace_back(continue_training, (threads - 1) * chunk_size, input.size());

    // Join all threads
    for (auto& thread : thread_pool)
        if (thread.joinable()) thread.join();

    currentTime = std::time(0);
    std::cout << "Finished training at " << std::string(std::ctime(&currentTime)) << '\n';
}

void Model::save(bool binary)
{
    std::string str = modelName + "-v" + std::to_string(modelVersion).substr(0, 3);
    NN.save(str, binary);
    return;
}

void Model::load(bool binary)
{
    std::string str = modelName + "-v" + std::to_string(modelVersion).substr(0, 3);
    NN.load(str, binary);
    return;
}

void Model::setInput(vec1 input)
{
    NN.forwardProp(input);
    return;
}

vec1 Model::getOutput() { return NN.getOutput(); }
