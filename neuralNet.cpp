#include <ctime>
#include <cmath>
#include <fstream>
#include <iostream>
#include "neuralNet.hpp"

void NeuralNetwork::initNetwork
(std::vector<size_t>& nNeurons, nntype(*nActvtn)(nntype x), nntype(*nDActvtn)(nntype x), nntype& nLearningRate)
{
    //Start: Set Tag Value's, Reserve Memory for Vectors;
    learningRate = nLearningRate;
    neurons = nNeurons;
    actvtn = nActvtn;
    dActvtn = nDActvtn;

    layers = nNeurons.size();
    //End: Set Tag Value's;

    neuronVal = vec2(layers);
    neuronBias = vec2(layers);
    weight = vec3(layers - 1);

    for (size_t i(0); i < layers - 1; i++)
    {
        neuronVal[i] = vec1(neurons[i], 0.0f);
        weight[i] = vec2(neurons[i], vec1(neurons[i + 1], 0.0f));
    }
    for (size_t i(1); i < layers - 1; i++) neuronBias[i] = vec1(neurons[i], 0.001f);

    neuronVal[layers - 1] = vec1(neurons[layers - 1], 0.0f);
    //End: Reserve Memory for Vectors;

    //Start: Set Weight Values to random;
    srand(static_cast<unsigned>(time(0)));
    for (size_t currentLayer(0); currentLayer < layers - 1; currentLayer++)
    {
        for (size_t currentNeuron(0); currentNeuron < neurons[currentLayer]; currentNeuron++)
        {
            for (size_t currentNeuronNL(0); currentNeuronNL < neurons[currentLayer + 1]; currentNeuronNL++)
            {
                weight[currentLayer][currentNeuron][currentNeuronNL] =
                    static_cast<nntype>(rand()) / static_cast<nntype>(RAND_MAX);
            }
        }
    }
    //End: Set Weight Values to random;

    return;
}

void NeuralNetwork::save(std::string& folder, bool binary)
{
    std::system(std::string("mkdir " + folder).c_str());

    if (binary)
    {
        std::ofstream WRITE_BIAS(std::string(folder + "/biases.xtai").c_str(), std::ios::binary | std::ios::out);
        std::ofstream WRITE_WEIGHTS(std::string(folder + "/weights.xtai").c_str(), std::ios::binary | std::ios::out);

        for (size_t currentLayer(1); currentLayer < layers - 1; currentLayer++)
        {
            for (size_t currentNeuron(0); currentNeuron < neurons[currentLayer]; currentNeuron++)
            {
                WRITE_BIAS.write((char*)&neuronBias[currentLayer][currentNeuron], sizeof(nntype));

                for (size_t currentNeuronNL(0); currentNeuronNL < neurons[currentLayer + 1]; currentNeuronNL++)
                    WRITE_WEIGHTS.write((char*)&weight[currentLayer][currentNeuron][currentNeuronNL], sizeof(nntype));
            }
        }
        WRITE_BIAS.close();

        for (size_t currentNeuron(0); currentNeuron < neurons[0]; currentNeuron++)
        {
            for (size_t currentNeuronNL(0); currentNeuronNL < neurons[1]; currentNeuronNL++)
                WRITE_WEIGHTS.write((char*)&weight[0][currentNeuron][currentNeuronNL], sizeof(nntype));
        }
        WRITE_WEIGHTS.close();
    }
    else
    {
        std::ofstream WRITE_BIAS(std::string(folder + "/biases.xtai").c_str(), std::ios::out);
        std::ofstream WRITE_WEIGHTS(std::string(folder + "/weights.xtai").c_str(), std::ios::out);

        for (size_t currentLayer(1); currentLayer < layers - 1; currentLayer++)
        {
            for (size_t currentNeuron(0); currentNeuron < neurons[currentLayer]; currentNeuron++)
            {
                WRITE_BIAS << neuronBias[currentLayer][currentNeuron] << '\n';

                for (size_t currentNeuronNL(0); currentNeuronNL < neurons[currentLayer + 1]; currentNeuronNL++)
                    WRITE_WEIGHTS << weight[currentLayer][currentNeuron][currentNeuronNL] << '\n';
            }
        }
        WRITE_BIAS.close();

        for (size_t currentNeuron(0); currentNeuron < neurons[0]; currentNeuron++)
        {
            for (size_t currentNeuronNL(0); currentNeuronNL < neurons[1]; currentNeuronNL++)
                WRITE_WEIGHTS << weight[0][currentNeuron][currentNeuronNL] << '\n';
        }
        WRITE_WEIGHTS.close();
    }

    return;
}

void NeuralNetwork::load(std::string& folder, bool binary)
{
    std::ifstream READ_BIAS(std::string(folder + "\\biases.xtai").c_str(), std::ios::in);
    std::ifstream READ_WEIGHTS(std::string(folder + "\\weights.xtai").c_str(), std::ios::in);

    if (READ_BIAS.is_open() && READ_WEIGHTS.is_open() && binary)
    {
        READ_BIAS.close();
        READ_WEIGHTS.close();
        READ_BIAS = std::ifstream(std::string(folder + "\\biases.xtai").c_str(), std::ios::binary | std::ios::in);
        READ_WEIGHTS = std::ifstream(std::string(folder + "\\weights.xtai").c_str(), std::ios::binary | std::ios::in);

        for (size_t currentLayer(1); currentLayer < layers - 1; currentLayer++)
        {
            READ_BIAS.read((char*)neuronBias[currentLayer].data(), neuronBias[currentLayer].size() * sizeof(nntype));
            for (size_t currentNeuron(0); currentNeuron < neurons[currentLayer]; currentNeuron++)
                READ_WEIGHTS.read((char*)weight[currentLayer][currentNeuron].data(), weight[currentLayer][currentNeuron].size() * sizeof(nntype));
        }
        READ_BIAS.close();

        for (size_t currentNeuron(0); currentNeuron < neurons[0]; currentNeuron++)
            READ_WEIGHTS.read((char*)weight[0][currentNeuron].data(), weight[0][currentNeuron].size() * sizeof(nntype));
        READ_WEIGHTS.close();
    }
    else if (READ_BIAS.is_open() && READ_WEIGHTS.is_open() && (!binary))
    {
        std::string str("");

        for (size_t currentLayer(1); currentLayer < layers - 1; currentLayer++)
        {
            for (size_t currentNeuron(0); currentNeuron < neurons[currentLayer]; currentNeuron++)
            {
                std::getline(READ_BIAS, str);
                neuronBias[currentLayer][currentNeuron] = (nntype)std::stold(str);

                for (size_t currentNeuronNL(0); currentNeuronNL < neurons[currentLayer + 1]; currentNeuronNL++)
                {
                    std::getline(READ_BIAS, str);
                    weight[currentLayer][currentNeuron][currentNeuronNL] = (nntype)std::stold(str);
                }
            }
        }
        READ_BIAS.close();

        for (size_t currentNeuron(0); currentNeuron < neurons[0]; currentNeuron++)
        {
            for (size_t currentNeuronNL(0); currentNeuronNL < neurons[1]; currentNeuronNL++)
            {
                std::getline(READ_WEIGHTS, str);
                weight[0][currentNeuron][currentNeuronNL] = (nntype)std::stold(str);
            }
        }
        READ_WEIGHTS.close();
    }
    else if (!READ_BIAS.is_open() || !READ_WEIGHTS.is_open())
    {
        READ_BIAS.close();
        std::cout << "Error: Either couldn't open \"" << folder << "\\biases.xtai\" or " << folder << "\\weights.xtai\", assuming that the files / folder doesn't exist.\n";
    }
    else std::cout << "Unexpected error occured.\n";
    return;
}

void NeuralNetwork::forwardProp
(vec1 input)
{
    for (size_t currentNeuron(0); currentNeuron < input.size(); currentNeuron++)
        neuronVal[0][currentNeuron] = input[currentNeuron];

    for (size_t currentNeuron(input.size()); currentNeuron < neurons[0]; currentNeuron++)
        neuronVal[0][currentNeuron] = 0.0f;

    size_t tmp[2] = { layers - 1, layers - 2 };

    for (size_t currentLayer(1); currentLayer < tmp[0]; currentLayer++)
    {
        for (size_t currentNeuron(0); currentNeuron < neurons[currentLayer]; currentNeuron++)
        {
            long double tmpI = 0.0f;
            neuronVal[currentLayer][currentNeuron] = 0.0f;

            for (size_t currentNeuronLL(0); currentNeuronLL < neurons[currentLayer - 1]; currentNeuronLL++)
                tmpI += neuronVal[currentLayer - 1][currentNeuronLL] * weight[currentLayer - 1][currentNeuronLL][currentNeuron];

            neuronVal[currentLayer][currentNeuron] = actvtn(tmpI + neuronBias[currentLayer][currentNeuron]);
        }
    }

    for (size_t currentNeuron(0); currentNeuron < neurons[tmp[0]]; currentNeuron++)
    {
        neuronVal[tmp[0]][currentNeuron] = 0.0f;
        long double tmpI(0.0f);
        for (size_t currentNeuronLL(0); currentNeuronLL < neurons[tmp[1]]; currentNeuronLL++)
            tmpI += neuronVal[tmp[1]][currentNeuronLL] * weight[tmp[1]][currentNeuronLL][currentNeuron];
        neuronVal[tmp[0]][currentNeuron] = actvtn(tmpI);
    }
    return;
}

void NeuralNetwork::setWeightDecay(nntype decay)
{
    weightDecay = decay;
    return;
}

void NeuralNetwork::gd
(vec1 output)
{
    vec2 deltaVal(layers, vec1(0, 0.0f));
    for (size_t currentLayer(0); currentLayer < layers; currentLayer++)
        deltaVal[currentLayer] = vec1(neurons[currentLayer], 0.0f);

    size_t tmp[2] = { layers - 1, layers - 2 }; 
    for (size_t currentNeuron(0); currentNeuron < neurons[tmp[0]]; currentNeuron++)
    {
        if (output.size() - 1 < currentNeuron) output.push_back((nntype)0.0f);
        deltaVal[tmp[0]][currentNeuron] =
            (output[currentNeuron] - neuronVal[tmp[0]][currentNeuron]) *
            dActvtn(neuronVal[tmp[0]][currentNeuron]);
    }

    for (size_t currentNeuron(output.size()); currentNeuron < neurons[tmp[0]]; currentNeuron++) output.push_back((nntype)0.0f);

    for (long long int currentLayer(tmp[1]); currentLayer >= 0; currentLayer--)
    {
        for (size_t currentNeuron(0); currentNeuron < neurons[currentLayer]; currentNeuron++)
        {
            long double error(0.0f);
            for (size_t currentNeuronNL(0); currentNeuronNL < neurons[currentLayer + 1]; currentNeuronNL++)
            {
                error += deltaVal[currentLayer + 1][currentNeuronNL] * weight[currentLayer][currentNeuron][currentNeuronNL];
            }
            deltaVal[currentLayer][currentNeuron] = error * dActvtn(neuronVal[currentLayer][currentNeuron]);
        }
    }

    for (size_t currentNeuron(0); currentNeuron < neurons[tmp[0]]; currentNeuron++)
    {
        for (size_t currentNeuronLL(0); currentNeuronLL < neurons[tmp[1]]; currentNeuronLL++)
        {
            weight[tmp[1]][currentNeuronLL][currentNeuron] +=
                neuronVal[tmp[1]][currentNeuronLL] * deltaVal[tmp[0]][currentNeuron] * learningRate;
        }
    }

    for (size_t currentLayer(tmp[1]); currentLayer > 0; currentLayer--)
    {
        for (size_t currentNeuron(0); currentNeuron < neurons[currentLayer]; currentNeuron++)
        {
            neuronBias[currentLayer][currentNeuron] += deltaVal[currentLayer][currentNeuron] * (learningRate / neurons[currentLayer]);
            for (size_t currentNeuronLL(0); currentNeuronLL < neurons[currentLayer - 1]; currentNeuronLL++)
            {
                weight[currentLayer - 1][currentNeuronLL][currentNeuron] +=
                    neuronVal[currentLayer - 1][currentNeuronLL] * deltaVal[currentLayer][currentNeuron] * learningRate;
            }
        }
    }

    return;
}

void NeuralNetwork::gdWeightDecay
(vec1 output)
{
    vec2 deltaVal(layers, vec1(0, 0.0f));
    for (size_t currentLayer(0); currentLayer < layers; currentLayer++)
        deltaVal[currentLayer] = vec1(neurons[currentLayer], 0.0f);

    size_t tmp[2] = { layers - 1, layers - 2 };
    for (size_t currentNeuron(0); currentNeuron < neurons[tmp[0]]; currentNeuron++)
    {
        if (output.size() - 1 < currentNeuron) output.push_back((nntype)0.0f);
        deltaVal[tmp[0]][currentNeuron] =
            (output[currentNeuron] - neuronVal[tmp[0]][currentNeuron]) *
            dActvtn(neuronVal[tmp[0]][currentNeuron]);
    }

    for (long long int currentLayer(tmp[1]); currentLayer >= 0; currentLayer--)
    {
        for (size_t currentNeuron(0); currentNeuron < neurons[currentLayer]; currentNeuron++)
        {
            long double error(0.0f);
            for (size_t currentNeuronNL(0); currentNeuronNL < neurons[currentLayer + 1]; currentNeuronNL++)
            {
                error += deltaVal[currentLayer + 1][currentNeuronNL] * weight[currentLayer][currentNeuron][currentNeuronNL];
            }
            deltaVal[currentLayer][currentNeuron] = error * dActvtn(neuronVal[currentLayer][currentNeuron]);
        }
    }

    for (size_t currentNeuron(0); currentNeuron < neurons[tmp[0]]; currentNeuron++)
    {
        for (size_t currentNeuronLL(0); currentNeuronLL < neurons[tmp[1]]; currentNeuronLL++)
        {
            weight[tmp[1]][currentNeuronLL][currentNeuron] +=
                neuronVal[tmp[1]][currentNeuronLL] * deltaVal[tmp[0]][currentNeuron] * learningRate
                - weight[tmp[1]][currentNeuronLL][currentNeuron] * weightDecay;
        }
    }

    for (size_t currentLayer(tmp[1]); currentLayer > 0; currentLayer--)
    {
        for (size_t currentNeuron(0); currentNeuron < neurons[currentLayer]; currentNeuron++)
        {
            neuronBias[currentLayer][currentNeuron] += deltaVal[currentLayer][currentNeuron] * (learningRate / neurons[currentLayer]);
            for (size_t currentNeuronLL(0); currentNeuronLL < neurons[currentLayer - 1]; currentNeuronLL++)
            {
                weight[currentLayer - 1][currentNeuronLL][currentNeuron] +=
                    neuronVal[currentLayer - 1][currentNeuronLL] * deltaVal[currentLayer][currentNeuron] * learningRate
                    - weight[currentLayer - 1][currentNeuronLL][currentNeuron] * weightDecay;
            }
        }
    }

    return;
}

void NeuralNetwork::adam(vec1 output)
{
    vec3 m(layers, vec2(0, vec1(0, 0.0))); //moment estimates
    vec3 v(layers, vec2(0, vec1(0, 0.0)));
    nntype beta1_t = beta1;
    nntype beta2_t = beta2;

    size_t tmp[2] = { layers - 1, layers - 2 };

    vec2 deltaVal(layers, vec1(0, 0.0f));

    for (size_t currentLayer(0); currentLayer < layers - 1; currentLayer++)
    {
        deltaVal[currentLayer] = vec1(neurons[currentLayer], 0.0f);
        m[currentLayer] = vec2(neurons[currentLayer], vec1(neurons[currentLayer + 1], 0.0f));
        v[currentLayer] = vec2(neurons[currentLayer], vec1(neurons[currentLayer + 1], 0.0f));
    }
    deltaVal[deltaVal.size() - 1] = vec1(neurons[deltaVal.size() - 1], 0.0f);

    for (size_t currentNeuron(0); currentNeuron < neurons[tmp[0]]; currentNeuron++)
    {
        if (output.size() - 1 < currentNeuron) output.push_back((nntype)0.0f);
        deltaVal[tmp[0]][currentNeuron] =
            (output[currentNeuron] - neuronVal[tmp[0]][currentNeuron]) *
            dActvtn(neuronVal[tmp[0]][currentNeuron]);
    }

    for (long long int currentLayer = tmp[1]; currentLayer >= 0; currentLayer--)
    {
        for (size_t currentNeuron = 0; currentNeuron < neurons[currentLayer]; currentNeuron++)
        {
            long double error = 0.0;
            for (size_t currentNeuronNL = 0; currentNeuronNL < neurons[currentLayer + 1]; currentNeuronNL++)
                error += deltaVal[currentLayer + 1][currentNeuronNL] * weight[currentLayer][currentNeuron][currentNeuronNL];
            deltaVal[currentLayer][currentNeuron] = error * dActvtn(neuronVal[currentLayer][currentNeuron]);
        }
    }

    nntype betatmp[4] = { (1 - beta1), (1 - beta2), (1 - beta1_t), (1 - beta2_t) };
    for (long long int currentLayer = tmp[1]; currentLayer >= 0; currentLayer--) //tmp[1] is correct, do not change!
    {
        size_t nxtlayer = currentLayer + 1;
        for (size_t currentNeuron = 0; currentNeuron < neurons[currentLayer]; currentNeuron++)
        {
            for (size_t currentNeuronNL = 0; currentNeuronNL < neurons[nxtlayer]; currentNeuronNL++)
            {
                nntype grad = neuronVal[currentLayer][currentNeuron] * deltaVal[nxtlayer][currentNeuronNL];

                //Update biased
                m[currentLayer][currentNeuron][currentNeuronNL] = beta1 * m[currentLayer][currentNeuron][currentNeuronNL] + betatmp[0] * grad;
                v[currentLayer][currentNeuron][currentNeuronNL] = beta2 * v[currentLayer][currentNeuron][currentNeuronNL] + betatmp[1] * (grad * grad);

                //Correct bias
                nntype m_hat = m[currentLayer][currentNeuron][currentNeuronNL] / betatmp[2];
                nntype v_hat = v[currentLayer][currentNeuron][currentNeuronNL] / betatmp[3];

                //Update weights
                weight[currentLayer][currentNeuron][currentNeuronNL] += (learningRate / (std::sqrt(v_hat) + epsilon)) * m_hat;
            }
        }
    }

    //Update beta1_t and beta2_t for bias correction
    beta1_t *= beta1;
    beta2_t *= beta2;
}

void NeuralNetwork::adamWeightDecay(vec1 output)
{
    vec3 m(layers, vec2(0, vec1(0, 0.0))); //moment estimates
    vec3 v(layers, vec2(0, vec1(0, 0.0)));
    nntype beta1_t = beta1;
    nntype beta2_t = beta2;

    size_t tmp[2] = { layers - 1, layers - 2 };

    vec2 deltaVal(layers, vec1(0, 0.0f));

    for (size_t currentLayer(0); currentLayer < layers - 1; currentLayer++)
    {
        deltaVal[currentLayer] = vec1(neurons[currentLayer], 0.0f);
        m[currentLayer] = vec2(neurons[currentLayer], vec1(neurons[currentLayer + 1], 0.0f));
        v[currentLayer] = vec2(neurons[currentLayer], vec1(neurons[currentLayer + 1], 0.0f));
    }
    deltaVal[deltaVal.size() - 1] = vec1(neurons[deltaVal.size() - 1], 0.0f);

    for (size_t currentNeuron(0); currentNeuron < neurons[tmp[0]]; currentNeuron++)
    {
        if (output.size() - 1 < currentNeuron) output.push_back((nntype)0.0f);
        deltaVal[tmp[0]][currentNeuron] =
            (output[currentNeuron] - neuronVal[tmp[0]][currentNeuron]) *
            dActvtn(neuronVal[tmp[0]][currentNeuron]);
    }

    for (long long int currentLayer = tmp[1]; currentLayer >= 0; currentLayer--)
    {
        for (size_t currentNeuron = 0; currentNeuron < neurons[currentLayer]; currentNeuron++)
        {
            long double error = 0.0;
            for (size_t currentNeuronNL = 0; currentNeuronNL < neurons[currentLayer + 1]; currentNeuronNL++)
                error += deltaVal[currentLayer + 1][currentNeuronNL] * weight[currentLayer][currentNeuron][currentNeuronNL];
            deltaVal[currentLayer][currentNeuron] = error * dActvtn(neuronVal[currentLayer][currentNeuron]);
        }
    }

    nntype betatmp[4] = { (1 - beta1), (1 - beta2), (1 - beta1_t), (1 - beta2_t) };
    for (long long int currentLayer = tmp[1]; currentLayer >= 0; currentLayer--) //tmp[1] is correct, do not change!
    {
        size_t nxtlayer = currentLayer + 1;
        for (size_t currentNeuron = 0; currentNeuron < neurons[currentLayer]; currentNeuron++)
        {
            for (size_t currentNeuronNL = 0; currentNeuronNL < neurons[nxtlayer]; currentNeuronNL++)
            {
                nntype grad = neuronVal[currentLayer][currentNeuron] * deltaVal[nxtlayer][currentNeuronNL] + weightDecay * weight[currentLayer][currentNeuron][currentNeuronNL];

                //Update biased
                m[currentLayer][currentNeuron][currentNeuronNL] = beta1 * m[currentLayer][currentNeuron][currentNeuronNL] + betatmp[0] * grad;
                v[currentLayer][currentNeuron][currentNeuronNL] = beta2 * v[currentLayer][currentNeuron][currentNeuronNL] + betatmp[1] * (grad * grad);

                //Correct bias
                nntype m_hat = m[currentLayer][currentNeuron][currentNeuronNL] / betatmp[2];
                nntype v_hat = v[currentLayer][currentNeuron][currentNeuronNL] / betatmp[3];

                //Update weights
                weight[currentLayer][currentNeuron][currentNeuronNL] += (learningRate / (std::sqrt(v_hat) + epsilon)) * m_hat;
            }
        }
    }

    //Update beta1_t and beta2_t for bias correction
    beta1_t *= beta1;
    beta2_t *= beta2;
}

void NeuralNetwork::setLearningRate(nntype newlr) { learningRate = newlr; return; }

void NeuralNetwork::appendLayer(size_t nNeurons) { std::cout << "appendLayer: undefined function.\n"; return; }

vec1 NeuralNetwork::getOutput() { return neuronVal[layers - 1]; }
