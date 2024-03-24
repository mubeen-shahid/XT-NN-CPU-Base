#include "xtnn.hpp"
#include <iostream>
#include <fstream>

std::string asciiToString(vec1 input)
{
    std::string output("");
    for (size_t i = 0; i < input.size(); i++) output += (char)input[i];
    return output;
}

vec1 stringToASCII(std::string input)
{
    vec1 output;
    for (size_t i = 0; i < input.size(); i++) output.push_back((short)input[i]);
    return output;
}

int main()
{
    //std::system("title XT-NN-Framework"); <-- only windows command

    nntype lr = 1E-45;
    size_t trainSize = 2;
    size_t in = 178;
    size_t out = 2;

    std::vector<size_t> neurons = { in, (in + out) / 2, ((in + out) / 2 + out) / 2, out};

    Model TestModel("GPLM", (float)0.01, neurons, activation::leakyReLU, activation::dLeakyReLU, lr); //GPLM stands for Generative Pretrained Language Model
    TestModel.save(false);
    
    std::string trainingDataRAW;
    {
        std::ifstream READ("input.txt");
        std::string buffer = "";

        for (size_t i = 0; i < trainSize && (!READ.eof()); i++)
        {
            std::getline(READ, buffer);
            trainingDataRAW += buffer + '\n';
            std::cout << "Read line: " << i << "\n";
        }
        std::cout << "\n\n\n";
    }

    vec3 trainingData(2);
    trainingData[0] = vec2(trainingDataRAW.size(), vec1(neurons[0], 0.0f));
    trainingData[1] = vec2(trainingDataRAW.size(), vec1(neurons[neurons.size() - 1], 0.0f));
    for (size_t inputMax = 1; inputMax < neurons[0]; inputMax++)
    {
        trainingData[0][inputMax] = stringToASCII(trainingDataRAW.substr(0, inputMax));
        std::cout << asciiToString(trainingData[0][inputMax]) << '\n';
        trainingData[1][inputMax][0] = (int)trainingDataRAW[inputMax - 1];
        trainingData[1][inputMax][1] = (int)trainingDataRAW[inputMax];
        std::cout << trainingData[1][inputMax][0] << " : " << trainingData[1][inputMax][1] << "\n\n\n";
    }
    

    TestModel.load(true);
    //std::system("pause"); <-- only windows command.
    TestModel.train(trainingData[0], trainingData[1], (size_t)4 * (size_t)1E15, lr * 1.0E-4, 4, 0);
    TestModel.save(true);
    std::cout << "---Outputs---\n\n";
    for (size_t i = 0; i < trainingData[0].size(); i++)
    {
        std::cout << "Input: \n" << asciiToString(trainingData[0][i]) << ";\n";
        TestModel.setInput(trainingData[0][i]);
        std::cout << "Output: " << (TestModel.getOutput()[0]) << ";\n\n";
    }

    //std::system("pause");

    return 0;
}
