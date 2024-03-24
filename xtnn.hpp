#include "neuralNet.hpp"

class Model
{
private:

    std::string modelName = "unnamed-model";

    float modelVersion = 0.01f;
    nntype learningRate = (nntype)1E-8;

    NeuralNetwork NN;

public:

    Model(std::string name, float version, std::vector<size_t> scale, nntype(*nActvtn)(nntype x), nntype(*nDActvtn)(nntype x), nntype nLearningRate)
    {
        learningRate = nLearningRate;
        NN.initNetwork
        (scale, nActvtn, nDActvtn, learningRate);
        modelName = name;
        modelVersion = version;
        return;
    }

    Model& operator=(Model copy)
    {
        if (this != &copy)
        {
            copy.modelName = modelName;
            copy.modelVersion = modelVersion;
            copy.learningRate = learningRate;
            copy.NN = NN;
        }
        return *this;
    }

    void train(vec2& input, vec2& output, size_t max_epochs, nntype weightDecay, size_t threads, unsigned short optimizer);

    void save(bool binary);
    void load(bool binary);

    void setInput(vec1 input);
    vec1 getOutput();
};