#include <string>
#include <vector>

#include "activations.hpp"

class NeuralNetwork
{
private:

    nntype learningRate = (nntype)1E-8;
    nntype weightDecay = (nntype)1E-2;
    nntype (*actvtn)(nntype) = activation::sigmoid;
    nntype (*dActvtn)(nntype) = activation::dSigmoid;

    //ADAM only parameters
    const nntype beta1 = 0.99; //Exponential decay rate for the first moment estimate
    const nntype beta2 = 0.998; //Exponential decay rate for the second moment estimate
    const nntype epsilon = 1.0E-08; //Small constant to prevent division by zero
public:
    size_t layers = 3;

    std::vector<size_t> neurons = { 2, 2, 1 };

    vec2 neuronVal = vec2(0, vec1(0, 0.0f));
    vec2 neuronBias = vec2(0, vec1(0, 0.0f));
    vec3 weight = vec3(0, vec2(0, vec1(0, 0.0f)));

    void initNetwork
    (std::vector<size_t>& nNeurons, nntype(*nActvtn)(nntype x), nntype(*nDActvtn)(nntype x), nntype& nLearningRate);

    void initGPU(bool train);

    void save(std::string& folder, bool binary);
    void load(std::string& folder, bool binary);

    void forwardProp(vec1 input);

    void setWeightDecay(nntype decay);

    void gd(vec1 output);
    void gdWeightDecay(vec1 output);
    void adam(vec1 output);
    void adamWeightDecay(vec1 output);

    void setLearningRate(nntype newlr);

    void appendLayer(size_t nNeurons);

    vec1 getOutput();

    NeuralNetwork& operator=(NeuralNetwork copy)
    {
        if (this != &copy)
        {
            copy.learningRate = learningRate;
            copy.layers = layers;
            copy.actvtn = actvtn;
            copy.dActvtn = dActvtn;
            copy.neurons = neurons;
            copy.neuronVal = neuronVal;
            copy.neuronBias = neuronBias;
            copy.weight = weight;
        }
        return *this;
    }
};
