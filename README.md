# XT-NN-CPU-Base
NOTE THAT THIS IS NOT PRODUCTION READY AND SHOULD NEVER BE USED AS ONE. This is only a small implementation of a dynamic neural network library to educate me about how LLMs, image gen and AI in general works.

# Licence
This is provided under the Apache 2.0 licence. Provided as-is and stuff cuz I don't want NSFW stuff to be built with it when I upload GPU finetuning and general GPU training. Just to be in the save side and stuff, you should consiter using llama.cpp or oobabooba webui when you want to do something like that.

# Features
Basically nothing. I'm being honest with you. It's portable and stuff but only at the starting stage.

# Create Model
You can create a model using "Model" like "Model XOR({for example}std::vector<long int> with { 2, 2, 1}, std::string name, float version, activation::method, activation::de-activation-method, float learningRate);"
After that, "prepare your training data in a "vec3" format:
"vec3 trainingData = 
{
  { {input one}, {input two}, {etc} }, //example { {0, 0}, {0, 1}, {1, 0}, {1, 1} }, 
  { {output to input one}, {etc} } //example { {0}, {1}, {1}, {0} }
};"
And use Model::train(traindingData[0], trainingData[1], how much epochs, weight decay, set to -1 if no weight decay, threads, 0 for gd, 1 for adam(adam is currently not working, will be fixed in the future));

Model::save( true for save in binary, false for raw numbers)
Model::load( true for saved in binary, false for saved in raw numbers)
Model::setInput(vec input) //will perform forward prop with the given vector
Model::getOutput() //will return the results of the last setInput vector in a vec1 format.

Feel free to read the .hpp and .cpp files if you want to know more.

# data types
nntype: defined in "config.hpp", can be set to float, double, long double. Int, short, long, etc will compile, but not train correctly.
vec1: std::vector<nntype>. That all.
vec2: std::vector<vec1>. Two dimensional vector.
vec3: std::vector<vec2>. Three dimensional vector.
Model: model.

Compile using: "./compile.sh" and run using "./out.sh". If it says that permission denied, fix it by entering "chmod +x ./out.sh" on linux.

This is not production ready, but everything except the adam functions is optimized. I will clean code up and update when my exams are finished. I will upload a small doc later.
Just saying: Not even one line of the .cpp or .hpp is AI generated! You can validate if you want:)
