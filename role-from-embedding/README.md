# prediction-model
This contains some model to predict job titles from embeddings.

I.e. given some vector (or multiple) in skill vector space, find the best fitting job title.
The vector could be an average of all given skill vectors or this model could take all individual vectors and work with them (averaging will be easier for the start).

Options include: 
- From the training data just assign each job title its average vector space "position" and match new average vectors
- Cluster the job titles
- Use a simple neural network with an input layer of the size of the vector space
- Use a graph neural network