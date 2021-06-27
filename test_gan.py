
from tensorflow.keras.models import load_model
from numpy import asarray
from matplotlib import pyplot
from GANCIFAR10 import generate_latent_points, generate_plot



# load model
model = load_model('generator_model_200.h5')
# generate images
latent_points = generate_latent_points(100, 100)
# generate images
X = model.predict(latent_points)
# scale from [-1,1] to [0,1]
X = (X + 1) / 2.0
# plot the result
generate_plot(X, 10)



# example of generating an image for a specific point in the latent space

# # load model
model = load_model('generator_model_200.h5')
# # all 0s
vector = asarray([[0.75 for _ in range(100)]])
# # generate image
X = model.predict(vector)
# # scale from [-1,1] to [0,1]
X = (X + 1) / 2.0
# # plot the result
pyplot.imshow(X[0, :, :])
pyplot.show()
