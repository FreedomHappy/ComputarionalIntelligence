import numpy
sizes = [1,2,3]
weights = [numpy.random.randn(y, x)
         for x, y in zip(sizes[:-1], sizes[1:])]
print(weights)

nabla_w = [numpy.zeros(w.shape) for w in weights]
print(nabla_w)