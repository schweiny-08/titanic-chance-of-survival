import matplotlib.pyplot as plot
import math

epochs = 100

test = [ int(i+1) for i in range(epochs)]
good_facts = [1,4,2,6,9,1,4,2,6,9,1,4,2,6,9,1,4,2,6,9,1,4,2,6,9,1,4,2,6,9,1,4,2,6,9,1,4,2,6,9,1,4,2,6,9,1,4,2,6,9,1,4,2,6,9,1,4,2,6,9,1,4,2,6,9,1,4,2,6,9,1,4,2,6,9,1,4,2,6,9,1,4,2,6,9,1,4,2,6,9,1,4,2,6,9,1,4,2,6,9]

plot.plot(test, good_facts)

plot.xlabel("Epochs")
plot.ylabel("Good Facts")
plot.title("Training Phase")

plot.savefig('TestingGraph.png')


plot.show()
