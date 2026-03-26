package kohonen;

public class neuron {

	int topology_x, topology_y;
	double[] weights;
	double distance;
	char letter;

	/** Constructor of the class */
	public neuron(int weight_size, int x, int y) {
		topology_x = x;
		topology_y = y;
		weights = new double[weight_size];
		for (int i = 0; i < weight_size; i++)
			weights[i] = Math.random();
	}

	/** Calculate total distance from neuron to each input */
	public void calculateDistance(double[] input) {
		distance = 0;
		for (int i = 0; i < weights.length; i++)
			distance = distance + Math.pow((input[i] - weights[i]), 2);
	}

	/** Adapts the weights of the existing grid neuron */
	public void adapt_weights(double[] input, double n, double h) {
		// Wij(t+1) = Wij(t) + n(t) * h(t) * ( Xi(t) - Wij(t) )
		for (int i = 0; i < weights.length; i++)
			weights[i] = weights[i] + n * h * (input[i] - weights[i]);
	}
}
