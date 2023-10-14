
/**
 * This class is a model of a neuron. A neuron is characterized by output, bias,
 * weights, previous bias and the layer to which it belongs. It does not have
 * any function inside. In our case it is used as an object from the
 * Neurons_Network class.
 * 
 * @author Elia Nicolaou 1012334 (enicol09)
 * @version 1.0
 * @see Neurons_Network,Layer
 */
public class Neuron {

	// the parameters
	String type;
	public double error;
	public double output;
	public double w_neurons[];
	public double ow_neurons[];
	public int layer_number;
	public double bias;
	public double bias_j;

	/**
	 * This is the constructor
	 * 
	 * @param belonging_layer - defines in which layer the neuron belongs
	 * @param weights         - defines how many neurons are connected to the
	 *                        neuron, how many weights must have.
	 */
	public Neuron(int belonging_layer, int weights) {

		layer_number = belonging_layer;

		switch (belonging_layer) {
		case 0:
			this.type = Layer.Input_Layer;
			break;
		case 1:
			this.type = Layer.Hidden_1_layer;
			break;
		case 2:
			if (Layer.hidden_2_neurons == 0) {
				this.type = Layer.Output_Layer;
			}

			else {
				this.type = Layer.Hidden_2_layer;
			}
			break;
		case 3:
			this.type = Layer.Output_Layer;
		}

		Initialize_weights(weights);
		Initialize_all();

	}

	/**
	 * Initialize the parameters.
	 */
	private void Initialize_all() {
		this.error = 0;
		this.bias = 0;
		this.bias_j = 0;

	}

	/**
	 * initialize the weights to 0.
	 * 
	 * @param weights
	 */
	private void Initialize_weights(int weights) {

		this.w_neurons = new double[weights];
		this.ow_neurons = new double[weights];
		for (int i = 0; i < weights; i++) {
			this.w_neurons[i] = 0;
			this.ow_neurons[i] = 0;
		}

	}

}
