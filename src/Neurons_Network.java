/**
 * 
 * This class is the most important class in our program. It is essentially a
 * model of a neural network. Contains a table of neurons (type neurons), which
 * are characterized by their layer and the weights that connect each neuron to
 * the neurons of the next layer. The class works like a neural network.
 * Initially it initializes its neurons, giving them random values ​​as well as
 * the threshold of each neuron in the same way.
 * 
 * Then it called by Drive, it is trained and checked for specific inputs and
 * outputs -> training using Forward_propagation & Back_propagations (see
 * functions) -> testing using Forward_propagation only.
 * 
 * @author Elia Nicolaou 1012334 (enicol09)
 * 
 * @version 1.0
 * @see Neuron,Layer
 */
public class Neurons_Network {

	// class parameters
	int total_neurons = 0;
	int layers_number = 0;
	int neurons;
	Neuron my_neurons[];
	int Hidden_layer_2_neurons;
	int Input_neurons;
	int Output_neurons;
	static double learning_rate;
	static double momentum;
	int max_iterations;
	int Hidden_layer_1_neurons;
	int position;
	double weighted_sum;
	int layer;
	int neuron;
	int pos;
	boolean condition = true;

	/**
	 * This function is the constructor. The function calls three other functions 1.
	 * fill_my_neurons(); = create the neurons that the neurons-network has. 2.
	 * set_weigths_randomly(); = sets the weights of the neurons randomly 3.
	 * set_bias_randomly(); = sets the bias of the neurons randomly
	 */
	public Neurons_Network() {

		// initialize
		this.Hidden_layer_2_neurons = Driver.Hidden_layer_2_neurons;
		this.Input_neurons = Driver.Input_neurons;
		this.Output_neurons = Driver.Output_neurons;
		Neurons_Network.learning_rate = Driver.learning_rate;
		Neurons_Network.momentum = Driver.momentum;
		this.Hidden_layer_1_neurons = Driver.Hidden_layer_1_neurons;

		Layer.hidden_1_neurons = this.Hidden_layer_1_neurons;
		Layer.hidden_2_neurons = this.Hidden_layer_2_neurons;
		Layer.input_neurons = this.Input_neurons;
		Layer.output_neurons = this.Output_neurons;

		total_neurons = this.Hidden_layer_1_neurons + this.Hidden_layer_2_neurons + this.Input_neurons
				+ this.Output_neurons;

		my_neurons = new Neuron[total_neurons]; // create the file

		// call the functions
		fill_my_neurons();
		set_weights_randomly();
		set_bias_randomly();
	}

	/**
	 * This function is used for setting randomly the bias of the neurons.
	 */
	private void set_bias_randomly() {

		for (int i = 0; i < this.total_neurons; i++) {

			my_neurons[i].bias = (double) Math.random() * 1.0;
			if ((int) Math.random() * 2 == 1) {
				my_neurons[i].bias *= (-1.0);
			}
		}

	}

	/**
	 * This function is used for setting randomly the weights of the neurons.
	 */
	private void set_weights_randomly() {

		for (int i = 0; i < this.total_neurons; i++) {
			for (int j = 0; j < my_neurons[i].w_neurons.length; i++) {
				my_neurons[i].w_neurons[j] = (double) Math.random() * 1.0;
				if ((int) Math.random() * 2 == 1) {
					my_neurons[i].w_neurons[j] = my_neurons[i].w_neurons[j] * (-1.0);
				}
			}
		}
	}

	/**
	 * This function is used for creating the neurons - specific number of
	 * input,hidden,output
	 */
	private void fill_my_neurons() {
		layers_number = 0;

		for (int i = 0; i < Layer.input_neurons; i++) {
			this.my_neurons[neurons] = new Neuron(layers_number, this.Hidden_layer_1_neurons);
			neurons++;
		}

		layers_number++;

		if (!Are_they_two_hidden()) {

			for (int i = 0; i < Layer.hidden_1_neurons; i++) {
				this.my_neurons[neurons] = new Neuron(layers_number, this.Output_neurons);
				neurons++;
			}

			layers_number++;
			for (int i = 0; i < Layer.output_neurons; i++) {
				this.my_neurons[neurons] = new Neuron(layers_number, 0);
				neurons++;
			}
		} else {

			for (int i = 0; i < Layer.hidden_1_neurons; i++) {
				this.my_neurons[neurons] = new Neuron(layers_number, this.Hidden_layer_2_neurons);
				neurons++;
			}

			layers_number++;
			for (int i = 0; i < Layer.hidden_2_neurons; i++) {
				this.my_neurons[neurons] = new Neuron(layers_number, this.Output_neurons);
				neurons++;
			}
			layers_number++;
			for (int i = 0; i < Layer.output_neurons; i++) {
				this.my_neurons[neurons] = new Neuron(layers_number, 0);
				neurons++;
			}
		}
	}

	/**
	 * This functions check if they are 2 hidden layers
	 * 
	 * @return true if there are and false if there are not.
	 */
	private boolean Are_they_two_hidden() {
		if (Layer.hidden_2_neurons == 0)
			return false;
		return true;
	}

	/**
	 * Is used for calculating the weighted sum
	 * 
	 * @param neuron
	 * @param position
	 * @return
	 */
	public static double calculate_sum(Neuron neuron, int position) {
		double x = neuron.w_neurons[position] * neuron.output;
		return x;
	}

	/**
	 * This function is used for putting the output into the sigmoid function
	 * 
	 * @param x - the output
	 * @return the output through the sigmoid function
	 */
	private double sigmoid(double x) {
		return 1.0 / (1.0 + (Math.exp(-x)));
	}

	/**
	 * This is the first part of the backpropagation algorithm - calculating the
	 * weighted_sum - output - and the calling the sigmoid function
	 * 
	 * @param testing_data_2
	 * @param testing_data
	 */
	public void Forward_Propagation(Double testing_data, Double testing_data_2) {

		setInput(testing_data, testing_data_2);
		condition = true;
		weighted_sum = 0;
		layer = 1;
		position = 0;
		while (condition) {
			position = 0;
			for (neuron = 0; neuron < total_neurons; neuron++) {
				if (my_neurons[neuron].layer_number == layer) {
					for (pos = 0; pos < neuron; pos++) {
						if (my_neurons[pos].layer_number == (layer - 1)) {
							weighted_sum += calculate_sum(my_neurons[pos], position);
						}

					}
					my_neurons[neuron].output = sigmoid(weighted_sum + my_neurons[neuron].bias);
					weighted_sum = 0;
					position++;
				}
			}
			layer++;
			if (layer > layers_number)
				condition = false;
		}
	}

	/**
	 * This function is used for calculating the derative of the gradient
	 * 
	 * @param target - desirable targer
	 * @param neuron - neuron to check
	 * 
	 * @return x
	 */
	public static double calculate_derative(double target, Neuron neuron) {
		double x = neuron.output * (1 - neuron.output) * (neuron.output - target);
		return x;
	}

	/**
	 * Calculate the error part 2
	 * 
	 * @param neuron - to change the error for
	 * 
	 * @return new changed error
	 */
	private double calculate_error2(Neuron neuron) {
		double x = neuron.output * (1 - neuron.output);
		return x;
	}

	/**
	 * Calculating the error
	 * 
	 * @param neuron     - to calculate the error for
	 * @param position   - position of the weight
	 * @param neuron_bef - the neuron before
	 * 
	 * @return new changed error
	 */
	public static double error_calculation(Neuron neuron, int position, Neuron neuron_bef) {
		double x = neuron.w_neurons[position] * neuron_bef.error;

		return x;
	}

	/**
	 * 
	 * This function is used in the back_propagation for changing the weights.
	 * 
	 * @param neuron_i - first neuron
	 * @param neuron_j - second neuron
	 * @param position - the position of the weight - 0 or 1
	 * @return the changed weight
	 */
	private double changed_weight(Neuron neuron_i, Neuron neuron_j, int position) {
		double x = neuron_j.w_neurons[position] - learning_rate * neuron_i.error * neuron_j.output
				+ momentum * (neuron_j.w_neurons[position] - neuron_j.ow_neurons[position]);
		return x;
	}

	/**
	 * This function is used in the back_propagation for changing the bias.
	 * 
	 * @param neuron - the neuron to change the bias.
	 * @return the changed bias
	 */
	private double changed_bias(Neuron neuron) {
		double x = neuron.bias - learning_rate * neuron.error + momentum * (neuron.bias - neuron.bias_j);
		return x;
	}

	/**
	 * This function is used by the Driver in order to initialize the input layer
	 * neurons with the training/testing values
	 * 
	 * @param input1 - the first input
	 * @param input2 - the second input
	 */
	public void setInput(Double input1, Double input2) {
		my_neurons[0].output = input1;
		my_neurons[1].output = input2;

	}

	/**
	 * This is the second part of the back_propagation algorithm Backpropagation :
	 * 
	 * The backpropagation algorithm works by computing the gradient of the loss
	 * function with respect to each weight by the chain rule, computing the
	 * gradient one layer at a time, iterating backward from the last layer to avoid
	 * redundant calculations of intermediate terms in the chain rule; this is an
	 * example of dynamic programming.
	 * 
	 * @param target = desirable target
	 */
	public void Back_propagation(double target) {

		for (int i = 0; i < total_neurons; i++) {
			if (my_neurons[i].layer_number == layers_number) {
				my_neurons[i].error = calculate_derative(target, my_neurons[i]);
			}
		}

		layer = 1;
		double error;
		condition = true;
		position = 0;
		while (condition) {
			for (int i = total_neurons - 1; i >= 0; i--)
				if (my_neurons[i].layer_number == layer) {
					error = 0;
					for (neuron = total_neurons - 1; neuron > i; neuron--)
						if (my_neurons[neuron].layer_number == (layer + 1)) {
							error += error_calculation(my_neurons[i], position, my_neurons[neuron]);
							position++;
						}
					my_neurons[i].error = error * calculate_error2(my_neurons[i]);
					position = 0;
				}

			layer--;
			if (layer <= 0)
				condition = false;
		}

		// changing the bias
		for (int i = 0; i < total_neurons; i++) {
			double save = my_neurons[i].bias;
			my_neurons[i].bias = changed_bias(my_neurons[i]);
			my_neurons[i].bias_j = save;
		}

		// set_weights
		layer = 1;
		int i;
		condition = true;
		while (condition) {
			position = 0;
			for (i = 0; i < this.total_neurons; i++)
				if (my_neurons[i].layer_number == layer) {
					for (neuron = 0; neuron < i; neuron++)
						if (this.my_neurons[neuron].layer_number == (layer - 1)) {
							double save = my_neurons[neuron].w_neurons[position];
							// changing the weights
							my_neurons[neuron].w_neurons[position] = changed_weight(my_neurons[i], my_neurons[neuron],
									position);
							my_neurons[neuron].ow_neurons[position] = save;
						}
					position++;
				}
			layer++;
			if (layer > layers_number)
				condition = false;
		}

	}

}
